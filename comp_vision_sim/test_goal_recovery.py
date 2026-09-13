"""Tests for speed and for recovering a lost goal.

    ../.venv/bin/python -m unittest -v test_goal_recovery

Speed:
  * thinking is switched off the way llama-server actually honours
    (chat_template_kwargs), which took a frame query from 17.2 s to 4.8 s
  * with --track depth, a goal the survey already ranged is confirmed from
    the depth image, so the LLM is not queried frame by frame

Recovery:
  * a sighting far from the current goal moves the goal there at once
  * a goal that should be in view but is not seen for `lost_after` seconds
    triggers the search again from wherever the robot is (a new survey when
    one is configured, otherwise a spin scan)
  * no route to the goal re-searches too; re-searches are capped and cooled
    down, and running out ends in STUCK rather than a false ARRIVED

Offline except `TestLiveSpeed`, which is skipped when the server is down (or
SKIP_LIVE_LLM is set).
"""
from __future__ import annotations

import json
import os
import sys
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "main_mujoco"))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from test_llm_scene_reasoning import (GOAL_BODY, LLM_BASE, FakeResponse,  # noqa: E402
                                      _server_up, goal_world, make_bot, place)
from test_llm_survey import OracleSurvey, shots_from  # noqa: E402
from vision_sim import llm_reasoner as L  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim.llm_reasoner import LlmGoalDetector  # noqa: E402
from vision_sim.llm_survey import LlmSurvey  # noqa: E402
from vision_sim.navigation import VisualNavigator  # noqa: E402


# --------------------------------------------------------------------- helpers
def move_goal(bot, x, y, z=None):
    """Teleport the (static) goal column. Static bodies take their pose from
    the model, so change it there and refresh kinematics."""
    bid = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, GOAL_BODY)
    pos = bot.model.body_pos[bid].copy()
    pos[0], pos[1] = x, y
    if z is not None:
        pos[2] = z
    bot.model.body_pos[bid] = pos
    mujoco.mj_forward(bot.model, bot.data)


class LiveGoalSurvey(OracleSurvey):
    """Oracle survey that looks the goal up at query time, so it sees a goal
    that has been moved since the navigator was built."""

    def __init__(self, bot, **kw):
        super().__init__(goal_world(bot), **kw)
        self.bot = bot

    def query(self, shots):
        self.goal = goal_world(self.bot)
        return super().query(shots)


def step_until(bot, nav, cond, limit):
    while bot.time < limit and not bot.fallen and not cond():
        bot.step(0.1, controller=nav)


def run_to_done(bot, nav, limit, record=None):
    while bot.time < limit and not bot.fallen and not nav.done:
        bot.step(0.1, controller=nav)
        if record is not None:
            record.append(nav.state)


def reply(content='{"goal_found": false}'):
    return FakeResponse({"choices": [{"message": {"content": content},
                                      "finish_reason": "stop"}],
                         "usage": {"completion_tokens": 5}})


# ======================================================================= speed
class TestThinkingSwitch(unittest.TestCase):
    def body_for(self, client, multi=False):
        with mock.patch.object(L.requests, "post", return_value=reply()) as post:
            img = np.zeros((4, 4, 3), np.uint8)
            if multi:
                client.ask_images([img, img], ["a", "b"], "PROMPT")
            else:
                client.ask_image(img, "PROMPT")
        return post.call_args[1]["json"]

    def test_off_by_default_and_sent_where_the_server_reads_it(self):
        # Regression: only the top-level key was sent, which llama-server
        # ignores; the model kept reasoning and queries took ~4x longer.
        body = self.body_for(L.LLMClient())
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertIs(body["enable_thinking"], False)
        self.assertTrue(body["messages"][0]["content"][0]["text"].endswith("/no_think"))

    def test_multi_image_requests_too(self):
        body = self.body_for(L.LLMClient(), multi=True)
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertTrue(body["messages"][0]["content"][0]["text"].endswith("/no_think"))

    def test_on_when_asked(self):
        body = self.body_for(L.LLMClient(thinking=True))
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": True})
        self.assertIs(body["enable_thinking"], True)
        self.assertEqual(body["messages"][0]["content"][0]["text"], "PROMPT")

    def test_detector_and_survey_pass_it_through(self):
        self.assertFalse(LlmGoalDetector().client.thinking)
        self.assertTrue(LlmGoalDetector(thinking=True).client.thinking)
        self.assertFalse(LlmSurvey().client.thinking)
        self.assertTrue(LlmSurvey(thinking=True).client.thinking)


class TestDetectorForget(unittest.TestCase):
    def test_forget_drops_the_cache_and_queries_at_once(self):
        det = LlmGoalDetector(query_period=3.0)
        calls = []

        def ask(rgb, prompt):
            calls.append(1)
            return '{"goal_found": false}', 0.0
        det.client.ask_image = ask

        class Obs:
            class intrinsics:
                width, height = 320, 240
            rgb = np.zeros((240, 320, 3), np.uint8)

        det(Obs(), t=0.0)
        det._cache = ["stale sighting"]
        self.assertEqual(det(Obs(), t=1.0), ["stale sighting"])   # gated, cached
        det.forget()
        self.assertEqual(det._cache, [])
        det(Obs(), t=1.1)
        self.assertEqual(len(calls), 2, "forget() must make the next call query")


# ==================================================================== geometry
class TestGoalVisibility(unittest.TestCase):
    """_expect_visible: should the camera see the goal? _present: does depth
    show something standing there?"""

    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.goal = goal_world(cls.bot)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def nav_at(self, x, y, yaw, goal_xy=None):
        place(self.bot, x, y, yaw)
        nav = VisualNavigator()
        nav.goal_xy = np.asarray(goal_xy if goal_xy is not None else self.goal[:2], float)
        obs = perception.observe(self.bot, width=320, height=240, max_range=12.0)
        return nav, obs

    def test_in_view_ahead(self):
        nav, obs = self.nav_at(4.0, 0.0, 0.0)
        self.assertTrue(nav._expect_visible(obs))
        self.assertTrue(nav._present(obs))

    def test_behind_the_robot(self):
        nav, obs = self.nav_at(4.0, 0.0, np.pi)
        self.assertFalse(nav._expect_visible(obs))

    def test_off_to_the_side(self):
        nav, obs = self.nav_at(4.0, 0.0, np.pi / 2)
        self.assertFalse(nav._expect_visible(obs))

    def test_beyond_sighting_range(self):
        nav, obs = self.nav_at(0.0, 0.0, 0.0)
        nav.sighting_range = 3.0
        self.assertFalse(nav._expect_visible(obs))

    def test_occluded_by_something_nearer(self):
        # Pretend the goal is far behind the goal column itself: the column
        # is in front of that spot, so not seeing it proves nothing.
        nav, obs = self.nav_at(4.0, 0.0, 0.0, goal_xy=(9.0, 0.0))
        nav.sighting_range = 12.0
        self.assertFalse(nav._expect_visible(obs))

    def test_empty_floor_is_not_present(self):
        # A patch of open floor well inside the view, clear of the column
        # (0.76 m away, beyond presence_radius): visible, but nothing stands there.
        nav, obs = self.nav_at(4.0, 0.0, 0.0, goal_xy=(5.3, 0.7))
        self.assertTrue(nav._expect_visible(obs))
        self.assertFalse(nav._present(obs))

    def test_bad_track_value(self):
        with self.assertRaises(ValueError):
            VisualNavigator(track="vibes")


# ================================================================== recovery
class TestRecoveryLogic(unittest.TestCase):
    """The decision rules, driven directly."""

    def setUp(self):
        self.bot = make_bot()

    def tearDown(self):
        self.bot.close()

    def test_cooldown_blocks_an_immediate_second_search(self):
        nav = VisualNavigator(lost_after=2.0, research_cooldown=5.0)
        nav._search_done_at = 10.0
        nav._miss_since = 10.0
        self.assertFalse(nav._goal_lost(13.0))     # missed 3 s, but cooling down
        self.assertTrue(nav._goal_lost(15.5))

    def test_research_resets_and_counts(self):
        nav = VisualNavigator(survey=OracleSurvey(goal_world(self.bot)), max_researches=2)
        nav.goal_xy = np.array([6.0, 0.0])
        nav.state = nav.NAVIGATE
        nav.path = [np.array([1.0, 0.0])]
        forgets = []
        nav.detector = mock.Mock(side_effect=lambda *a, **k: [], forget=lambda: forgets.append(1))
        nav._research(self.bot, 5.0, "test")
        self.assertEqual((nav.state, nav.researches, nav.goal_xy, nav.path), (nav.SURVEY, 1, None, []))
        self.assertIsNone(nav._survey_plan)
        self.assertEqual(forgets, [1], "the detector's cached sighting must be dropped")

    def test_without_a_survey_research_is_a_scan(self):
        nav = VisualNavigator()
        nav.goal_xy, nav.state = np.array([6.0, 0.0]), VisualNavigator.NAVIGATE
        nav._research(self.bot, 5.0, "test")
        self.assertEqual(nav.state, nav.SCAN)

    def test_running_out_of_researches_is_stuck(self):
        nav = VisualNavigator(max_researches=1)
        nav.goal_xy, nav.state = np.array([6.0, 0.0]), VisualNavigator.NAVIGATE
        nav._research(self.bot, 1.0, "first")
        nav.goal_xy, nav.state = np.array([6.0, 0.0]), VisualNavigator.NAVIGATE
        nav._research(self.bot, 9.0, "second")
        self.assertEqual((nav.state, nav.researches), (nav.STUCK, 1))
        self.assertTrue(any("giving up" in m for m in nav.log))

    def test_no_route_re_searches_instead_of_stuck(self):
        nav = VisualNavigator()
        nav.goal_xy, nav.state = np.array([6.0, 0.0]), VisualNavigator.NAVIGATE
        with mock.patch("vision_sim.navigation.planning.astar", return_value=None):
            for _ in range(8):
                nav.replan(self.bot)
        self.assertEqual((nav.state, nav.researches), (nav.SCAN, 1))

    def test_fixed_coordinate_goal_with_no_route_is_stuck(self):
        nav = VisualNavigator(goal=(6.0, 0.0))
        nav.goal_xy, nav.state = np.array([6.0, 0.0]), VisualNavigator.NAVIGATE
        with mock.patch("vision_sim.navigation.planning.astar", return_value=None):
            for _ in range(8):
                nav.replan(self.bot)
        self.assertEqual((nav.state, nav.researches), (nav.STUCK, 0))

    def test_clock_reset_clears_a_pending_miss(self):
        nav = VisualNavigator()
        nav._last_t, nav._miss_since, nav._search_done_at = 50.0, 45.0, 40.0
        self.bot.balance.enable(self.bot.state)
        nav(self.bot, 0.1)                           # time ran backwards
        self.assertIsNone(nav._search_done_at)
        self.assertFalse(nav._goal_lost(0.2))


# =============================================================== end to end
class TestRecoveryEndToEnd(unittest.TestCase):
    def test_normal_runs_never_re_search(self):
        for label, kw in (("colour", {}),
                          ("survey + depth tracking", {"track": "depth"})):
            with self.subTest(label):
                bot = make_bot()
                try:
                    bot.balance.enable(bot.state)
                    if kw:
                        kw = dict(kw, survey=OracleSurvey(goal_world(bot)))
                    nav = VisualNavigator(**kw)
                    run_to_done(bot, nav, 90.0)
                    self.assertEqual(nav.state, nav.ARRIVED)
                    self.assertEqual(nav.researches, 0, nav.log)
                finally:
                    bot.close()

    def test_moved_goal_is_followed_by_the_detector(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator()
            step_until(bot, nav, lambda: nav.state == nav.NAVIGATE and bot.position[0] > 1.0, 60.0)
            self.assertEqual(nav.state, nav.NAVIGATE)
            move_goal(bot, 6.0, 1.3)
            run_to_done(bot, nav, 120.0)
            new_goal = goal_world(bot)
            self.assertEqual(nav.state, nav.ARRIVED, nav.log[-5:])
            self.assertLess(np.linalg.norm(bot.position[:2] - new_goal[:2]), nav.stop_distance + 0.5)
            self.assertTrue(any("moving the goal there" in m for m in nav.log), nav.log)
        finally:
            bot.close()

    def test_moved_goal_triggers_a_second_survey(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            survey = LiveGoalSurvey(bot)
            detector_calls = []

            def detector(obs, **kw):
                detector_calls.append(nav.state)
                return perception.detect(obs, **kw)

            nav = VisualNavigator(survey=survey, track="depth", detector=detector)
            step_until(bot, nav, lambda: nav.state == nav.NAVIGATE and bot.position[0] > 1.5, 60.0)
            self.assertEqual(nav.state, nav.NAVIGATE)
            self.assertEqual(survey.calls, 1)
            move_goal(bot, 4.0, 2.6)                 # out of the robot's current view
            run_to_done(bot, nav, 150.0)

            new_goal = goal_world(bot)
            self.assertEqual(survey.calls, 2, f"expected a second survey; log: {nav.log}")
            self.assertEqual(nav.researches, 1)
            self.assertEqual(nav.state, nav.ARRIVED, nav.log[-6:])
            self.assertLess(np.linalg.norm(bot.position[:2] - new_goal[:2]), nav.stop_distance + 0.5)
            # speed: depth tracking never consulted the per-frame detector
            self.assertNotIn(nav.NAVIGATE, detector_calls)
        finally:
            bot.close()

    def test_vanished_goal_re_searches_then_gives_up(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator(max_researches=2, lost_after=3.0)
            step_until(bot, nav, lambda: nav.state == nav.NAVIGATE and bot.position[0] > 1.0, 60.0)
            move_goal(bot, 6.0, 0.0, z=-10.0)        # gone
            states = []
            run_to_done(bot, nav, 150.0, states)
            self.assertNotIn(nav.ARRIVED, states, "must not arrive at an empty spot")
            self.assertEqual(nav.researches, 2)
            self.assertEqual(nav.state, nav.STUCK)
            self.assertTrue(any("re-running the search" in m for m in nav.log))
        finally:
            bot.close()


# ========================================================================= CLI
class TestRecoveryCommandLine(unittest.TestCase):
    def test_defaults(self):
        import run_navigation as rn
        a = rn.parse_args([])
        self.assertEqual((a.llm_thinking, a.track, a.lost_after, a.max_researches),
                         (False, None, 4.0, 3))
        self.assertEqual(rn.resolve_track(a), "detector")
        self.assertEqual(rn.resolve_track(rn.parse_args(["--survey"])), "depth")
        self.assertEqual(rn.resolve_track(rn.parse_args(["--survey", "--track", "detector"])),
                         "detector")

    def test_thinking_flag_reaches_detector_and_survey(self):
        import run_navigation as rn
        a = rn.parse_args(["--detector", "llm", "--survey", "--llm-thinking"])
        with mock.patch("builtins.print"):
            self.assertTrue(rn.build_detector(a).client.thinking)
            self.assertTrue(rn.build_survey(a).client.thinking)


# ======================================================================== live
@unittest.skipIf(os.environ.get("SKIP_LIVE_LLM"), "SKIP_LIVE_LLM is set")
@unittest.skipUnless(_server_up(LLM_BASE), f"no LLM server at {LLM_BASE}")
class TestLiveSpeed(unittest.TestCase):
    """Guards the thinking switch on the real server: fast, and no reasoning."""

    def test_frame_query_is_fast_and_does_not_reason(self):
        bot = make_bot()
        try:
            place(bot, 0.0, 0.0, 0.0)
            obs = perception.observe(bot, width=320, height=240)
            det = LlmGoalDetector(base_url=LLM_BASE.rstrip("/") + "/v1")
            t0 = time.time()
            dets = det(obs, robot_yaw=bot.yaw, t=0.0)
            wall = time.time() - t0
            self.assertEqual(det.client.last_reasoning_chars, 0)
            self.assertLess(wall, 10.0, f"frame query took {wall:.1f}s")
            self.assertTrue(dets, det.last_reasoning)
            print(f"\n    live frame: {wall:.1f}s, no reasoning", file=sys.stderr)
        finally:
            bot.close()

    def test_survey_query_is_fast_and_does_not_reason(self):
        bot = make_bot()
        try:
            shots = shots_from(bot, 0.0, 0.0, 0.3)
            sv = LlmSurvey(base_url=LLM_BASE.rstrip("/") + "/v1")
            t0 = time.time()
            res = sv.query(shots)
            wall = time.time() - t0
            self.assertIsNone(res.error, res.error)
            self.assertEqual(sv.client.last_reasoning_chars, 0)
            self.assertLess(wall, 20.0, f"survey took {wall:.1f}s")
            self.assertTrue(res.found, res.reason)
            print(f"\n    live survey: {wall:.1f}s, no reasoning", file=sys.stderr)
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
