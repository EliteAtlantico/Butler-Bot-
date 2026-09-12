"""Tests for YOLO search with LLM-chosen waypoints (--explore).

    ../.venv/bin/python -m unittest -v test_explore

The robot surveys, the fast detector (pretrained open-vocabulary YOLO) checks every photo,
and when it sees nothing the local LLM picks where to go next. The robot drives
there, surveys again, and repeats until the detector finds the object; then it
navigates to it and stops at the stand-off distance, ready for a pick.

Scene: search_course.xml, where the target is hidden behind a wall from the
start and the only way through is a 1.2 m doorway.

Offline except `TestExploreLive`, skipped when the server is down or
SKIP_LIVE_LLM is set.
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

import numpy as np  # noqa: E402

from test_llm_scene_reasoning import (LLM_BASE, FakeResponse, _server_up,  # noqa: E402
                                      place, project, visible_pixel)
from test_goal_recovery import move_goal  # noqa: E402
from vision_sim import llm_reasoner as L  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim.llm_explore import (LlmExplorer, fallback_waypoint,  # noqa: E402
                                    free_distance, waypoint_from_pixel)
from vision_sim.llm_survey import SurveyShot, pixel_heading  # noqa: E402
from vision_sim.navigation import VisualNavigator  # noqa: E402

SEARCH = os.path.join(HERE, "search_course.xml")
COURSE = os.path.join(HERE, "obstacle_course.xml")
DOOR = np.array([3.0, 2.0])          # centre of the doorway in search_course.xml
CEILING = 1.73


def make_bot(scene=SEARCH):
    from bracketbot_sim.robot import BracketBot
    return BracketBot(xml=scene)


def goal_xyz(bot):
    return np.r_[bot.body_position("target_column")[:2], 0.6]


def shots_at(bot, x, y, yaw0=0.0, n=8):
    ex = LlmExplorer(n_shots=n)
    out = []
    for k, yaw in enumerate(ex.headings(yaw0)):
        place(bot, x, y, yaw)
        out.append(SurveyShot(k, yaw, np.array([x, y], float),
                              perception.observe(bot, width=320, height=240)))
    return out


def box_around(u, v, w=320, h=240, half=15):
    return [(u - half) / w * 1000, (v - half) / h * 1000,
            (u + half) / w * 1000, (v + half) / h * 1000]


_YOLO = None


def yolo():
    """The pretrained open-vocabulary YOLO looking for the red cylinder, loaded once."""
    global _YOLO
    if _YOLO is None:
        from vision_sim.yolo_detector import YoloDetector
        _YOLO = YoloDetector(target="red cylinder")
    return _YOLO


class OracleExplorer(LlmExplorer):
    """A sensible stand-in for the model: reports the goal when it is visible
    in a photo, otherwise boxes the doorway (or, once the robot is at the
    doorway, a spot beyond it). The answer goes through the REAL interpret(),
    so the waypoint geometry under test is the production code."""

    BEYOND = np.array([4.4, 1.9])

    def __init__(self, bot, **kw):
        kw.setdefault("ceiling", CEILING)
        super().__init__(**kw)
        self.bot = bot
        self.calls = 0

    def query(self, shots, visited=()):
        self.calls += 1
        goal = goal_xyz(self.bot)
        for s in shots:
            px = visible_pixel(s.obs, goal)
            if px is not None:
                raw = {"goal_found": True, "goal_photo": s.index,
                       "goal_bbox_2d": box_around(*px), "confidence": 0.9, "reason": "oracle goal"}
                res = self.interpret(shots, raw, visited)
                self.queries += 1
                return self._finish(res)
        here = shots[0].robot_xy
        aim = DOOR if np.linalg.norm(here - DOOR) > 1.0 else self.BEYOND
        best = None
        for s in shots:
            pr = project(s.obs, np.r_[aim, 0.05])
            if pr is None:
                continue
            u, v, _ = pr
            if 0 <= u < 320 and 0 <= v < 240 and (best is None or abs(u - 160) < abs(best[1] - 160)):
                best = (s.index, u, v)
        raw = ({"goal_found": False, "explore_photo": best[0],
                "explore_bbox_2d": box_around(best[1], best[2]), "confidence": 0.7,
                "reason": "oracle doorway"} if best else {"goal_found": False})
        res = self.interpret(shots, raw, visited)
        self.queries += 1
        return self._finish(res)


def run(bot, nav, limit, states=None):
    while bot.time < limit and not bot.fallen and not nav.done:
        bot.step(0.1, controller=nav)
        if states is not None and (not states or states[-1] != nav.state):
            states.append(nav.state)


# ================================================================== geometry
class TestExploreGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.shots = shots_at(cls.bot, 0.0, 0.0)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def test_free_distance_sees_the_wall_and_the_open_room(self):
        ahead = free_distance(self.shots[0].obs, 160, self.shots[0].robot_xy, ceiling=CEILING)
        along = free_distance(self.shots[2].obs, 160, self.shots[2].robot_xy, ceiling=CEILING)
        self.assertAlmostEqual(ahead, 2.9, delta=0.3)       # the divider is at x = 3.0
        self.assertGreater(along, 5.0)                      # open along the wall

    def test_doorway_pixel_becomes_a_doorway_waypoint(self):
        s = self.shots[1]
        u, v, _ = project(s.obs, np.r_[DOOR, 0.3])
        wp, heading = waypoint_from_pixel(s, u, v, ceiling=CEILING)
        self.assertIsNotNone(wp)
        self.assertLess(np.linalg.norm(wp - DOOR), 0.5)
        self.assertAlmostEqual(heading, pixel_heading(s, u))

    def test_a_waypoint_is_never_placed_into_a_near_wall(self):
        place(self.bot, 2.0, -1.0, 0.0)               # 1.0 m from the divider, facing it
        obs = perception.observe(self.bot, width=320, height=240)
        shot = SurveyShot(0, 0.0, np.array([2.0, -1.0]), obs)
        wp, _ = waypoint_from_pixel(shot, 160, 150, ceiling=CEILING)
        self.assertIsNone(wp, "0.3 m of open floor minus the margin is not a waypoint")

    def test_fallback_is_open_and_avoids_visited_places(self):
        wp, heading = fallback_waypoint(self.shots, [], ceiling=CEILING)
        self.assertIsNotNone(wp)
        wp2, _ = fallback_waypoint(self.shots, [wp], ceiling=CEILING)
        self.assertGreater(np.linalg.norm(wp2 - wp), 1.0)


# ================================================================= interpret
class TestExploreInterpret(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.goal = goal_xyz(cls.bot)
        cls.start = shots_at(cls.bot, 0.0, 0.0)
        cls.door = shots_at(cls.bot, 3.7, 2.0, -0.9)
        cls.ex = LlmExplorer(ceiling=CEILING)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def test_explore_box_on_the_doorway(self):
        u, v, _ = project(self.start[1].obs, np.r_[DOOR, 0.3])
        r = self.ex.interpret(self.start, {"goal_found": False, "explore_photo": 1,
                                           "explore_bbox_2d": box_around(u, v)})
        self.assertFalse(r.found)
        self.assertEqual(r.waypoint_source, "model")
        self.assertLess(np.linalg.norm(r.waypoint - DOOR), 0.5)

    def test_goal_box_is_ranged(self):
        view = [(s, visible_pixel(s.obs, self.goal)) for s in self.door]
        s, px = next((s, px) for s, px in view if px is not None)
        r = self.ex.interpret(self.door, {"goal_found": True, "goal_photo": s.index,
                                          "goal_bbox_2d": box_around(*px), "confidence": 0.9})
        self.assertTrue(r.found)
        self.assertIsNone(r.waypoint)
        self.assertLess(np.linalg.norm(r.goal.detection.position[:2] - self.goal[:2]), 0.5)

    def test_unusable_answers_fall_back(self):
        for name, raw in {"no photo": {"goal_found": False},
                          "bad photo": {"goal_found": False, "explore_photo": 42},
                          "no box": {"goal_found": False, "explore_photo": 1},
                          "bad box": {"goal_found": False, "explore_photo": 1,
                                      "explore_bbox_2d": "doorway"}}.items():
            with self.subTest(name):
                r = self.ex.interpret(self.start, raw)
                self.assertEqual(r.waypoint_source, "fallback")
                self.assertIsNotNone(r.waypoint)

    def test_already_explored_target_falls_back(self):
        u, v, _ = project(self.start[1].obs, np.r_[DOOR, 0.3])
        r = self.ex.interpret(self.start, {"goal_found": False, "explore_photo": 1,
                                           "explore_bbox_2d": box_around(u, v)},
                              visited=[DOOR])
        self.assertEqual(r.waypoint_source, "fallback")
        self.assertGreater(np.linalg.norm(r.waypoint - DOOR), 1.0)

    def test_goal_claimed_but_not_locatable_keeps_exploring(self):
        r = self.ex.interpret(self.start, {"goal_found": True, "goal_photo": None,
                                           "goal_bbox_2d": None, "explore_photo": None})
        self.assertFalse(r.found)
        self.assertIsNotNone(r.waypoint)
        self.assertTrue(any("could not be located" in n for n in r.notes))

    def test_prompt_lists_explored_places(self):
        self.assertIn("Nothing has been explored yet", self.ex.prompt(self.start, []))
        p = self.ex.prompt(self.start, [np.array([0.0, 0.0]), np.array([2.1, 1.6])])
        self.assertIn("(0.0, 0.0), (2.1, 1.6)", p)
        self.assertIn("explore_bbox_2d", p)
        self.assertIn("0-1000", p)


class TestExploreRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.shots = shots_at(cls.bot, 0.0, 0.0)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def reply(self, content):
        return FakeResponse({"choices": [{"message": {"content": content},
                                          "finish_reason": "stop"}],
                             "usage": {"completion_tokens": 9}})

    def test_one_request_every_photo_labelled(self):
        ex = LlmExplorer(ceiling=CEILING)
        with mock.patch.object(L.requests, "post",
                               return_value=self.reply('{"goal_found": false}')) as post:
            ex.query(self.shots, visited=[(0.0, 0.0)])
        body = post.call_args[1]["json"]
        parts = body["messages"][0]["content"]
        self.assertEqual(sum(p["type"] == "image_url" for p in parts), 8)
        for i, s in enumerate(self.shots):
            self.assertEqual(parts[1 + 2 * i]["text"], s.label(self.shots[0].heading))
        self.assertIn("(0.0, 0.0)", parts[0]["text"])
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})

    def test_server_down_still_gives_a_waypoint(self):
        ex = LlmExplorer(ceiling=CEILING)
        with mock.patch.object(L.requests, "post", side_effect=ConnectionError("down")):
            r = ex.query(self.shots)
        self.assertIn("ConnectionError", r.error)
        self.assertEqual((r.waypoint_source, ex.errors), ("fallback", 1))

    def test_unparseable_answer_still_gives_a_waypoint(self):
        ex = LlmExplorer(ceiling=CEILING)
        with mock.patch.object(L.requests, "post", return_value=self.reply("go left I guess")):
            r = ex.query(self.shots)
        self.assertEqual(r.waypoint_source, "fallback")


# ================================================================ navigation
class TestExploreNavigation(unittest.TestCase):
    def test_hidden_goal_found_via_llm_waypoints_and_reached(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            explorer = OracleExplorer(bot)
            nav = VisualNavigator.for_bot(bot, explorer=explorer, detector=yolo(), track="depth")
            self.assertEqual(nav.state, nav.SURVEY)
            states = []
            run(bot, nav, 240.0, states)
            goal = goal_xyz(bot)
            dist = float(np.linalg.norm(bot.position[:2] - goal[:2]))
            self.assertFalse(bot.fallen)
            self.assertEqual(nav.state, nav.ARRIVED, f"ended {nav.state} {dist:.2f} m away; {nav.log[-6:]}")
            self.assertLess(dist, nav.stop_distance + 0.5)
            self.assertGreaterEqual(explorer.calls, 1, "the LLM should have chosen at least one waypoint")
            self.assertIn(nav.EXPLORE, states)
            self.assertTrue(any("while exploring" in m or "detector found the goal" in m
                                for m in nav.log), nav.log)
        finally:
            bot.close()

    def test_detector_seeing_the_goal_in_the_survey_skips_the_model(self):
        bot = make_bot(COURSE)                  # target visible from the start here
        try:
            bot.balance.enable(bot.state)
            explorer = OracleExplorer(bot)
            nav = VisualNavigator.for_bot(bot, explorer=explorer, detector=yolo(), track="depth")
            run(bot, nav, 90.0)
            self.assertEqual(explorer.calls, 0)
            self.assertEqual(nav.state, nav.ARRIVED)
            self.assertTrue(any("no model query needed" in m for m in nav.log))
        finally:
            bot.close()

    def test_reaching_a_waypoint_surveys_again(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator.for_bot(bot, explorer=OracleExplorer(bot),
                                          detector=lambda obs, **_: [])   # never sees anything
            states = []
            while bot.time < 90 and not nav.done and not any("reached waypoint" in m for m in nav.log):
                bot.step(0.1, controller=nav)
            self.assertTrue(any("reached waypoint" in m for m in nav.log), nav.log)
            self.assertEqual(nav.state, nav.SURVEY)
            self.assertEqual(len(nav.explored), 1)
        finally:
            bot.close()

    def test_step_cap_ends_in_stuck_without_arriving(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator.for_bot(bot, explorer=OracleExplorer(bot), max_explore_steps=2,
                                          detector=lambda obs, **_: [])
            states = []
            run(bot, nav, 240.0, states)
            self.assertEqual(nav.state, nav.STUCK)
            self.assertNotIn(nav.ARRIVED, states)
            self.assertEqual(nav.explore_steps, 2)
            self.assertTrue(any("without finding the goal" in m for m in nav.log))
        finally:
            bot.close()

    def test_unreachable_waypoint_surveys_again(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            explorer = OracleExplorer(bot)
            nav = VisualNavigator.for_bot(bot, explorer=explorer, detector=lambda obs, **_: [])
            while bot.time < 30 and nav.state != nav.EXPLORE:
                bot.step(0.1, controller=nav)
            self.assertEqual(nav.state, nav.EXPLORE)
            nav.explore_target = np.array([3.0, -3.0])        # inside the divider
            with mock.patch("vision_sim.navigation.planning.astar", return_value=None):
                while bot.time < 60 and nav.state == nav.EXPLORE:
                    bot.step(0.1, controller=nav)
            self.assertEqual(nav.state, nav.SURVEY)
            self.assertTrue(any("no route to waypoint" in m for m in nav.log))
            self.assertTrue(any(np.allclose(p, [3.0, -3.0]) for p in nav.explored))
        finally:
            bot.close()

    def test_losing_the_goal_goes_back_to_exploring(self):
        bot = make_bot()
        try:
            nav = VisualNavigator.for_bot(bot, explorer=OracleExplorer(bot), detector=lambda obs, **_: [])
            nav.goal_xy, nav.state = np.array([6.5, -2.5]), nav.NAVIGATE
            nav._research(bot, 5.0, "test")
            self.assertEqual((nav.state, nav.goal_xy, nav.explore_target), (nav.SURVEY, None, None))
        finally:
            bot.close()

    def test_llm_detector_is_not_run_on_every_survey_photo(self):
        bot = make_bot()
        try:
            det = L.LlmGoalDetector()
            nav = VisualNavigator.for_bot(bot, explorer=OracleExplorer(bot), detector=det)
            self.assertFalse(nav._cheap_detector())
            self.assertTrue(VisualNavigator.for_bot(bot, detector=yolo())._cheap_detector())
        finally:
            bot.close()


# ======================================================================= CLI
class TestExploreCommandLine(unittest.TestCase):
    def test_explore_searches_with_yolo_and_tracks_by_depth(self):
        import run_navigation as rn
        a = rn.parse_args(["--explore"])
        self.assertEqual((rn.resolve_detector(a), rn.resolve_track(a)), ("yolo", "depth"))
        self.assertEqual(rn.resolve_detector(rn.parse_args(["--explore", "--detector", "colour"])), "colour")
        self.assertEqual(rn.resolve_detector(rn.parse_args([])), "colour")
        self.assertIsNone(rn.build_explorer(rn.parse_args([])))

    def test_flags_reach_the_explorer(self):
        import run_navigation as rn
        a = rn.parse_args(["--explore", "--survey-shots", "6", "--survey-conf", "0.5",
                           "--explore-step", "2.5", "--max-explore-steps", "4", "--llm-thinking",
                           "--target", "a blue mug"])
        with mock.patch("builtins.print"):
            ex = rn.build_explorer(a)
        self.assertEqual((ex.n_shots, ex.min_confidence, ex.max_step, ex.client.thinking, ex.target),
                         (6, 0.5, 2.5, True, "a blue mug"))
        self.assertEqual(a.max_explore_steps, 4)


# ======================================================================= live
@unittest.skipIf(os.environ.get("SKIP_LIVE_LLM"), "SKIP_LIVE_LLM is set")
@unittest.skipUnless(_server_up(LLM_BASE), f"no LLM server at {LLM_BASE}")
class TestExploreLive(unittest.TestCase):
    def test_real_model_picks_a_useful_waypoint_from_the_start(self):
        bot = make_bot()
        try:
            shots = shots_at(bot, 0.0, 0.0)
            ex = LlmExplorer(base_url=LLM_BASE.rstrip("/") + "/v1", ceiling=CEILING)
            t0 = time.time()
            r = ex.query(shots)
            wall = time.time() - t0
            self.assertIsNone(r.error, r.error)
            self.assertFalse(r.found, "the target is hidden from the start")
            self.assertIsNotNone(r.waypoint)
            self.assertLess(wall, 20.0)
            print(f"\n    live explore: {wall:.1f}s, waypoint ({r.waypoint[0]:.2f}, {r.waypoint[1]:.2f}) "
                  f"via {r.waypoint_source}, {np.linalg.norm(r.waypoint - DOOR):.2f} m from the doorway",
                  file=sys.stderr)
        finally:
            bot.close()

    def test_real_search_finds_and_reaches_the_hidden_goal(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            ex = LlmExplorer(base_url=LLM_BASE.rstrip("/") + "/v1")
            nav = VisualNavigator.for_bot(bot, explorer=ex, detector=yolo(), track="depth")
            t0 = time.time()
            run(bot, nav, 300.0)
            goal = goal_xyz(bot)
            dist = float(np.linalg.norm(bot.position[:2] - goal[:2]))
            self.assertEqual(nav.state, nav.ARRIVED, f"ended {nav.state} {dist:.2f} m away; {nav.log}")
            self.assertLess(dist, nav.stop_distance + 0.5)
            print(f"\n    live search: {time.time() - t0:.1f}s wall, {bot.time:.1f}s sim, "
                  f"{nav.explore_steps} LLM waypoint(s), arrived {dist:.2f} m from the goal",
                  file=sys.stderr)
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
