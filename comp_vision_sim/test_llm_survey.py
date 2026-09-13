"""Tests for the 8-photo LLM survey (--survey).

    ../.venv/bin/python -m unittest -v test_llm_survey

The robot photographs N evenly spaced headings, and the local LLM is asked
once -- every photo labelled with the robot's position and the camera's world
heading -- which way the goal is. Everything here runs offline except
`TestSurveyLive`, which is skipped when the llama-server is unreachable (or when
SKIP_LIVE_LLM is set).
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "main_mujoco"))
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

# Helpers only -- importing TestCase classes here would run them twice.
from test_llm_scene_reasoning import (LLM_BASE, FakeResponse, _server_up,  # noqa: E402
                                      goal_world, make_bot, place, visible_pixel)
from vision_sim import llm_reasoner as L  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim.llm_survey import (LlmSurvey, SurveyShot, _wrap,  # noqa: E402
                                   pixel_heading)


def shots_from(bot, x, y, yaw0, n=8):
    """Photograph N headings by teleporting (no physics): the survey's input."""
    sv = LlmSurvey(n_shots=n)
    out = []
    for k, yaw in enumerate(sv.headings(yaw0)):
        place(bot, x, y, yaw)
        obs = perception.observe(bot, width=320, height=240, max_range=12.0)
        out.append(SurveyShot(k, yaw, np.array([x, y], float), obs))
    return out


def best_view(shots, goal):
    """(photo index, pixel) of the shot showing the goal closest to centre."""
    best = None
    for s in shots:
        px = visible_pixel(s.obs, goal)
        if px is None:
            continue
        off = abs(px[0] - s.obs.intrinsics.cx)
        if best is None or off < best[2]:
            best = (s.index, px, off)
    return None if best is None else best[:2]


def true_heading(shot_or_xy, goal):
    xy = shot_or_xy.robot_xy if hasattr(shot_or_xy, "robot_xy") else shot_or_xy
    d = np.asarray(goal[:2]) - np.asarray(xy)
    return float(np.arctan2(d[1], d[0]))


def ang(a, b):
    return abs(float(_wrap(a - b)))


class OracleSurvey(LlmSurvey):
    """A survey whose 'model' answers from ground truth, then goes through the
    real `interpret`. `mode` is "truth", "heading_only" or "not_found"."""

    def __init__(self, goal, mode="truth", **kw):
        super().__init__(**kw)
        self.goal, self.mode = goal, mode
        self.calls = 0
        self.seen_shots = None

    def query(self, shots):
        self.calls += 1
        self.seen_shots = list(shots)
        if self.mode == "not_found":
            raw = {"goal_found": False, "photo": None, "goal_px": None,
                   "heading_deg": None, "confidence": 0.9, "reason": "nothing red"}
        elif self.mode == "heading_only":
            raw = {"goal_found": True, "photo": None, "goal_px": None,
                   "heading_deg": np.degrees(true_heading(shots[0], self.goal)),
                   "confidence": 0.8, "reason": "that way"}
        else:
            view = best_view(shots, self.goal)
            raw = ({"goal_found": False, "confidence": 0.9, "reason": "not visible"}
                   if view is None else
                   {"goal_found": True, "photo": view[0], "goal_px": list(view[1]),
                    "heading_deg": None, "confidence": 0.9, "reason": "oracle"})
        res = self.interpret(shots, raw)
        res.latency = 0.0
        self.queries += 1
        self.last_result = res
        return res


# ======================================================================= plan
class TestSurveyPlan(unittest.TestCase):
    def test_eight_headings_evenly_spaced_from_current_yaw(self):
        h = LlmSurvey(n_shots=8).headings(0.3)
        self.assertEqual(len(h), 8)
        self.assertAlmostEqual(h[0], 0.3)
        for a, b in zip(h, h[1:] + h[:1]):
            self.assertAlmostEqual(ang(b, a), np.pi / 4, places=6)
        self.assertTrue(all(-np.pi <= x < np.pi for x in h))

    def test_other_counts(self):
        self.assertEqual(len(LlmSurvey(n_shots=4).headings(0.0)), 4)
        with self.assertRaises(ValueError):
            LlmSurvey(n_shots=0)

    def test_label_carries_position_and_heading(self):
        shot = SurveyShot(3, np.radians(135), np.array([1.25, -2.5]), obs=None)
        self.assertEqual(shot.label(),
                         "Photo 3: robot at (1.25, -2.50); this photo faces world heading 135 deg")
        neg = SurveyShot(5, np.radians(-90), np.array([0.0, 0.0]), obs=None)
        self.assertIn("faces world heading 270 deg", neg.label())

    def test_label_states_the_angle_from_photo_0(self):
        ref = np.radians(20)
        first = SurveyShot(0, ref, np.array([0.0, 0.0]), obs=None)
        third = SurveyShot(3, ref + np.radians(135), np.array([0.0, 0.0]), obs=None)
        wrap = SurveyShot(7, ref - np.radians(45), np.array([0.0, 0.0]), obs=None)
        self.assertIn("reference direction", first.label(ref))
        self.assertIn("135 deg counter-clockwise from photo 0", third.label(ref))
        self.assertIn("315 deg counter-clockwise from photo 0", wrap.label(ref))


# ================================================================== interpret
class TestSurveyLabelsWithCamera(unittest.TestCase):
    """Labels on real photos: each states its own view, and the edge headings
    it states agree with the camera geometry used to range the goal."""

    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.shots = shots_from(cls.bot, 1.0, -0.5, 0.7)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def test_every_label_describes_its_own_photo(self):
        for s in self.shots:
            label = s.label(self.shots[0].heading)
            with self.subTest(photo=s.index):
                self.assertTrue(label.startswith(f"Photo {s.index}:"))
                self.assertIn(f"faces world heading {np.degrees(s.heading) % 360:.0f} deg", label)
                self.assertIn("left edge looks along", label)
                self.assertIn("robot at (1.00, -0.50)", label)

    def test_edge_headings_match_the_pixel_geometry(self):
        for s in self.shots:
            left, right = s.view_span()
            w = s.obs.intrinsics.width
            with self.subTest(photo=s.index):
                self.assertLess(ang(left, pixel_heading(s, 0)), np.radians(0.5))
                self.assertLess(ang(right, pixel_heading(s, w)), np.radians(0.5))
                # left edge is counter-clockwise of centre, right edge clockwise
                self.assertGreater(float(_wrap(left - s.heading)), 0)
                self.assertLess(float(_wrap(right - s.heading)), 0)

    def test_neighbouring_photos_overlap(self):
        # 8 photos 45 deg apart with a ~73 deg view: nothing falls between photos.
        for a, b in zip(self.shots, self.shots[1:] + self.shots[:1]):
            a_left = a.view_span()[0]
            b_right = b.view_span()[1]
            self.assertGreater(float(_wrap(a_left - b_right)), 0,
                               f"gap between photo {a.index} and {b.index}")


class TestSurveyInterpret(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.goal = goal_world(cls.bot)
        cls.shots = shots_from(cls.bot, 0.0, 0.0, 0.0)
        cls.view = best_view(cls.shots, cls.goal)
        cls.sv = LlmSurvey()

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def test_goal_is_visible_from_start(self):
        self.assertIsNotNone(self.view)
        self.assertEqual(self.view[0], 0)

    def test_photo_and_pixel_are_ranged(self):
        k, px = self.view
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": k,
                                           "goal_px": list(px), "confidence": 0.9})
        self.assertTrue(r.found)
        self.assertEqual(r.heading_source, "range")
        self.assertIsNotNone(r.detection)
        self.assertLess(np.linalg.norm(r.detection.position[:2] - self.goal[:2]), 0.5)
        self.assertLess(ang(r.heading, true_heading(self.shots[k], self.goal)), np.radians(3))

    def test_normalised_box_is_ranged_in_the_chosen_photo(self):
        k, px = self.view
        intr = self.shots[k].obs.intrinsics
        box = [(px[0] - 6) / intr.width * 1000, (px[1] - 25) / intr.height * 1000,
               (px[0] + 6) / intr.width * 1000, (px[1] + 25) / intr.height * 1000]
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": k,
                                           "bbox_2d": box, "confidence": 0.9})
        self.assertEqual(r.heading_source, "range")
        self.assertLessEqual(max(abs(r.pixel[0] - px[0]), abs(r.pixel[1] - px[1])), 1)
        self.assertLess(np.linalg.norm(r.detection.position[:2] - self.goal[:2]), 0.5)

    def test_wrong_model_heading_does_not_override_the_ranged_box(self):
        # Live, the no-think model's heading_deg was once 30 deg off while its
        # box ranged to 0.15 m; the ranged heading must win.
        k, px = self.view
        intr = self.shots[k].obs.intrinsics
        box = [(px[0] - 6) / intr.width * 1000, (px[1] - 25) / intr.height * 1000,
               (px[0] + 6) / intr.width * 1000, (px[1] + 25) / intr.height * 1000]
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": k, "bbox_2d": box,
                                           "heading_deg": 30, "confidence": 0.9})
        self.assertEqual(r.heading_source, "range")
        self.assertLess(ang(r.heading, true_heading(self.shots[k], self.goal)), np.radians(3))

    def test_pixel_dict_form(self):
        k, px = self.view
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": k, "confidence": 0.9,
                                           "goal_px": {"horizontal_px": px[0], "vertical_px": px[1]}})
        self.assertEqual(r.heading_source, "range")

    def test_pixel_without_depth_gives_pixel_bearing(self):
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": 0,
                                           "goal_px": [100, 0], "confidence": 0.9})
        self.assertTrue(r.found)
        self.assertEqual(r.heading_source, "pixel")
        self.assertIsNone(r.detection)
        self.assertAlmostEqual(r.heading, pixel_heading(self.shots[0], 100))

    def test_heading_only_is_converted_and_wrapped(self):
        r = self.sv.interpret(self.shots, {"goal_found": True, "heading_deg": 270,
                                           "confidence": 0.8})
        self.assertEqual(r.heading_source, "model")
        self.assertAlmostEqual(r.heading, -np.pi / 2)

    def test_photo_only_uses_that_photos_heading(self):
        r = self.sv.interpret(self.shots, {"goal_found": True, "photo": 2, "confidence": 0.8})
        self.assertEqual(r.heading_source, "photo")
        self.assertAlmostEqual(r.heading, self.shots[2].heading)

    def test_missing_goal_found_is_inferred_from_a_photo(self):
        k, px = self.view
        r = self.sv.interpret(self.shots, {"photo": k, "goal_px": list(px), "confidence": 0.9})
        self.assertTrue(r.found)

    def test_rejections(self):
        k, px = self.view
        cases = {
            "not found": {"goal_found": False, "photo": k, "goal_px": list(px)},
            "string false": {"goal_found": "false", "photo": k, "goal_px": list(px)},
            "low confidence": {"goal_found": True, "photo": k, "goal_px": list(px),
                               "confidence": 0.1},
            "bad photo, nothing else": {"goal_found": True, "photo": 12, "confidence": 0.9},
        }
        for name, raw in cases.items():
            with self.subTest(name):
                self.assertFalse(self.sv.interpret(self.shots, raw).found)


# ====================================================================== query
class TestSurveyRequest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.shots = shots_from(cls.bot, 1.0, -1.0, 0.5)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def reply(self, content='{"goal_found": false}', finish="stop", reasoning=""):
        return FakeResponse({"choices": [{"message": {"content": content,
                                                      "reasoning_content": reasoning},
                                          "finish_reason": finish}],
                             "usage": {"completion_tokens": 77}})

    def test_one_request_with_every_photo_labelled(self):
        sv = LlmSurvey()
        with mock.patch.object(L.requests, "post", return_value=self.reply()) as post:
            res = sv.query(self.shots)
        self.assertEqual(post.call_count, 1)
        body = post.call_args[1]["json"]
        parts = body["messages"][0]["content"]
        texts = [p["text"] for p in parts if p["type"] == "text"]
        images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
        self.assertEqual(len(images), 8)
        self.assertEqual(texts[1:], [s.label(self.shots[0].heading) for s in self.shots])
        # every label comes immediately before its own photo, and names that
        # photo's own direction
        for i, s in enumerate(self.shots):
            label = parts[1 + 2 * i]["text"]
            self.assertEqual(label, s.label(self.shots[0].heading))
            self.assertIn(f"faces world heading {np.degrees(s.heading) % 360:.0f} deg", label)
            self.assertIn("left edge looks along", label)
            self.assertEqual(parts[2 + 2 * i]["type"], "image_url")
        self.assertIn("8 photos", texts[0])
        self.assertIn("320 px wide", texts[0])
        self.assertTrue(texts[0].rstrip().endswith("/no_think"))
        self.assertIs(body["enable_thinking"], False)
        self.assertEqual(body["max_tokens"], 6144)
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(images[0].split(",", 1)[1])))
        self.assertEqual(img.size, (320, 240))
        self.assertFalse(res.found)
        self.assertEqual(res.completion_tokens, 77)

    def test_prompt_asks_for_a_normalised_box(self):
        p = LlmSurvey().prompt(self.shots)
        self.assertIn('"bbox_2d"', p)
        self.assertIn("0-1000", p)
        self.assertIn("THAT photo", p)

    def test_http_error_is_a_clean_not_found(self):
        sv = LlmSurvey()
        with mock.patch.object(L.requests, "post",
                               return_value=FakeResponse({"error": "x"}, status=500)):
            res = sv.query(self.shots)
        self.assertFalse(res.found)
        self.assertIn("RuntimeError", res.error)
        self.assertEqual(sv.errors, 1)

    def test_unparseable_answer_is_an_error(self):
        sv = LlmSurvey()
        with mock.patch.object(L.requests, "post", return_value=self.reply("I think left")):
            res = sv.query(self.shots)
        self.assertFalse(res.found)
        self.assertIn("unparseable", res.error)

    def test_truncation_and_reasoning_are_recorded(self):
        sv = LlmSurvey()
        with mock.patch.object(L.requests, "post",
                               return_value=self.reply("", finish="length", reasoning="x" * 500)):
            res = sv.query(self.shots)
        self.assertEqual(sv.truncated, 1)
        self.assertEqual(res.finish_reason, "length")
        self.assertEqual(sv.client.last_reasoning_chars, 500)

    def test_mismatched_images_and_labels_rejected(self):
        with self.assertRaises(ValueError):
            L.LLMClient().ask_images([np.zeros((2, 2, 3), np.uint8)], [], "p")

    def test_no_shots_rejected(self):
        with self.assertRaises(ValueError):
            LlmSurvey().query([])


# =================================================================== geometry
class TestPixelHeading(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.goal = goal_world(cls.bot)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def test_centre_left_and_right(self):
        shot = shots_from(self.bot, 0.0, 0.0, 1.0, n=1)[0]
        cx = shot.obs.intrinsics.cx
        self.assertAlmostEqual(pixel_heading(shot, cx), shot.heading)
        self.assertGreater(ang(pixel_heading(shot, 0), shot.heading), 0.1)
        self.assertGreater(float(_wrap(pixel_heading(shot, 0) - shot.heading)), 0)      # left
        self.assertLess(float(_wrap(pixel_heading(shot, 319) - shot.heading)), 0)       # right

    def test_matches_true_bearing_off_centre(self):
        # Goal visible but well off centre in the 17 deg photo from this pose.
        shots = shots_from(self.bot, 2.0, -2.5, 0.3)
        view = best_view(shots, self.goal)
        self.assertIsNotNone(view)
        k, (u, _) = view
        self.assertLess(ang(pixel_heading(shots[k], u), true_heading(shots[k], self.goal)),
                        np.radians(2))


# ================================================================ navigation
def run_nav(bot, nav, limit, record=None):
    while bot.time < limit and not bot.fallen and not nav.done:
        bot.step(0.1, controller=nav)
        if record is not None:
            record.append((nav.state, bot.position[:2].copy(), bot.time))


class TestNavigatorSurvey(unittest.TestCase):
    def nav(self, bot, survey, **kw):
        from vision_sim.navigation import VisualNavigator
        return VisualNavigator(survey=survey, **kw)

    def test_photographs_eight_headings_in_place_then_arrives(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            goal = goal_world(bot)
            survey = OracleSurvey(goal)
            nav = self.nav(bot, survey)
            self.assertEqual(nav.state, nav.SURVEY)
            start = bot.position[:2].copy()
            log = []
            run_nav(bot, nav, 120.0, log)

            self.assertEqual(survey.calls, 1, "the model must be asked exactly once")
            shots = survey.seen_shots
            self.assertEqual(len(shots), 8)
            plan = nav._survey_plan
            for s, want in zip(shots, plan):
                self.assertLess(ang(s.heading, want), 0.12,
                                f"photo {s.index} at {np.degrees(s.heading):.0f}, "
                                f"planned {np.degrees(want):.0f}")
            # turned in place: the base barely moved while surveying
            surveying = [p for st, p, _ in log if st == nav.SURVEY]
            drift = max(np.linalg.norm(p - start) for p in surveying)
            self.assertLess(drift, 0.35, f"drifted {drift:.2f} m during the survey")
            # the survey's depth went into the map in every direction
            self.assertGreater(int(nav.grid.seen.sum()), 2000)

            self.assertEqual(nav.survey_result.heading_source, "range")
            self.assertFalse(bot.fallen)
            dist = float(np.linalg.norm(bot.position[:2] - goal[:2]))
            self.assertEqual(nav.state, nav.ARRIVED, f"ended {nav.state} {dist:.2f} m away")
            self.assertLess(dist, nav.stop_distance + 0.5)
        finally:
            bot.close()

    def test_not_found_falls_back_to_a_spin_scan(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            goal = goal_world(bot)
            nav = self.nav(bot, OracleSurvey(goal, mode="not_found"))
            log = []
            run_nav(bot, nav, 120.0, log)
            states = [s for s, _, _ in log]
            self.assertIn(nav.SCAN, states)
            self.assertLess(states.index(nav.SURVEY), states.index(nav.SCAN))
            self.assertEqual(nav.state, nav.ARRIVED)
        finally:
            bot.close()

    def test_heading_only_goal_is_provisional_until_sighted(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            goal = goal_world(bot)
            nav = self.nav(bot, OracleSurvey(goal, mode="heading_only"))
            provisional_seen = False
            while bot.time < 120.0 and not bot.fallen and not nav.done:
                bot.step(0.1, controller=nav)
                provisional_seen |= nav._goal_provisional
            self.assertTrue(provisional_seen, "a heading-only answer must start provisional")
            self.assertFalse(nav._goal_provisional, "the sighting must replace it")
            self.assertEqual(nav.survey_result.heading_source, "model")
            self.assertEqual(nav.state, nav.ARRIVED)
            self.assertLess(np.linalg.norm(bot.position[:2] - goal[:2]), nav.stop_distance + 0.5)
            self.assertTrue(any("replacing the survey's provisional goal" in m for m in nav.log))
        finally:
            bot.close()

    def test_reaching_a_provisional_goal_unsighted_is_not_arriving(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            # A detector that never sees anything, so the guess is never confirmed.
            nav = self.nav(bot, OracleSurvey(goal_world(bot), mode="heading_only"),
                           detector=lambda obs, **_: [], provisional_distance=1.8)
            log = []
            run_nav(bot, nav, 40.0, log)
            states = [s for s, _, _ in log]
            self.assertNotIn(nav.ARRIVED, states)
            self.assertIn(nav.SCAN, states[states.index(nav.NAVIGATE):])
        finally:
            bot.close()

    def test_coordinate_goal_skips_the_survey(self):
        bot = make_bot()
        try:
            survey = OracleSurvey(goal_world(bot))
            nav = self.nav(bot, survey, goal=(6.0, 0.0))
            self.assertEqual(nav.state, nav.SCAN)
            bot.balance.enable(bot.state)
            run_nav(bot, nav, 3.0)
            self.assertEqual(survey.calls, 0)
        finally:
            bot.close()

    def test_per_frame_detector_is_not_queried_while_surveying(self):
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            calls = []

            def detector(obs, **_):
                calls.append(nav.state)
                return perception.detect(obs)

            nav = self.nav(bot, OracleSurvey(goal_world(bot)), detector=detector)
            run_nav(bot, nav, 60.0)
            self.assertTrue(calls, "detector should run once navigating")
            self.assertNotIn(nav.SURVEY, calls)
        finally:
            bot.close()


# ======================================================================== CLI
class TestSurveyCommandLine(unittest.TestCase):
    def test_flags_reach_the_survey(self):
        import run_navigation as rn
        args = rn.parse_args(["--detector", "llm", "--survey", "--survey-shots", "6",
                              "--survey-max-tokens", "4096", "--survey-conf", "0.55",
                              "--llm-url", "http://h:2/v1", "--llm-model", "m2"])
        with mock.patch("builtins.print"):
            sv = rn.build_survey(args)
        self.assertIsInstance(sv, LlmSurvey)
        self.assertEqual((sv.n_shots, sv.client.max_tokens, sv.min_confidence),
                         (6, 4096, 0.55))
        self.assertEqual((sv.client.base_url, sv.client.model), ("http://h:2/v1", "m2"))

    def test_off_by_default(self):
        import run_navigation as rn
        args = rn.parse_args([])
        self.assertFalse(args.survey)
        self.assertEqual((args.survey_shots, args.survey_max_tokens, args.survey_conf),
                         (8, 6144, 0.40))
        self.assertIsNone(rn.build_survey(args))


# ======================================================================= live
@unittest.skipIf(os.environ.get("SKIP_LIVE_LLM"), "SKIP_LIVE_LLM is set")
@unittest.skipUnless(_server_up(LLM_BASE), f"no LLM server at {LLM_BASE}")
class TestSurveyLive(unittest.TestCase):
    """One real 8-photo query from an off-axis pose where the goal straddles
    two photos -- the case that exhausted a 1024-token budget."""

    def test_real_model_points_at_the_goal(self):
        bot = make_bot()
        try:
            goal = goal_world(bot)
            shots = shots_from(bot, 2.0, -2.5, 0.3)
            sv = LlmSurvey(base_url=LLM_BASE.rstrip("/") + "/v1")
            res = sv.query(shots)
            self.assertIsNone(res.error, res.error)
            self.assertEqual(sv.truncated, 0)
            self.assertTrue(res.found, f"model did not find the goal: {res.reason}")
            err = ang(res.heading, true_heading(shots[0], goal))
            self.assertLess(err, np.radians(10), f"heading off by {np.degrees(err):.1f} deg")
            msg = f"\n    live survey: {res.latency:.1f}s, heading error {np.degrees(err):.1f} deg"
            if res.detection is not None:
                derr = float(np.linalg.norm(res.detection.position[:2] - goal[:2]))
                self.assertLess(derr, 0.6)
                msg += f", position error {derr:.2f} m"
            print(msg, file=sys.stderr)
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
