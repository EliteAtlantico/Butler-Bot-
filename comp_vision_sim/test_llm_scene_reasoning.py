"""Test suite for the LLM scene-reasoning goal detector (--detector llm).

Runs with the standard library, from comp_vision_sim/:

    ../.venv/bin/python -m unittest -v test_llm_scene_reasoning

(pytest collects it too.) Everything runs offline except `TestLiveServer`,
which makes one real query and is skipped automatically when the llama-server
is unreachable. Set SKIP_LIVE_LLM=1 to skip it anyway, or LLM_URL to point it at
a different server.

What is covered, and why:

  * parsing       -- the model answers in loosely-formatted JSON; a parse
                     failure silently costs a whole ~10 s query
  * cadence       -- queries are gated on sim time; a reset, a down server or
                     a clock switch must not make the robot blind or stall it
  * client        -- the request quirks that keep the Qwen build answering at
                     all (/no_think, enable_thinking=false) must not regress
  * geometry      -- pixel -> world through the depth image, at several robot
                     poses, including the bearing sign
  * integration   -- VisualNavigator drives to ARRIVED using the LLM detector
                     with the model replaced by an oracle, and the colour and
                     YOLO paths still work with the new `t` keyword
  * CLI           -- the --llm-* flags reach the detector
"""
from __future__ import annotations

import base64
import inspect
import io
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

from vision_sim import llm_reasoner as L  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim.llm_reasoner import LlmGoalDetector, _extract_json  # noqa: E402

SCENE = os.path.join(HERE, "obstacle_course.xml")
GOAL_BODY = "target_column"
GOAL_Z = 0.6            # a point on the column's axis, above the barriers


# --------------------------------------------------------------------- helpers
def make_bot():
    from bracketbot_sim.robot import BracketBot
    return BracketBot(xml=SCENE)


def goal_world(bot):
    xy = bot.body_position(GOAL_BODY)[:2]
    return np.array([xy[0], xy[1], GOAL_Z])


def place(bot, x, y, yaw):
    """Teleport the robot to a pose and refresh kinematics (no stepping)."""
    bot.reset()
    bot.data.qpos[0:2] = (x, y)
    bot.data.qpos[3:7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
    mujoco.mj_forward(bot.model, bot.data)


def project(obs, world):
    """World point -> (u, v, optical depth), or None if behind the camera."""
    p = obs.cam_mat.T @ (np.asarray(world, float) - obs.cam_pos)
    if p[2] >= -1e-6:
        return None
    intr = obs.intrinsics
    u = intr.fx * p[0] / -p[2] + intr.cx
    v = intr.fy * -p[1] / -p[2] + intr.cy
    return int(round(u)), int(round(v)), float(-p[2])


def visible_pixel(obs, world, tolerance=0.5):
    """The pixel a perfect VLM would report: in frame and not occluded."""
    pr = project(obs, world)
    if pr is None:
        return None
    u, v, depth = pr
    h, w = obs.depth.shape
    if not (0 <= u < w and 0 <= v < h):
        return None
    seen = obs.depth[v, u]
    if not np.isfinite(seen) or abs(seen - depth) > tolerance:
        return None                      # something is in front of the goal
    return u, v


def answer(found=True, u=None, v=None, conf=0.9, **extra):
    a = {"scene": "indoor course", "goal_found": found,
         "goal_horizontal_px": u, "goal_vertical_px": v,
         "goal_confidence": conf, "goal_depth_cue": "mid, centre",
         "other_objects": [{"label": "barrier", "horizontal_px": 10,
                            "vertical_px": 150, "note": "orange wall"}],
         "movement": "advance"}
    a.update(extra)
    return a


def stub(detector, reply):
    """Replace the model with a fixed or callable reply; returns a call log."""
    calls = []

    def ask_image(rgb, prompt):
        calls.append(prompt)
        r = reply() if callable(reply) else reply
        if isinstance(r, Exception):
            raise r
        return (r if isinstance(r, str) else json.dumps(r)), 0.0

    detector.client.ask_image = ask_image
    return calls


class FakeObs:
    """Just enough of an Observation for the cadence tests (no rendering)."""
    class intrinsics:
        width, height = 320, 240
    rgb = np.zeros((240, 320, 3), np.uint8)


# ===================================================================== parsing
class TestExtractJson(unittest.TestCase):
    NESTED = {"scene": "s", "goal_found": True, "goal_horizontal_px": 160,
              "other_objects": [{"label": "pillar", "horizontal_px": 12}]}

    def test_plain_compact_object(self):
        self.assertEqual(_extract_json(json.dumps(self.NESTED)), self.NESTED)

    def test_fenced_with_nested_objects(self):
        text = "```json\n" + json.dumps(self.NESTED, indent=2) + "\n```"
        self.assertEqual(_extract_json(text), self.NESTED)

    def test_prose_before_and_after(self):
        text = "Sure! Here it is:\n" + json.dumps(self.NESTED) + "\nHope that helps {really}."
        self.assertEqual(_extract_json(text), self.NESTED)

    def test_closing_brace_inside_a_string_value(self):
        # Regression: the old brace counter ended the object at the `}` in the
        # string and the whole query was discarded as unparseable.
        obj = {"scene": "a sign reading } on the wall", "goal_found": False}
        self.assertEqual(_extract_json(json.dumps(obj)), obj)

    def test_opening_brace_inside_a_string_value(self):
        obj = {"note": "curly { brace", "goal_found": True}
        self.assertEqual(_extract_json(json.dumps(obj)), obj)

    def test_skips_a_stray_brace_before_the_real_object(self):
        text = "reasoning {not json here} ... " + json.dumps(self.NESTED)
        self.assertEqual(_extract_json(text), self.NESTED)

    def test_no_object_raises(self):
        with self.assertRaises(ValueError):
            _extract_json("the goal is to the left")

    def test_none_raises(self):
        with self.assertRaises(ValueError):
            _extract_json(None)

    def test_truncated_object_raises(self):
        with self.assertRaises(ValueError):
            _extract_json('{"scene": "cut off mid-sent')


class TestValueCoercion(unittest.TestCase):
    def test_to_bool_real_booleans(self):
        self.assertIs(L._to_bool(True), True)
        self.assertIs(L._to_bool(False), False)

    def test_to_bool_string_false_is_false(self):
        # Regression: `not "false"` is False, so the string used to count as found.
        for v in ("false", "False", " no ", "0", "null", "", None, 0):
            self.assertIs(L._to_bool(v), False, repr(v))

    def test_to_bool_string_true_is_true(self):
        for v in ("true", "TRUE", "yes", "1", 1):
            self.assertIs(L._to_bool(v), True, repr(v))

    def test_to_bool_garbage_is_false(self):
        self.assertIs(L._to_bool("maybe"), False)

    def test_to_int_and_float(self):
        self.assertEqual(L._to_int("160.6"), 161)
        self.assertIsNone(L._to_int(None))
        self.assertIsNone(L._to_int("left"))
        self.assertEqual(L._to_float("0.75"), 0.75)
        self.assertIsNone(L._to_float("high"))


# ===================================================================== cadence
class TestQueryCadence(unittest.TestCase):
    def setUp(self):
        self.det = LlmGoalDetector(query_period=3.0)
        self.calls = stub(self.det, answer(found=False))

    def test_first_call_queries(self):
        self.det(FakeObs(), t=0.0)
        self.assertEqual(len(self.calls), 1)

    def test_gated_on_sim_time_not_calls(self):
        for k in range(30):                       # 30 sense ticks, 0.2 s apart
            self.det(FakeObs(), t=0.2 * k)
        # t = 0.0, 3.0, ... -> queries at 0.0, 3.0 (5.8 s of sim)
        self.assertEqual(len(self.calls), 2)

    def test_cache_returned_between_queries(self):
        det = LlmGoalDetector(query_period=3.0)
        sentinel = object()
        stub(det, answer(found=False))
        det(FakeObs(), t=0.0)
        det._cache = [sentinel]
        self.assertEqual(det(FakeObs(), t=1.0), [sentinel])

    def test_sim_reset_queries_immediately(self):
        # Regression: after a reset zeroed t, the detector stayed silent until
        # t climbed back past the pre-reset query time.
        self.det(FakeObs(), t=40.0)
        self.det(FakeObs(), t=0.1)                # clock ran backwards
        self.assertEqual(len(self.calls), 2)
        self.det(FakeObs(), t=0.3)                # and gating resumes from there
        self.assertEqual(len(self.calls), 2)
        self.det(FakeObs(), t=3.2)
        self.assertEqual(len(self.calls), 3)

    def test_failures_back_off_by_the_period(self):
        # Regression: failures were never stamped, so a down server was retried
        # on every sense tick, each blocking the sim for up to `timeout` s.
        det = LlmGoalDetector(query_period=3.0)
        calls = stub(det, ConnectionError("server down"))
        for k in range(10):
            det(FakeObs(), t=0.2 * k)
        self.assertEqual(len(calls), 1)
        self.assertEqual(det.errors, 1)
        det(FakeObs(), t=3.1)
        self.assertEqual(len(calls), 2)

    def test_bad_json_keeps_last_good_detection(self):
        det = LlmGoalDetector(query_period=1.0)
        good = object()
        stub(det, "no json at all")
        det._cache = [good]
        out = det(FakeObs(), t=0.0)
        self.assertEqual(out, [good])
        self.assertEqual(det.errors, 1)
        self.assertEqual(det.queries, 0)

    def test_wall_clock_fallback_when_no_sim_time(self):
        det = LlmGoalDetector(query_period=3.0)
        calls = stub(det, answer(found=False))
        # A clock the test moves by hand: the detector may read it any number
        # of times per call (it checks, then stamps), so a fixed list of
        # readings would run out.
        clock = {"now": 100.0}
        with mock.patch.object(L.time, "monotonic", side_effect=lambda: clock["now"]):
            det(FakeObs())                 # 100.0: first call queries
            clock["now"] = 101.0
            det(FakeObs())                 # 1 s later: gated
            clock["now"] = 103.5
            det(FakeObs())                 # 3.5 s after the query: due
        self.assertEqual(len(calls), 2)
        self.assertEqual(det._clock, "wall")

    def test_switching_clocks_queries(self):
        det = LlmGoalDetector(query_period=3.0)
        calls = stub(det, answer(found=False))
        det(FakeObs(), t=50.0)
        with mock.patch.object(L.time, "monotonic", return_value=1.0):
            det(FakeObs())                        # sim -> wall: stamps not comparable
        self.assertEqual(len(calls), 2)

    def test_period_has_a_floor(self):
        self.assertEqual(LlmGoalDetector(query_period=0.0).query_period, 0.5)

    def test_truncated_answers_are_counted(self):
        det = LlmGoalDetector(query_period=1.0)

        def ask_image(rgb, prompt):
            det.client.last_finish_reason = "length"
            return '{"scene": "cut', 0.0
        det.client.ask_image = ask_image
        det(FakeObs(), t=0.0)
        self.assertEqual((det.truncated, det.errors), (1, 1))
        self.assertIn("truncated=1", repr(det))


# ====================================================================== client
class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload, self.status_code = payload, status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class TestClientRequest(unittest.TestCase):
    def reply(self, content='{"goal_found": false}', finish="stop"):
        return FakeResponse({"choices": [{"message": {"content": content},
                                          "finish_reason": finish}],
                             "usage": {"completion_tokens": 42}})

    def test_payload_keeps_the_qwen_quirks(self):
        client = L.LLMClient("http://example:9/v1/", model="m", max_tokens=777,
                             temperature=0.3, timeout=5.0)
        rgb = np.zeros((24, 32, 3), np.uint8)
        rgb[..., 0] = 200
        with mock.patch.object(L.requests, "post", return_value=self.reply()) as post:
            content, dt = client.ask_image(rgb, "PROMPT")
        url, kwargs = post.call_args[0][0], post.call_args[1]
        body = kwargs["json"]
        self.assertEqual(url, "http://example:9/v1/chat/completions")
        self.assertEqual(kwargs["timeout"], 5.0)
        self.assertEqual((body["model"], body["max_tokens"], body["temperature"]),
                         ("m", 777, 0.3))
        # Without these two the Qwen build spends its whole budget thinking
        # and returns empty content.
        self.assertIs(body["enable_thinking"], False)
        text_part, image_part = body["messages"][0]["content"]
        self.assertTrue(text_part["text"].startswith("PROMPT"))
        self.assertTrue(text_part["text"].rstrip().endswith("/no_think"))
        url = image_part["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        from PIL import Image
        img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
        self.assertEqual(img.size, (32, 24))
        self.assertEqual(content, '{"goal_found": false}')
        self.assertEqual(client.last_finish_reason, "stop")

    def test_http_error_raises(self):
        client = L.LLMClient()
        with mock.patch.object(L.requests, "post",
                               return_value=FakeResponse({"error": "boom"}, status=500)):
            with self.assertRaises(RuntimeError):
                client.ask_image(np.zeros((4, 4, 3), np.uint8), "p")

    def test_finish_reason_length_recorded(self):
        client = L.LLMClient()
        with mock.patch.object(L.requests, "post",
                               return_value=self.reply('{"scene": "cu', finish="length")):
            client.ask_image(np.zeros((4, 4, 3), np.uint8), "p")
        self.assertEqual(client.last_finish_reason, "length")

    def test_prompt_asks_for_a_normalised_box(self):
        det = LlmGoalDetector()
        calls = stub(det, answer(found=False))
        det(FakeObs(), t=0.0)
        self.assertIn('"bbox_2d"', calls[0])
        self.assertIn("0-1000", calls[0])

    def test_prompt_states_the_frame_size(self):
        det = LlmGoalDetector()
        calls = stub(det, answer(found=False))
        det(FakeObs(), t=0.0)
        self.assertIn("320 px wide", calls[0])
        self.assertIn("240 px tall", calls[0])


# ==================================================================== geometry
class TestPixelToWorld(unittest.TestCase):
    """Feed the detector the goal's true pixel; it must recover the goal."""

    @classmethod
    def setUpClass(cls):
        cls.bot = make_bot()
        cls.goal = goal_world(cls.bot)

    @classmethod
    def tearDownClass(cls):
        cls.bot.close()

    def locate(self, x, y, yaw, **reply):
        place(self.bot, x, y, yaw)
        obs = perception.observe(self.bot, width=320, height=240, max_range=12.0)
        det = LlmGoalDetector(query_period=1.0)
        px = visible_pixel(obs, self.goal)
        if not reply:
            self.assertIsNotNone(px, f"goal not visible from ({x}, {y}, {yaw:.2f})")
            reply = answer(u=px[0], v=px[1], conf=0.9)
        stub(det, reply)
        return obs, det(obs, robot_yaw=self.bot.yaw, t=0.0)

    def assert_on_goal(self, dets, obs, tol=0.5):
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.label, "target")
        err = float(np.linalg.norm(d.position[:2] - self.goal[:2]))
        self.assertLess(err, tol, f"position error {err:.2f} m")
        self.assertGreater(d.pixels, 0, "depth fusion should find the column")
        self.assertEqual(d.position.shape, (3,))
        self.assertGreater(d.distance, 0.0)
        # bearing must agree with the true direction to the goal
        delta = self.goal[:2] - obs.robot_xy
        true_bearing = L.np.arctan2(delta[1], delta[0]) - self.bot.yaw
        wrapped = (d.bearing - true_bearing + np.pi) % (2 * np.pi) - np.pi
        self.assertLess(abs(wrapped), 0.15, f"bearing off by {np.degrees(wrapped):.1f} deg")
        return d

    def test_start_pose_facing_goal(self):
        obs, dets = self.locate(0.0, 0.0, 0.0)
        self.assert_on_goal(dets, obs)

    def test_close_range(self):
        obs, dets = self.locate(4.0, 0.0, 0.0)
        self.assert_on_goal(dets, obs)

    def test_goal_off_to_the_right(self):
        # Robot turned left, so the goal sits right of centre: bearing < 0.
        obs, dets = self.locate(4.0, 0.0, 0.35)
        d = self.assert_on_goal(dets, obs)
        self.assertLess(d.bearing, -0.1)

    def test_goal_off_to_the_left(self):
        obs, dets = self.locate(4.0, -0.8, 0.05)
        d = self.assert_on_goal(dets, obs)
        self.assertGreater(d.bearing, 0.1)

    def test_normalised_box_answer_is_ranged(self):
        place(self.bot, 0.0, 0.0, 0.0)
        obs = perception.observe(self.bot, width=320, height=240, max_range=12.0)
        u, v = visible_pixel(obs, self.goal)
        box = [(u - 6) / 320 * 1000, (v - 25) / 240 * 1000,
               (u + 6) / 320 * 1000, (v + 25) / 240 * 1000]
        obs, dets = self.locate(0.0, 0.0, 0.0, goal_found=True, bbox_2d=box,
                                goal_confidence=0.9)
        self.assert_on_goal(dets, obs)

    def test_goal_not_found(self):
        _, dets = self.locate(0.0, 0.0, 0.0, **answer(found=False))
        self.assertEqual(dets, [])

    def test_goal_found_as_string_false(self):
        # Regression: pixels present, "false" as a string -> used to be accepted.
        place(self.bot, 0.0, 0.0, 0.0)
        obs = perception.observe(self.bot, width=320, height=240, max_range=12.0)
        u, v = visible_pixel(obs, self.goal)
        _, dets = self.locate(0.0, 0.0, 0.0, **answer(found="false", u=u, v=v))
        self.assertEqual(dets, [])

    def test_low_confidence_rejected(self):
        place(self.bot, 0.0, 0.0, 0.0)
        obs = perception.observe(self.bot, width=320, height=240, max_range=12.0)
        u, v = visible_pixel(obs, self.goal)
        _, dets = self.locate(0.0, 0.0, 0.0, **answer(u=u, v=v, conf=0.2))
        self.assertEqual(dets, [])

    def test_missing_pixels_rejected(self):
        _, dets = self.locate(0.0, 0.0, 0.0, **answer(u=None, v=None))
        self.assertEqual(dets, [])

    def test_out_of_frame_pixel_is_clipped_not_crashing(self):
        _, dets = self.locate(0.0, 0.0, 0.0, **answer(u=5000, v=-40))
        self.assertLessEqual(len(dets), 1)

    def test_no_depth_falls_back_to_ground_ray(self):
        # Top row of the frame looks at sky: no depth to fuse, so the ray is
        # dropped to the ground plane instead.
        obs, dets = self.locate(0.0, 0.0, 0.0, **answer(u=160, v=0))
        self.assertEqual(len(dets), 1)
        d = dets[0]
        self.assertEqual(d.pixels, 0)
        self.assertTrue(np.all(np.isfinite(d.position)))
        self.assertGreater(d.position[0], obs.cam_pos[0], "ray should point forward")


class TestBoxAnswers(unittest.TestCase):
    """bbox_2d normalised to 0-1000: the format the model emits natively."""

    def test_centre_of_a_normalised_box(self):
        self.assertEqual(L._goal_pixel({"bbox_2d": [450, 200, 550, 400]}, 320, 240), (160, 72))

    def test_the_real_answer_shape(self):
        # the no-think answer from the live experiment at the start pose
        self.assertEqual(L._goal_pixel({"goal_found": True, "bbox_2d": [481, 238, 519, 348]},
                                       320, 240), (160, 70))

    def test_values_are_clipped_and_ordered(self):
        self.assertEqual(L._goal_pixel({"bbox_2d": [1200, 700, -50, 500]}, 320, 240), (160, 144))
        self.assertEqual(L._goal_pixel({"bbox_2d": [1000, 1000, 1000, 1000]}, 320, 240), (319, 239))

    def test_string_numbers(self):
        self.assertEqual(L._goal_pixel({"bbox_2d": ["450", "200", "550", "400"]}, 320, 240), (160, 72))

    def test_legacy_pixel_fields_still_work(self):
        self.assertEqual(L._goal_pixel({"goal_horizontal_px": 12, "goal_vertical_px": 34}, 320, 240),
                         (12, 34))

    def test_box_wins_over_legacy_fields(self):
        r = {"bbox_2d": [450, 200, 550, 400], "goal_horizontal_px": 1, "goal_vertical_px": 1}
        self.assertEqual(L._goal_pixel(r, 320, 240), (160, 72))

    def test_malformed_boxes(self):
        for bad in (None, [1, 2, 3], "450,200,550,400", [1, None, 3, 4], ["a", 1, 2, 3]):
            with self.subTest(bad=bad):
                self.assertIsNone(L._goal_pixel({"bbox_2d": bad}, 320, 240))


# ================================================================= integration
class OracleVLM(LlmGoalDetector):
    """A perfect vision model: reports the goal's true pixel whenever the
    camera can actually see it, and 'not found' otherwise. Exercises every
    part of the real detector except the network call."""

    def __init__(self, goal, **kw):
        super().__init__(**kw)
        self.goal = goal
        self._obs = None
        self.client.ask_image = self._ask

    def __call__(self, obs, robot_yaw=0.0, t=None, **kw):
        self._obs = obs
        return super().__call__(obs, robot_yaw=robot_yaw, t=t, **kw)

    def _ask(self, rgb, prompt):
        px = visible_pixel(self._obs, self.goal)
        if px is None:
            return json.dumps(answer(found=False)), 0.0
        return json.dumps(answer(u=px[0], v=px[1], conf=0.9)), 0.0


def drive(bot, nav, limit):
    while bot.time < limit and not bot.fallen and not nav.done:
        bot.step(0.1, controller=nav)


class TestNavigationEndToEnd(unittest.TestCase):
    """The full scan -> reason -> plan -> drive loop, offline."""

    def test_llm_detector_reaches_the_goal(self):
        from vision_sim.navigation import VisualNavigator
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            goal = goal_world(bot)
            det = OracleVLM(goal, query_period=3.0)
            nav = VisualNavigator(detector=det)
            drive(bot, nav, 90.0)
            dist = float(np.linalg.norm(bot.position[:2] - goal[:2]))
            self.assertFalse(bot.fallen)
            self.assertEqual(nav.state, nav.ARRIVED, f"ended {nav.state} {dist:.2f} m away")
            self.assertLess(dist, nav.stop_distance + 0.5)
            # sim-time gating survives the trip through VisualNavigator.sense
            expected = bot.time / det.query_period
            self.assertLessEqual(det.queries, expected + 2)
            self.assertGreaterEqual(det.queries, expected - 2)
            self.assertEqual(det.errors, 0)
        finally:
            bot.close()

    def test_colour_detector_still_arrives(self):
        from vision_sim.navigation import VisualNavigator
        bot = make_bot()
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator()
            drive(bot, nav, 60.0)
            goal = goal_world(bot)
            dist = float(np.linalg.norm(bot.position[:2] - goal[:2]))
            self.assertEqual(nav.state, nav.ARRIVED, f"ended {nav.state} {dist:.2f} m away")
            self.assertLess(dist, nav.stop_distance + 0.5)
        finally:
            bot.close()

    def test_every_detector_accepts_sim_time(self):
        from vision_sim.yolo_detector import YoloDetector
        for fn in (perception.detect, YoloDetector.__call__, LlmGoalDetector.__call__):
            kinds = {p.kind for p in inspect.signature(fn).parameters.values()}
            has_t = "t" in inspect.signature(fn).parameters
            self.assertTrue(has_t or inspect.Parameter.VAR_KEYWORD in kinds,
                            f"{fn.__qualname__} cannot take t=")


# ========================================================================= CLI
class TestCommandLine(unittest.TestCase):
    def test_llm_flags_reach_the_detector(self):
        import run_navigation as rn
        args = rn.parse_args(["--detector", "llm", "--llm-url", "http://h:1/v1",
                              "--llm-model", "some-model", "--llm-period", "2.5",
                              "--llm-conf", "0.6", "--llm-max-tokens", "1024"])
        with mock.patch("builtins.print"):
            det = rn.build_detector(args)
        self.assertIsInstance(det, LlmGoalDetector)
        self.assertEqual(det.client.base_url, "http://h:1/v1")
        self.assertEqual(det.client.model, "some-model")
        self.assertEqual(det.query_period, 2.5)
        self.assertEqual(det.min_confidence, 0.6)
        self.assertEqual(det.client.max_tokens, 1024)

    def test_defaults(self):
        import run_navigation as rn
        args = rn.parse_args(["--detector", "llm"])
        self.assertEqual((args.llm_url, args.llm_model, args.llm_period,
                          args.llm_conf, args.llm_max_tokens),
                         ("http://localhost:8080/v1", "Qwen/Qwen3.8-27B", 3.0, 0.40, 2048))
        self.assertIsNone(rn.build_detector(rn.parse_args([])))


# ======================================================================== live
def _server_up(base):
    try:
        import requests
        return requests.get(base.rstrip("/") + "/health", timeout=3).ok
    except Exception:
        return False


LLM_BASE = os.environ.get("LLM_URL", "http://localhost:8080")


@unittest.skipIf(os.environ.get("SKIP_LIVE_LLM"), "SKIP_LIVE_LLM is set")
@unittest.skipUnless(_server_up(LLM_BASE), f"no LLM server at {LLM_BASE}")
class TestLiveServer(unittest.TestCase):
    """One real query: the running model must find the red column."""

    def test_real_model_locates_the_goal(self):
        bot = make_bot()
        try:
            place(bot, 0.0, 0.0, 0.0)
            obs = perception.observe(bot, width=320, height=240, max_range=12.0)
            det = LlmGoalDetector(base_url=LLM_BASE.rstrip("/") + "/v1", query_period=1.0)
            t0 = time.time()
            dets = det(obs, robot_yaw=bot.yaw, t=0.0)
            wall = time.time() - t0
            self.assertEqual(det.errors, 0, f"query failed after {wall:.1f}s")
            self.assertEqual(det.truncated, 0)
            self.assertTrue(dets, f"model did not find the goal: {det.last_reasoning}")
            err = float(np.linalg.norm(dets[0].position[:2] - goal_world(bot)[:2]))
            self.assertLess(err, 0.6, f"model's goal is {err:.2f} m off")
            print(f"\n    live query: {wall:.1f}s, goal error {err:.2f} m", file=sys.stderr)
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
