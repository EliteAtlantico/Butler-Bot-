"""Tests for the pretrained open-vocabulary YOLO detector (vision_sim.yolo_detector).

    ../.venv/bin/python -m unittest -v test_pretrained_yolo

No training on this simulator: YOLO-World is told what to look for by name.
These check that the object named as the target comes back labelled as the
navigator's goal, lands where the object really is, and that nothing else --
the blue pillars, a hidden column, the rest of the room -- is mistaken for it.
The weights are fetched into weights/ on first use.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from test_llm_scene_reasoning import place, visible_pixel  # noqa: E402
from test_explore import make_bot, shots_at  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim import yolo_detector as Y  # noqa: E402

COURSE = os.path.join(HERE, "obstacle_course.xml")
SEARCH = os.path.join(HERE, "search_course.xml")
APARTMENT = os.path.join(HERE, "apartment.xml")
TRAINED = os.path.join(HERE, "runs", "bracketbot_yolo", "weights", "best.pt")

_DETECTOR = None


def detector():
    """One pretrained model for the whole file; set_classes swaps the target."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = Y.YoloDetector()
    return _DETECTOR


def body_xy(bot, name):
    return bot.data.xpos[mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, name)][:2].copy()


def facing(bot, xy, dist, bearing):
    """Stand `dist` metres from xy (approached from `bearing`) looking straight at it."""
    x, y = xy[0] - dist * np.cos(bearing), xy[1] - dist * np.sin(bearing)
    place(bot, x, y, bearing)
    return perception.observe(bot, width=320, height=240)


def targets(dets):
    return [d for d in dets if d.label == perception.GOAL_LABEL]


# ================================================================== weights
class TestWeights(unittest.TestCase):
    def test_existing_path_is_used_as_is(self):
        self.assertEqual(Y.resolve_weights(TRAINED), TRAINED)

    def test_pretrained_name_resolves_into_the_weights_dir(self):
        self.assertEqual(Y.resolve_weights("yolov8l-worldv2.pt"),
                         str(Y.WEIGHTS_DIR / "yolov8l-worldv2.pt"))
        self.assertEqual(Y.resolve_weights(Y.DEFAULT_WEIGHTS), str(Y.WEIGHTS_DIR / Y.DEFAULT_WEIGHTS))

    def test_unknown_weights_fail_clearly(self):
        with self.assertRaises(FileNotFoundError):
            Y.resolve_weights("not-a-real-model.pt")
        with self.assertRaises(FileNotFoundError):
            Y.resolve_weights(os.path.join(tempfile.gettempdir(), "missing", "best.pt"))

    def test_command_line_default_matches_the_detector(self):
        import run_navigation as rn
        self.assertEqual(rn.YOLO_DEFAULT_WEIGHTS, Y.DEFAULT_WEIGHTS)


# ============================================================ vocabulary
class TestVocabulary(unittest.TestCase):
    def test_default_is_open_vocabulary_looking_for_the_red_cylinder(self):
        det = detector()
        det.set_classes([Y.DEFAULT_TARGET, *Y.HOUSEHOLD_CLASSES])
        self.assertTrue(det.open_vocab)
        self.assertEqual(det.target, Y.DEFAULT_TARGET)
        self.assertEqual(det.names[0], Y.DEFAULT_TARGET)
        self.assertEqual(det.label_for(0), perception.GOAL_LABEL)
        self.assertEqual(det.label_for(det.names_index("chair")), "chair")

    def test_target_is_not_duplicated_in_the_vocabulary(self):
        det = Y.YoloDetector(target="mug", classes=["cup", "mug", " ", "plate"])
        self.assertEqual(list(det.names.values()), ["mug", "cup", "plate"])
        self.assertEqual(det.label_for(0), perception.GOAL_LABEL)
        self.assertEqual(det.label_for(1), "cup")

    def test_runs_on_the_gpu_when_there_is_one(self):
        import torch
        self.assertEqual(detector().device, "cuda" if torch.cuda.is_available() else "cpu")

    def test_text_encoder_is_not_kept_after_naming_the_classes(self):
        self.assertIsNone(getattr(detector().model.model, "clip_model", None))

    def test_out_of_gpu_memory_falls_back_to_the_cpu(self):
        """The LLM server holds most of the GPU; a detector that cannot fit
        must keep working rather than end the run."""
        import torch
        det = Y.YoloDetector(target="red cylinder", device="cuda")
        real = det.model.predict
        calls = []

        def predict(*a, **kw):
            calls.append(kw["device"])
            if kw["device"] != "cpu":
                raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 20.00 MiB")
            return real(*a, **kw)

        bot = make_bot(COURSE)
        try:
            obs = facing(bot, body_xy(bot, "target_column"), 3.0, 0.0)
        finally:
            bot.close()
        with mock.patch.object(det.model, "predict", side_effect=predict), mock.patch("builtins.print"):
            found = targets(det(obs))
            targets(det(obs))
        self.assertEqual(calls, ["cuda", "cpu", "cpu"])
        self.assertEqual(det.device, "cpu")
        self.assertTrue(found)

    def test_other_errors_are_not_swallowed(self):
        det = detector()
        with mock.patch.object(det.model, "predict", side_effect=RuntimeError("shape mismatch")):
            with self.assertRaises(RuntimeError):
                det._predict(None)
        self.assertEqual(det.device, "cuda" if __import__("torch").cuda.is_available() else "cpu")

    def test_trained_fixed_class_net_still_loads(self):
        det = Y.YoloDetector(TRAINED)
        self.assertFalse(det.open_vocab)
        self.assertIsNone(det.target)
        self.assertIn(perception.GOAL_LABEL, det.names.values())
        self.assertEqual(det.label_for(det.names_index(perception.GOAL_LABEL)), perception.GOAL_LABEL)
        with self.assertRaises(ValueError):
            Y.YoloDetector(TRAINED, target="mug")


# ============================================================ detections
class TestFindsTheTarget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.det = detector()
        cls.det.set_classes([Y.DEFAULT_TARGET, *Y.HOUSEHOLD_CLASSES])

    def test_red_column_is_the_goal_and_ranged_to_it(self):
        bot = make_bot(COURSE)
        try:
            goal = body_xy(bot, "target_column")
            for dist in (2.5, 4.0, 6.0):
                obs = facing(bot, goal, dist, 0.0)
                self.assertIsNotNone(visible_pixel(obs, np.r_[goal, 0.6]))
                found = targets(self.det(obs, robot_yaw=0.0))
                self.assertTrue(found, f"no goal from {dist} m")
                err = float(np.linalg.norm(found[0].position[:2] - goal))
                self.assertLess(err, 0.4, f"goal from {dist} m ranged {err:.2f} m off")
        finally:
            bot.close()

    def test_blue_pillar_is_not_mistaken_for_the_goal(self):
        bot = make_bot(COURSE)
        try:
            goal = body_xy(bot, "target_column")
            for name in ("pillar_a", "pillar_b", "pillar_c"):
                xy = body_xy(bot, name)
                for bearing in np.linspace(0, 2 * np.pi, 6, endpoint=False):
                    obs = facing(bot, xy, 2.0, bearing)
                    for d in targets(self.det(obs, robot_yaw=bearing)):
                        self.assertLess(float(np.linalg.norm(d.position[:2] - goal)), 1.0,
                                        f"{name} seen from {np.degrees(bearing):.0f} deg called the goal")
        finally:
            bot.close()

    def test_hidden_column_gives_no_goal(self):
        bot = make_bot(SEARCH)
        try:
            for shot in shots_at(bot, 0.0, 0.0):
                self.assertEqual(targets(self.det(shot.obs, robot_yaw=shot.heading)), [],
                                 f"photo {shot.index} reported a goal that is behind a wall")
        finally:
            bot.close()

    def test_any_object_can_be_the_target(self):
        """Ask for a lamp in the apartment instead: it is found and ranged, and
        nothing else in the room is called the goal."""
        bot = make_bot(APARTMENT)
        det = self.det
        try:
            det.set_classes(["floor lamp", *Y.HOUSEHOLD_CLASSES])
            lamp = body_xy(bot, "lamp")
            hits = 0
            for dist in (2.0, 2.5, 3.0):
                for bearing in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    obs = facing(bot, lamp, dist, bearing)
                    for d in targets(det(obs, robot_yaw=bearing)):
                        err = float(np.linalg.norm(d.position[:2] - lamp))
                        self.assertLess(err, 1.0, f"a goal {err:.2f} m from the lamp")
                        hits += 1
            self.assertGreaterEqual(hits, 3)
        finally:
            det.set_classes([Y.DEFAULT_TARGET, *Y.HOUSEHOLD_CLASSES])
            bot.close()


# ======================================================================= CLI
class TestCommandLine(unittest.TestCase):
    def test_yolo_is_pretrained_by_default(self):
        import run_navigation as rn
        a = rn.parse_args(["--detector", "yolo"])
        with mock.patch("builtins.print"):
            det = rn.build_detector(a)
        self.assertEqual((det.weights, det.target, det.open_vocab),
                         (str(Y.WEIGHTS_DIR / Y.DEFAULT_WEIGHTS), Y.DEFAULT_TARGET, True))

    def test_target_and_classes_flags_reach_the_detector_and_explorer(self):
        import run_navigation as rn
        a = rn.parse_args(["--explore", "--target", "mug", "--classes", "cup, plate", "--conf", "0.3"])
        with mock.patch("builtins.print"):
            det = rn.build_detector(a)
            ex = rn.build_explorer(a)
        self.assertEqual((list(det.names.values()), det.conf), (["mug", "cup", "plate"], 0.3))
        self.assertEqual(ex.target, "mug")

    def test_explore_prompt_names_the_target(self):
        from vision_sim.llm_explore import LlmExplorer
        bot = make_bot(SEARCH)
        try:
            shots = shots_at(bot, 0.0, 0.0, n=2)
        finally:
            bot.close()
        self.assertIn("for a tall RED cylinder.", LlmExplorer().prompt(shots, []))
        p = LlmExplorer(target="a potted plant").prompt(shots, [])
        self.assertIn("searching an indoor space for a potted plant.", p)
        self.assertIn("look for a potted plant in every photo", p)
        self.assertNotIn("RED", p)


if __name__ == "__main__":
    unittest.main()
