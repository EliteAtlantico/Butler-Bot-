"""Tests for natural-language search: "find me a key" -> reason where keys are
kept -> search those places -> go to the keys, or give up and hand over.

    ../.venv/bin/python -m unittest -v test_command_search

Scene: home_search.xml, the two-room apartment with a set of keys on the
living-room sideboard, behind the divider and out of sight from the start.

Offline except `TestCommandLive`, skipped when the server is down or
SKIP_LIVE_LLM is set.
"""
from __future__ import annotations

import io
import json
import os
import sys
import time
import unittest
from contextlib import redirect_stdout
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np  # noqa: E402

from test_llm_scene_reasoning import LLM_BASE, _server_up, project, visible_pixel  # noqa: E402
from test_explore import box_around, make_bot, run, shots_at  # noqa: E402
from vision_sim import llm_reasoner as L  # noqa: E402
from vision_sim.llm_command import (CommandInterpreter, SearchTask,  # noqa: E402
                                    fallback_task)
from vision_sim.llm_explore import ExploreResult, LlmExplorer  # noqa: E402
from vision_sim.navigation import VisualNavigator  # noqa: E402

HOME = os.path.join(HERE, "home_search.xml")
DOORWAY = np.array([5.0, 1.2])           # the only way between the rooms
SIDEBOARD_TOP = np.array([9.6, -3.1, 0.84])
KEYS_TASK = SearchTask(command="find me a key", target="keys", description="a set of keys",
                       likely_places=["entryway table", "kitchen counter", "sideboard", "coffee table"],
                       reply="I'll look for your keys.")


class Reply:
    """A llama-server chat completion carrying `content`."""
    status_code, text = 200, ""

    def __init__(self, content):
        self.content = content

    def json(self):
        return {"choices": [{"message": {"content": self.content}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 42}}


def answer(**kw):
    return Reply(json.dumps(kw))


_KEYS_YOLO = None


def keys_yolo():
    global _KEYS_YOLO
    if _KEYS_YOLO is None:
        from vision_sim.yolo_detector import YoloDetector
        _KEYS_YOLO = YoloDetector(target="keys")
    return _KEYS_YOLO


# ==================================================================== parsing
class TestFallbackParsing(unittest.TestCase):
    def test_the_object_is_pulled_out_of_the_sentence(self):
        cases = {
            "find me a key": "key",
            "Can you find my keys?": "keys",
            "where did I leave my phone": "phone",
            "go to the sofa": "sofa",
            "please look for the remote control for me": "remote control",
            "Hey robot, where are my glasses?": "glasses",
        }
        for command, target in cases.items():
            task = fallback_task(command)
            self.assertEqual((task.target, task.source), (target, "fallback"), command)
            self.assertTrue(task.ok)

    def test_nothing_named_is_not_a_task(self):
        task = fallback_task("find me")
        self.assertFalse(task.ok)
        self.assertIn("couldn't tell", task.reply)


class TestInterpret(unittest.TestCase):
    def setUp(self):
        self.ci = CommandInterpreter()

    def test_model_answer_becomes_a_task(self):
        t = self.ci.interpret("find me a key", {
            "target": " Keys ", "description": "a set of keys",
            "likely_places": ["entryway table", "kitchen counter", "", "sideboard", "coffee table",
                              "nightstand", "coat pocket", "desk"],
            "reply": "On it -- checking the tables first."})
        self.assertEqual((t.target, t.description, t.source), ("keys", "a set of keys", "model"))
        self.assertEqual(t.likely_places, ["entryway table", "kitchen counter", "sideboard",
                                           "coffee table", "nightstand", "coat pocket"])
        self.assertEqual(t.reply, "On it -- checking the tables first.")
        self.assertIn("sideboard", t.summary())

    def test_places_as_one_string_are_split(self):
        t = self.ci.interpret("find the remote", {"target": "remote control",
                                                  "likely_places": "sofa, coffee table ,tv stand"})
        self.assertEqual(t.likely_places, ["sofa", "coffee table", "tv stand"])
        self.assertEqual(t.description, "remote control")
        self.assertIn("remote control", t.reply)

    def test_a_request_that_is_not_a_search_is_refused(self):
        t = self.ci.interpret("what's the weather like", {"target": None,
                                                          "reply": "I can only look for things."})
        self.assertFalse(t.ok)
        self.assertEqual((t.source, t.reply), ("model", "I can only look for things."))

    def test_an_answer_without_a_target_falls_back_to_the_sentence(self):
        t = self.ci.interpret("find my wallet", {"likely_places": ["desk"]})
        self.assertEqual((t.target, t.source, t.likely_places), ("wallet", "fallback", ["desk"]))


class TestParseRequest(unittest.TestCase):
    def test_one_text_only_request_with_thinking_off(self):
        ci = CommandInterpreter()
        with mock.patch.object(L.requests, "post", return_value=answer(
                target="keys", description="a set of keys", likely_places=["sideboard"],
                reply="Looking.")) as post:
            t = ci.parse("  find me   a key ")
        self.assertEqual(post.call_count, 1)
        payload = post.call_args.kwargs["json"]
        content = payload["messages"][0]["content"]
        self.assertEqual([c["type"] for c in content], ["text"])
        self.assertIn('The user said: "find me a key"', content[0]["text"])
        self.assertEqual(payload["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual((t.command, t.target, t.source), ("find me a key", "keys", "model"))
        self.assertIsNotNone(t.latency)

    def test_server_down_still_gives_a_task(self):
        with mock.patch.object(L.requests, "post", side_effect=ConnectionError("refused")):
            t = CommandInterpreter().parse("find me a key")
        self.assertEqual((t.target, t.source), ("key", "fallback"))
        self.assertIn("refused", t.error)

    def test_unparseable_answer_still_gives_a_task(self):
        with mock.patch.object(L.requests, "post", return_value=Reply("keys are usually by the door")):
            t = CommandInterpreter().parse("find my keys")
        self.assertEqual((t.target, t.source), ("keys", "fallback"))

    def test_empty_command_is_an_error(self):
        with self.assertRaises(ValueError):
            CommandInterpreter().parse("   ")


# ===================================================================== prompt
class TestExplorePrompt(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        bot = make_bot(HOME)
        try:
            cls.shots = shots_at(bot, 0.0, 0.0, n=2)
        finally:
            bot.close()

    def test_task_puts_the_request_and_likely_places_in_the_prompt(self):
        ex = LlmExplorer(target=KEYS_TASK.description, task=KEYS_TASK)
        p = ex.prompt(self.shots, [])
        self.assertIn('The user asked: "find me a key".', p)
        self.assertIn("usually found, most likely first: entryway table, kitchen counter, sideboard, "
                      "coffee table.", p)
        self.assertIn("check the top of every table", p)
        self.assertIn("look for a set of keys in every photo", p)
        self.assertIn("the most likely place for it that you can see", p)
        self.assertNotIn("Earlier choices", p)

    def test_earlier_choices_are_remembered(self):
        ex = LlmExplorer(target=KEYS_TASK.description, task=KEYS_TASK)
        ex._finish(ExploreResult(waypoint=np.array([2.2, -0.9]), waypoint_source="model",
                                 reason="dining table: keys are often left on tables"))
        p = ex.prompt(self.shots, [np.array([0.0, 0.0])])
        self.assertIn("Earlier choices (not found there): (2.2, -0.9) dining table", p)
        ex._finish(ExploreResult(waypoint=np.array([5.0, 1.0]), waypoint_source="fallback",
                                 reason="coffee table is the most likely place"))
        p = ex.prompt(self.shots, [])
        self.assertIn("(5.0, 1.0) open floor (no usable model choice)", p)
        self.assertNotIn("(5.0, 1.0) coffee table", p)

    def test_without_a_task_the_prompt_is_unchanged(self):
        p = LlmExplorer().prompt(self.shots, [])
        self.assertNotIn("The user asked", p)
        self.assertIn("-- a doorway, a gap between obstacles, the entrance to an unexplored area", p)


# ==================================================================== giving up
class NowhereExplorer(LlmExplorer):
    """Never sees the goal and never has anywhere to go."""

    def query(self, shots, visited=()):
        self.queries += 1
        return self._finish(ExploreResult(notes=["every direction is blocked"]))


class TestGivingUp(unittest.TestCase):
    def search(self, **kw):
        bot = make_bot(HOME)
        calls = []
        try:
            bot.balance.enable(bot.state)
            nav = VisualNavigator.for_bot(bot, explorer=NowhereExplorer(task=KEYS_TASK),
                                          detector=lambda obs, **_: [], track="depth",
                                          on_give_up=lambda n, why: calls.append((n, why)), **kw)
            run(bot, nav, 60.0)
            for _ in range(20):                    # stays given up, hook stays silent
                bot.step(0.1, controller=nav)
            return nav, calls
        finally:
            bot.close()

    def test_nowhere_to_go_gives_up_and_hands_over_once(self):
        nav, calls = self.search()
        self.assertEqual((nav.state, nav.outcome), (nav.STUCK, "gave_up"))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], nav)
        self.assertIn("nowhere left to go", calls[0][1])
        self.assertEqual(nav.give_up_reason, calls[0][1])

    def test_round_cap_gives_up_and_hands_over(self):
        nav, calls = self.search(max_explore_steps=0)
        self.assertEqual(nav.outcome, "gave_up")
        self.assertEqual([why for _, why in calls], ["explored 0 place(s) without finding the goal"])
        self.assertTrue(any("giving up" in m for m in nav.log))

    def test_still_searching_has_no_outcome(self):
        bot = make_bot(HOME)
        try:
            nav = VisualNavigator.for_bot(bot, explorer=NowhereExplorer(), detector=lambda obs, **_: [])
            self.assertIsNone(nav.outcome)
        finally:
            bot.close()


# ======================================================================= search
class KeysOracle(LlmExplorer):
    """Stands in for the model with common sense about this flat: reports the
    keys when they are in a photo, otherwise heads for the sideboard if it is in
    view, else the doorway to the other room. Answers go through the REAL
    interpret(), so waypoint geometry and goal ranging are production code."""

    def __init__(self, bot, **kw):
        super().__init__(target=KEYS_TASK.description, task=KEYS_TASK, **kw)
        self.bot = bot
        self.calls = 0

    def query(self, shots, visited=()):
        self.calls += 1
        self.queries += 1
        keys = self.bot.body_position("keys").copy()
        for s in shots:
            px = visible_pixel(s.obs, keys, tolerance=0.08)
            if px is not None:
                return self._finish(self.interpret(shots, {
                    "goal_found": True, "goal_photo": s.index, "goal_bbox_2d": box_around(*px, half=6),
                    "confidence": 0.9, "reason": "keys on the sideboard"}, visited))
        here = shots[0].robot_xy
        aims = [SIDEBOARD_TOP] if here[0] > DOORWAY[0] - 0.5 else [np.r_[DOORWAY, 0.05]]
        for aim in aims:
            best = None
            for s in shots:
                pr = project(s.obs, aim)
                if pr is None:
                    continue
                u, v, _ = pr
                if 0 <= u < 320 and 0 <= v < 240 and (best is None or abs(u - 160) < abs(best[1] - 160)):
                    best = (s.index, u, v)
            if best is not None:
                return self._finish(self.interpret(shots, {
                    "goal_found": False, "explore_photo": best[0],
                    "explore_bbox_2d": box_around(best[1], best[2]), "confidence": 0.7,
                    "reason": "sideboard" if aim is SIDEBOARD_TOP else "doorway to the living room"},
                    visited))
        return self._finish(self.interpret(shots, {"goal_found": False}, visited))


class TestKeysSearch(unittest.TestCase):
    def test_keys_found_in_the_other_room_and_reached(self):
        bot = make_bot(HOME)
        calls = []
        try:
            bot.balance.enable(bot.state)
            explorer = KeysOracle(bot)
            nav = VisualNavigator.for_bot(bot, explorer=explorer, detector=keys_yolo(), track="depth",
                                          max_explore_steps=8,
                                          on_give_up=lambda n, why: calls.append(why))
            run(bot, nav, 400.0)
            keys = bot.body_position("keys")[:2]
            dist = float(np.linalg.norm(bot.position[:2] - keys))
            self.assertFalse(bot.fallen)
            self.assertEqual(nav.outcome, "found", f"{nav.state} {dist:.2f} m away; {nav.log[-8:]}")
            self.assertLess(dist, nav.stop_distance + 0.5)
            self.assertLess(float(np.linalg.norm(nav.goal_xy - keys)), 0.5)
            self.assertEqual(calls, [])
            self.assertGreaterEqual(explorer.calls, 2, "the keys are out of sight from the start")
            self.assertTrue(all(np.linalg.norm(d.position[:2] - keys) < 1.0
                                for d in nav.best_detections if d.label == "target"),
                            "YOLO put a goal somewhere other than the keys")
        finally:
            bot.close()

    def test_keys_are_hidden_from_the_start(self):
        bot = make_bot(HOME)
        try:
            keys = bot.body_position("keys").copy()
            self.assertTrue(all(visible_pixel(s.obs, keys, tolerance=0.08) is None
                                for s in shots_at(bot, 0.0, 0.0)))
        finally:
            bot.close()


# ======================================================================= CLI
class TestCommandLine(unittest.TestCase):
    def test_a_command_is_a_search_with_a_few_rounds(self):
        import run_navigation as rn
        a = rn.parse_args(["--command", "find me a key"])
        self.assertTrue(a.explore)
        self.assertEqual((rn.resolve_detector(a), rn.resolve_track(a), rn.resolve_rounds(a)),
                         ("yolo", "depth", 6))
        self.assertEqual(rn.resolve_rounds(rn.parse_args(["--command", "x", "--max-explore-steps", "3"])), 3)
        self.assertEqual(rn.resolve_rounds(rn.parse_args(["--explore"])), 8)
        self.assertIsNone(rn.parse_args([]).command)

    def test_command_sets_the_target_and_reaches_the_explorer(self):
        import run_navigation as rn
        a = rn.parse_args(["--command", "find me a key"])
        with mock.patch("vision_sim.llm_command.CommandInterpreter.parse", return_value=KEYS_TASK), \
                mock.patch("builtins.print"):
            a.task = rn.build_command(a)
            ex = rn.build_explorer(a)
        self.assertEqual(a.target, "keys")
        self.assertIs(ex.task, KEYS_TASK)
        self.assertEqual(ex.target, "a set of keys")

    def test_explicit_target_wins_and_no_text_asks(self):
        import run_navigation as rn
        a = rn.parse_args(["--command", "--target", "key ring"])
        asked = []
        with mock.patch("vision_sim.llm_command.CommandInterpreter.parse", return_value=KEYS_TASK) as parse, \
                mock.patch("builtins.print"):
            rn.build_command(a, ask=lambda q: asked.append(q) or "find me a key")
        self.assertEqual(asked, ["What should I find? "])
        parse.assert_called_once_with("find me a key")
        self.assertEqual(a.target, "key ring")

    def test_hand_off_says_what_was_not_found(self):
        import run_navigation as rn
        out = io.StringIO()
        with redirect_stdout(out):
            rn.hand_off_to_remote_control(None, "explored 6 place(s) without finding the goal", KEYS_TASK)
        self.assertIn("could not find a set of keys (explored 6 place(s)", out.getvalue())
        self.assertIn("remote control", out.getvalue())

    def test_nothing_to_find_does_not_start_the_sim(self):
        import run_navigation as rn
        refusal = SearchTask(command="what's the weather", target=None, reply="I can only look for things.")
        with mock.patch.object(sys, "argv", ["run_navigation.py", "--command", "what's the weather"]), \
                mock.patch("run_navigation.build_command", return_value=refusal), \
                mock.patch("bracketbot_sim.robot.BracketBot", side_effect=AssertionError("sim started")), \
                redirect_stdout(io.StringIO()) as out:
            rn.main()
        self.assertIn("nothing to search for", out.getvalue())


# ======================================================================= live
@unittest.skipIf(os.environ.get("SKIP_LIVE_LLM"), "SKIP_LIVE_LLM is set")
@unittest.skipUnless(_server_up(LLM_BASE), f"no LLM server at {LLM_BASE}")
class TestCommandLive(unittest.TestCase):
    URL = LLM_BASE.rstrip("/") + "/v1"

    def test_real_model_reasons_where_keys_are_kept(self):
        t = CommandInterpreter(base_url=self.URL).parse("find me a key")
        self.assertEqual(t.source, "model", t.error)
        self.assertIn("key", t.target)
        self.assertGreaterEqual(len(t.likely_places), 3)
        self.assertTrue(any(w in " ".join(t.likely_places).lower()
                            for w in ("table", "counter", "hook", "bowl", "dresser", "shelf", "sideboard")),
                        t.likely_places)
        self.assertLess(t.latency, 15.0)
        print(f"\n    live command: {t.latency:.1f}s {t.summary()} -- {t.reply!r}", file=sys.stderr)

    def test_real_model_refuses_a_non_search(self):
        t = CommandInterpreter(base_url=self.URL).parse("what's the weather like today")
        self.assertFalse(t.ok, t.summary())

    def test_real_keys_search_in_the_home(self):
        bot = make_bot(HOME)
        try:
            bot.balance.enable(bot.state)
            task = CommandInterpreter(base_url=self.URL).parse("find me a key")
            ex = LlmExplorer(base_url=self.URL, target=task.description, task=task)
            calls = []
            nav = VisualNavigator.for_bot(bot, explorer=ex, detector=keys_yolo(), track="depth",
                                          # the CLI default is 6; live runs here took 5-6
                                          max_explore_steps=8, on_give_up=lambda n, why: calls.append(why))
            t0 = time.time()
            run(bot, nav, 600.0)
            keys = bot.body_position("keys")[:2]
            dist = float(np.linalg.norm(bot.position[:2] - keys))
            self.assertEqual(nav.outcome, "found", f"{nav.state} {dist:.2f} m away; {calls}; {nav.log}")
            self.assertLess(dist, nav.stop_distance + 0.5)
            print(f"\n    live keys search: {time.time() - t0:.1f}s wall, {bot.time:.1f}s sim, "
                  f"{nav.explore_steps} LLM round(s), arrived {dist:.2f} m from the keys", file=sys.stderr)
        finally:
            bot.close()


if __name__ == "__main__":
    unittest.main()
