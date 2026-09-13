"""The LLM tool-calling agent: the loop against a scripted model, the tool
plumbing against fakes, and scene-independent manipulation against the real
robot in two scenes (`integration`)."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
from types import SimpleNamespace

import pytest

from robot_agent import __main__ as cli
from robot_agent.agent import RobotAgent
from robot_agent.tools import (RobotTools, Tool, blocked_by, lead_in, nearest_free, object_spec,
                               resolve_scene, tool_catalogue)

SURFACES = ["coffee_table", "basket", "sideboard"]


class FakeTools:
    def __init__(self, results=None):
        self.calls = []
        self.results = results or {}

    def world_summary(self):
        return "Surfaces: basket: drop into, 0.01 m, (0.2, -1.3)"

    def specs(self):
        return [t.spec() for t in tool_catalogue(SURFACES)]

    def call(self, name, arguments):
        self.calls.append((name, json.loads(arguments)))
        return dict(self.results.get(name, {"ok": True}), robot={"x": 0, "y": 0})


def scripted(*messages):
    """A fake requests.post that replays assistant messages and records payloads."""
    sent = []

    def post(url, json=None, timeout=None):
        sent.append(copy.deepcopy(json))     # the agent keeps appending to the same list
        message = messages[min(len(sent), len(messages)) - 1]
        return SimpleNamespace(status_code=200, text="",
                               json=lambda: {"choices": [{"message": message}]})
    return post, sent


def call(name, **arguments):
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": f"id-{name}", "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)}}]}


def reply(text):
    return {"role": "assistant", "content": text}


def test_agent_calls_tools_in_turn_and_feeds_results_back():
    tools = FakeTools({"pick_up": {"ok": False, "failure": "fingers closed on nothing"}})
    post, sent = scripted(call("pick_up", object="mug", grasp="top", handle=True),
                          call("pick_up", object="mug", grasp="side"),
                          reply("Got the mug on the second try."))
    agent = RobotAgent(tools, post=post, verbose=False)
    assert agent.run("pick up the mug") == "Got the mug on the second try."
    assert tools.calls == [("pick_up", {"object": "mug", "grasp": "top", "handle": True}),
                           ("pick_up", {"object": "mug", "grasp": "side"})]
    second = sent[1]["messages"]
    assert second[0]["role"] == "system" and "basket: drop into" in second[0]["content"]
    assert "plan_grasp" in second[0]["content"]              # the model is taught the steps
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "id-pick_up"
    assert json.loads(second[-1]["content"])["failure"] == "fingers closed on nothing"
    assert {t["function"]["name"] for t in sent[0]["tools"]} >= {"pick_up", "place_held_item"}
    assert sent[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_conversation_carries_over_until_reset():
    post, sent = scripted(reply("Done."))
    agent = RobotAgent(FakeTools(), post=post, verbose=False)
    agent.run("pick up the mug")
    agent.run("now put it on the side table")
    users = [m["content"] for m in sent[-1]["messages"] if m["role"] == "user"]
    assert users == ["pick up the mug", "now put it on the side table"]
    agent.reset()
    assert [m["role"] for m in agent.messages] == ["system"]


def test_step_limit_asks_for_a_summary():
    tools = FakeTools()
    post, sent = scripted(*([call("wait", seconds=1)] * 3), reply("Still working on it."))
    agent = RobotAgent(tools, post=post, max_steps=3, verbose=False)
    assert agent.run("wait forever") == "Still working on it."
    assert len(tools.calls) == 3
    assert "Step limit reached" in sent[-1]["messages"][-1]["content"]


def test_an_identical_failing_call_is_refused_after_two_tries():
    tools = FakeTools({"go_to": {"ok": False, "error": "too close to the cartons"}})
    stuck = call("go_to", x=0.98, y=2.61)
    post, sent = scripted(stuck, stuck, stuck, call("go_near", x=0.98, y=2.61), reply("Moved."))
    agent = RobotAgent(tools, post=post, verbose=False)
    assert agent.run("go to the box") == "Moved."
    assert [c[0] for c in tools.calls] == ["go_to", "go_to", "go_near"]      # third never ran
    refused = json.loads(sent[3]["messages"][-1]["content"])
    assert "already failed 2 times" in refused["error"]


def test_nearest_free_spot_names_what_is_in_the_way():
    import numpy as np
    cartons = ("cartons", np.array([0.8, 2.4]), np.array([1.4, 3.0]))
    wall = ("wall_n", np.array([-1.0, 3.9]), np.array([11.0, 4.1]))
    assert blocked_by((1.1, 2.7), [cartons, wall]) == "cartons"
    assert blocked_by((1.1, 1.5), [cartons, wall]) is None
    spot, why = nearest_free((1.1, 2.7), [cartons, wall], toward=np.array([0.0, 0.0]))
    assert why == "cartons" and blocked_by(spot, [cartons, wall]) is None
    assert spot[1] < 2.4                     # comes out on the robot's side, not by the wall
    here, why = nearest_free((0.0, 0.0), [cartons, wall])
    assert why is None and np.allclose(here, (0.0, 0.0))


def test_decoration_is_not_scenery_a_drive_has_to_avoid():
    """Skirting and rugs are visual-only geoms. home_search.xml's skirting is one
    body of eleven strips round the whole flat, so merging it into a single
    footprint walled off all 94 m^2 and every go_to there was refused as "too
    close to the trim". Whatever else changes, the open floor must stay open."""
    import mujoco
    import numpy as np

    from handwrist.surfaces import obstacle_footprints

    model = mujoco.MjModel.from_xml_path(str(resolve_scene("apartment")))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    names = {name for name, _, _ in obstacle_footprints(model, data)}
    assert {"trim", "rug"}.isdisjoint(names), f"decoration is blocking the drive: {names}"
    assert {"table", "sofa", "bin", "cartons", "wall_n"} <= names, names
    # the spot the box is picked from, and the middle of each room
    for point in ((0.8, -1.6), (2.5, 2.0), (8.0, -2.0)):
        assert blocked_by(np.array(point), obstacle_footprints(model, data)) is None, point


def test_lead_in_is_behind_the_parking_spot_on_its_heading():
    import numpy as np
    assert np.allclose(lead_in((2.0, 1.0), 0.0, 0.8), (1.2, 1.0))
    assert np.allclose(lead_in((0.0, 0.0), np.pi / 2, 0.5), (0.0, -0.5))


def test_http_error_is_raised_with_the_body():
    def post(url, json=None, timeout=None):
        return SimpleNamespace(status_code=500, text="model crashed", json=lambda: {})
    with pytest.raises(RuntimeError, match="model crashed"):
        RobotAgent(FakeTools(), post=post, verbose=False).run("hi")


def test_catalogue_exposes_the_steps_not_fixed_chores():
    tools = {t.name: t for t in tool_catalogue(SURFACES, ["lj0", "rj1"])}
    assert {"detect_objects", "inspect_object", "plan_grasp", "pick_up", "list_surfaces",
            "place_held_item", "go_to", "go_near", "go_to_surface", "move", "turn", "search_for",
            "look_around",
            "describe_view", "set_gripper", "move_arm_joint", "stow_arms", "get_status",
            "wait"} == set(tools)
    for t in tools.values():
        spec = t.spec()["function"]
        assert spec["description"] and set(spec["parameters"]["required"]) <= set(t.parameters)
    grasp = tools["pick_up"].parameters
    assert {"grasp", "arm", "align_wrist", "grip_at", "handle", "shape", "squeeze_mm"} <= set(grasp)
    assert tools["plan_grasp"].parameters == grasp and not tools["plan_grasp"].physical
    assert grasp["object"]["type"] == "string" and "enum" not in grasp["object"]   # any object
    assert tools["place_held_item"].parameters["surface"]["enum"] == SURFACES
    assert tools["move_arm_joint"].parameters["joint"]["enum"] == ["lj0", "rj1"]


def test_object_spec_carries_the_models_grasp_choices():
    s = object_spec("tv remote", grasp="side", align_wrist=True, grip_at="center",
                    handle=True, shape="box", squeeze_mm=99, arm="left")
    assert (s.name, s.grasp, s.aligned, s.grip_at_center, s.shape) == \
        ("tv remote", "side", True, True, "box")
    assert s.handle_geom is not None and s.squeeze == 0.030          # clamped to 30 mm
    assert object_spec("mug").grasp == "top" and object_spec("mug").handle_geom is None
    for bad in ({"grasp": "pinch"}, {"arm": "third"}, {"shape": "cube"}, {"grip_at": "base"}):
        with pytest.raises(ValueError, match="must be one of"):
            object_spec("mug", **bad)
    with pytest.raises(ValueError):
        object_spec("  ")


def test_looks_like_is_remembered_as_what_the_detector_asks_for():
    tools = ToolsWithoutSim()
    tools.detection = SimpleNamespace(aliases={})
    spec, sides = tools._grasp_args({"object": " keys ", "arm": "left",
                                     "looks_like": "small grey block on the floor"})
    assert (spec.name, sides) == ("keys", ("left",))
    assert tools.detection.aliases == {"keys": "small grey block on the floor"}
    tools._grasp_args({"object": "keys"})                 # later calls keep the description
    assert tools.detection.aliases["keys"] == "small grey block on the floor"


def test_scene_shortcuts():
    assert resolve_scene(None).name == "scene_home.xml"
    assert resolve_scene("apartment").name == "home_search.xml"


class ToolsWithoutSim(RobotTools):
    """RobotTools.call's plumbing, with no robot behind it."""

    def __init__(self, fallen=False):
        self.bot = SimpleNamespace(fallen=fallen)
        self.last_result = None
        self._tools = {"wiggle": Tool("wiggle", "test", {"n": {"type": "number"}}),
                       "peek": Tool("peek", "test", physical=False)}

    def pose(self):
        return {"x": 1.0}

    def tool_wiggle(self, n):
        if n < 0:
            raise ValueError("n must be positive")
        if n == 99:
            raise ZeroDivisionError("boom")
        return {"wiggled": n}

    def tool_peek(self):
        return {"ok": True, "saw": "a mug"}


def test_call_never_raises_and_reports_failures_in_words():
    tools = ToolsWithoutSim()
    assert tools.call("wiggle", '{"n": 2}') == {"wiggled": 2, "ok": True, "robot": {"x": 1.0}}
    assert tools.last_result == {"tool": "wiggle", "arguments": {"n": 2}, "wiggled": 2, "ok": True}
    assert tools.call("wiggle", {"n": -1})["error"] == "n must be positive"
    assert "ZeroDivisionError: boom" in tools.call("wiggle", {"n": 99})["error"]
    assert "valid JSON" in tools.call("wiggle", "{n: 2")["error"]
    assert "unexpected keyword" in tools.call("wiggle", {"n": 1, "speed": 3})["error"]
    assert "no tool called 'fly'" in tools.call("fly", {})["error"]
    assert tools.call("wiggle", {"n": 1, "unused": None})["ok"]      # nulls are dropped


def test_a_fallen_robot_refuses_to_move_but_still_answers_questions():
    tools = ToolsWithoutSim(fallen=True)
    assert "fallen over" in tools.call("wiggle", {"n": 1})["error"]
    assert tools.call("peek", {})["saw"] == "a mug"


def test_cli_requests_from_the_command_line_or_typed_until_quit():
    args = cli.parse_args(["put", "the", "mug", "away"])
    assert list(cli.requests_from(args)) == ["put the mug away"]
    typed = iter(["pick up the ball", "", "  tidy up ", "quit", "never reached"])
    assert list(cli.requests_from(cli.parse_args([]), ask=lambda p: next(typed))) == \
        ["pick up the ball", "tidy up"]
    assert cli.parse_args(["--scene", "apartment"]).scene == "apartment"


def test_cli_voice_requests_are_transcribed_until_quit(monkeypatch, capsys):
    from vision_sim import speech
    heard = iter(["put the ball in the basket", "", "Quit."])
    monkeypatch.setattr(speech.SpeechToText, "load", lambda self: None)
    monkeypatch.setattr(speech, "listen", lambda stt, seconds, ask=input, device=None: next(heard))
    args = cli.parse_args(["--voice", "--whisper-model", "tiny.en", "--mic", "2"])
    assert list(cli.requests_from(args)) == ["put the ball in the basket"]
    assert "(heard nothing)" in capsys.readouterr().out


def test_cli_lists_tools(capsys):
    cli.main(["--list-tools"])
    out = capsys.readouterr().out
    assert "pick_up(object, grasp, arm" in out and "go_to(x, y, heading_deg)" in out


# ---------------------------------------------------------- the real robot
needs_yolo = pytest.mark.skipif(importlib.util.find_spec("ultralytics") is None,
                                reason="ultralytics (YOLO) is not installed")


@pytest.fixture(scope="module")
def living_room():
    tools = RobotTools(truth=True, verbose=False)
    yield tools
    tools.close()


@pytest.mark.integration
@pytest.mark.slow
def test_real_robot_base_motion_and_arms(living_room):
    robot = living_room
    turned = robot.call("turn", {"degrees": 90})
    assert turned["ok"] and abs(turned["turned_deg"] - 90) < 12, turned
    moved = robot.call("move", {"distance_m": 0.5})
    assert moved["ok"], moved
    assert robot.call("turn", {"degrees": -90})["ok"]
    assert robot.call("move", {"distance_m": -0.5})["ok"]
    lift = robot.arm_joints[0]
    arm = robot.call("move_arm_joint", {"joint": lift, "value": 10.0})
    assert arm["target"] == arm["range"][1]
    assert robot.call("stow_arms", {})["ok"]
    assert robot.call("set_gripper", {"side": "both", "action": "close"})["ok"]
    assert robot.call("set_gripper", {"side": "both", "action": "open"})["ok"]
    assert not robot.bot.fallen


@pytest.mark.integration
@pytest.mark.slow
def test_real_robot_grasp_choices_pick_and_place_on_a_found_surface(living_room):
    robot = living_room
    assert {"coffee_table", "side_table", "basket", "person"} <= set(robot.surfaces)
    assert robot.surfaces["basket"].mode == "drop"
    boxy = robot.call("plan_grasp", {"object": "remote", "shape": "box", "align_wrist": True})
    assert boxy["ok"] or "wide" in boxy.get("error", "")
    plans = robot.call("plan_grasp", {"object": "can", "arm": "left"})
    assert plans["ok"] and {p["arm"] for p in plans["plans"]} == {"left"}, plans
    picked = robot.call("pick_up", {"object": "can"})
    assert picked["ok"], picked
    assert picked["phases"][-1] == "done" and "can" in robot.call("get_status", {})["holding"].values()
    assert "already holding" in robot.call("pick_up", {"object": "mug"})["error"]
    placed = robot.call("place_held_item", {"surface": "side_table"})
    assert placed["ok"], placed
    assert robot.held is None


def _llm_up():
    if os.environ.get("SKIP_LIVE_LLM"):          # CI sets it: no llama-server there
        return False
    try:
        import requests
        return requests.get("http://localhost:8080/v1/models", timeout=2).status_code == 200
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.slow
@needs_yolo
@pytest.mark.skipif(not _llm_up(), reason="the local LLM server is not running")
def test_camera_measures_objects_it_was_never_calibrated_for():
    """No colour windows: YOLO-World or the vision LLM boxes it, depth measures it.
    Viewpoints as in the pick benchmarks, 1.3-1.9 m away and facing the item."""
    import numpy as np
    from handwrist import scenarios
    from handwrist.detection import TruthByName
    robot = RobotTools(verbose=False)
    try:
        truth = TruthByName()
        for name, kw in (("mug", {"handle": True}), ("can", {}),
                         ("remote", {"shape": "box", "align_wrist": True}),
                         ("ball", {"shape": "sphere", "grip_at": "center"})):
            scenarios.setup(robot.bot, name, np.random.default_rng(1))
            spec = object_spec(name, **kw)
            est, true = robot.detection(robot.bot, spec), truth(robot.bot, spec)
            assert est is not None, name
            assert np.linalg.norm(est.center[:2] - true.center[:2]) < 0.015, name
            assert abs(est.width - true.width) < 0.01, name
    finally:
        robot.close()


@pytest.mark.integration
@pytest.mark.slow
def test_another_scene_finds_its_own_surfaces_and_picks_there():
    robot = RobotTools(scene="apartment", truth=True, verbose=False)
    try:
        assert {"table", "coffee_table", "sideboard", "bin", "chair_a"} <= set(robot.surfaces)
        assert not any(n.startswith("wall") for n in robot.surfaces)
        # the ball on the floor of the dining room, from 1.6 m south of it
        went = robot.call("go_to", {"x": 0.5, "y": 1.6, "heading_deg": 90})
        assert went["ok"], went
        picked = robot.call("pick_up", {"object": "ball", "shape": "sphere", "grip_at": "center"})
        assert picked["ok"], picked
        # through the doorway into the other room: the lead-in is planned with A*
        placed = robot.call("place_held_item", {"surface": "bin"})
        assert placed["ok"], placed
        assert placed.get("drove_to_lead_in"), placed
    finally:
        robot.close()
