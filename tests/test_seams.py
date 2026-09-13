"""Phase 6: the seams between the four stacks, which no subsystem suite covers.

Seam D (give-up -> remote) lives in test_remote_handoff.py.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "comp_vision_sim"
DEMO_SCENE = VISION / "home_search.xml"          # D2: the end-to-end demo scene


# ------------------------------------------------------------------ Phase 2
@pytest.mark.parametrize("module", [
    "bracketbot_sim.robot",          # main_mujoco
    "vision_sim.navigation",         # comp_vision_sim
    "handwrist.skills",              # Hand_and_Wrists
    "remote_control.server",         # remote_control
    "integration.pick_adapter",      # Seam C
    "integration.remote_adapter",    # Seam D
])
def test_every_package_imports_from_the_repo_root(module):
    importlib.import_module(module)


# ------------------------------------------------------------------ Phase 3
def test_command_prompt_does_not_say_the_robot_cannot_pick():
    from vision_sim.llm_command import COMMAND_PROMPT

    assert "cannot pick" not in COMMAND_PROMPT.lower()
    assert "pick up" in COMMAND_PROMPT


# ------------------------------------------------------------------- Seam C
@pytest.mark.parametrize("spoken, name", [
    ("the tv remote", "remote"), ("the remote control", "remote"), ("my mug", "mug"),
    ("Coffee Cup!", "mug"), ("mugs", "mug"), ("a tennis ball", "ball"),
])
def test_navigator_targets_resolve_to_catalogue_names(spoken, name):
    from integration.pick_adapter import resolve_object

    assert resolve_object(spoken) == name


@pytest.mark.parametrize("spoken", ["my sunglasses", "", None])
def test_unknown_targets_resolve_to_nothing(spoken):
    from integration.pick_adapter import resolve_object

    assert resolve_object(spoken) is None


@pytest.mark.integration
def test_every_catalogue_object_is_graspable_in_the_demo_scene():
    import mujoco
    from handwrist.objects import CATALOGUE, ObjectSpec

    model = mujoco.MjModel.from_xml_path(str(DEMO_SCENE))
    for name, spec in CATALOGUE.items():
        assert isinstance(spec, ObjectSpec) and spec.name == name
        for geom in (spec.grasp_geom, spec.handle_geom):
            if geom is not None:
                assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom) >= 0, \
                    f"{name}: geom {geom!r} missing from {DEMO_SCENE.name}"


@pytest.mark.integration
def test_arrival_builds_a_pick_on_the_catalogue_spec_with_the_head_camera():
    from bracketbot_sim.robot import BracketBot
    from handwrist.detection import DetectionEstimator
    from handwrist.objects import CATALOGUE
    from handwrist.skills import Pick
    from integration.pick_adapter import make_pick

    bot = BracketBot(xml=DEMO_SCENE)
    try:
        pick = make_pick(bot, "remote")
        assert isinstance(pick, Pick) and pick.bot is bot
        assert pick.spec is CATALOGUE["remote"]
        # found by what it is, not by a calibrated colour
        assert isinstance(pick.estimator, DetectionEstimator)
        assert pick.estimator.aliases == {}
    finally:
        bot.close()


def test_pick_after_arrival_hands_pick_the_resolved_name():
    from handwrist.detection import DetectionEstimator
    from integration.pick_adapter import pick_after_arrival

    bot = SimpleNamespace(time=0.0)
    done = SimpleNamespace(done=True, succeeded=True, failure=None, phase="done")
    with mock.patch("handwrist.skills.Pick", return_value=done) as pick:
        outcome = pick_after_arrival(bot, "the tv remote", verbose=False)

    args, kwargs = pick.call_args
    assert args == (bot, "remote") and isinstance(kwargs["estimator"], DetectionEstimator)
    assert (outcome.attempted, outcome.succeeded, outcome.object_name) == (True, True, "remote")


def test_a_colour_in_the_request_reaches_the_detector_as_a_hint():
    """"the red mug" is still the mug; red only ranks what the detector finds."""
    from integration.pick_adapter import pick_after_arrival

    done = SimpleNamespace(done=True, succeeded=True, failure=None, phase="done")
    with mock.patch("handwrist.skills.Pick", return_value=done) as pick:
        pick_after_arrival(SimpleNamespace(time=0.0), "the red mug", verbose=False)
    args, kwargs = pick.call_args
    assert args[1] == "mug" and kwargs["estimator"].aliases == {"mug": "red mug"}


def test_pick_after_arrival_skips_what_the_arm_cannot_hold():
    from integration.pick_adapter import pick_after_arrival

    with mock.patch("handwrist.skills.Pick", side_effect=AssertionError("Pick built")):
        outcome = pick_after_arrival(SimpleNamespace(time=0.0), "my sunglasses", verbose=False)
    assert not outcome.attempted and "sunglasses" in outcome.skipped


def _arrived_nav(bot):
    return SimpleNamespace(
        state="arrived", done=True, give_up_reason=None, ARRIVED="arrived",
        goal_label="target", goal_xy=None, obs=None, best_time=0.0, best_detections=[],
        detector=None, survey=None, researches=0, explorer=None, outcome="found", bot=bot)


@pytest.mark.integration
def test_run_navigation_hands_its_own_bot_to_the_arm_on_arrival(monkeypatch):
    if str(VISION) not in sys.path:
        sys.path.insert(0, str(VISION))
    import run_navigation as rn
    from integration.pick_adapter import PickOutcome
    from vision_sim.navigation import VisualNavigator

    navs = []

    def fake_for_bot(bot, **_kw):
        navs.append(_arrived_nav(bot))
        return navs[-1]

    monkeypatch.setattr(sys, "argv", ["run_navigation.py", "--goal", "4,0", "--pick",
                                      "--target", "a set of keys"])
    outcome = PickOutcome(attempted=True, target="a set of keys", object_name="keys",
                          succeeded=True)
    with mock.patch.object(VisualNavigator, "for_bot", side_effect=fake_for_bot), \
            mock.patch.object(rn, "build_detector", lambda args, info=None: None), \
            mock.patch("integration.pick_adapter.pick_after_arrival",
                       return_value=outcome) as pick, \
            mock.patch("integration.remote_adapter.hand_off") as hand_off:
        rn.main()

    pick.assert_called_once()
    assert pick.call_args.args == (navs[0].bot, "a set of keys")
    assert pick.call_args.kwargs["use_camera"] is True
    hand_off.assert_not_called()                  # arrived, so no give-up hand-off
