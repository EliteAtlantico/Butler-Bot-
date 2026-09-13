"""pytest-only configuration for this directory's unittest suites.

These modules stay runnable with plain `python -m unittest`, so they cannot
import pytest to decorate themselves. This marks, from outside, the tests that
create an off-screen GL renderer, so CI can deselect them with
`-m "not rendering"` if headless GL proves flaky on a runner.

The list is measured, not guessed: a full run recorded every
mujoco.Renderer construction, update_scene and render call per test. Tests
that could not reach a renderer on the measuring machine (no ultralytics
there) were added from their source: each builds a bot and observes. A test
that starts rendering needs adding here; one listed that stops is harmless.
"""
from __future__ import annotations

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent

# module -> "Class::method", or "Class" for every test in the class
RENDERING = {
    "test_command_search.py": {
        "TestExplorePrompt::test_earlier_choices_are_remembered",
        "TestGivingUp::test_nowhere_to_go_gives_up_and_hands_over_once",
        "TestGivingUp::test_round_cap_gives_up_and_hands_over",
        "TestKeysSearch::test_keys_are_hidden_from_the_start",
        "TestKeysSearch::test_keys_found_in_the_other_room_and_reached",
    },
    "test_explore.py": {
        "TestExploreGeometry::test_a_waypoint_is_never_placed_into_a_near_wall",
        "TestExploreInterpret::test_already_explored_target_falls_back",
        "TestExploreNavigation::test_detector_seeing_the_goal_in_the_survey_skips_the_model",
        "TestExploreNavigation::test_hidden_goal_found_via_llm_waypoints_and_reached",
        "TestExploreNavigation::test_llm_detector_is_not_run_on_every_survey_photo",
        "TestExploreNavigation::test_reaching_a_waypoint_surveys_again",
        "TestExploreNavigation::test_step_cap_ends_in_stuck_without_arriving",
        "TestExploreNavigation::test_unreachable_waypoint_surveys_again",
        "TestExploreRequest::test_one_request_every_photo_labelled",
    },
    "test_goal_recovery.py": {
        "TestGoalVisibility",
        "TestRecoveryEndToEnd",
        "TestRecoveryLogic::test_clock_reset_clears_a_pending_miss",
    },
    "test_llm_scene_reasoning.py": {
        "TestNavigationEndToEnd",
        "TestPixelToWorld",
    },
    "test_llm_survey.py": {
        "TestNavigatorSurvey",
        "TestPixelHeading",
        "TestSurveyInterpret::test_goal_is_visible_from_start",
        "TestSurveyLabelsWithCamera",
        "TestSurveyRequest::test_http_error_is_a_clean_not_found",
    },
    "test_pretrained_yolo.py": {
        "TestFindsTheTarget",
        "TestCommandLine::test_explore_prompt_names_the_target",
    },
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        if item.path.parent != HERE:
            continue
        wanted = RENDERING.get(item.path.name)
        if not wanted:
            continue
        test = item.nodeid.split("::", 1)[1] if "::" in item.nodeid else ""
        if test in wanted or test.split("::")[0] in wanted:
            item.add_marker(pytest.mark.rendering)
