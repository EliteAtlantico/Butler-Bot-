from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bracketbot_sim import algorithms as alg


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0, 0), (np.pi, -np.pi), (-np.pi, -np.pi), (3 * np.pi, -np.pi),
     (-3 * np.pi, -np.pi), (2 * np.pi + 0.2, 0.2)],
)
def test_wrap(angle, expected):
    assert alg._wrap(angle) == pytest.approx(expected)


def test_stand_never_finishes_and_stops(drive_bot):
    stand = alg.Stand()
    stand(drive_bot, 12.0)
    assert drive_bot.commands == [(0.0, 0.0)]
    assert stand.done is False


def test_drive_uses_relative_start_time_and_stops_at_boundary(drive_bot):
    drive = alg.Drive(0.4, -0.2, duration=2.0)
    assert drive.done is False
    drive(drive_bot, 7.0)
    drive(drive_bot, 8.999)
    drive(drive_bot, 9.0)
    assert drive_bot.commands == [(0.4, -0.2), (0.4, -0.2), (0.0, 0.0)]
    assert drive.t0 == 7.0
    assert drive.done is True


def test_waypoint_empty_is_immediately_done(drive_bot):
    follower = alg.WaypointFollower([])
    follower(drive_bot, 0)
    assert follower.done
    assert drive_bot.commands[-1] == (0.0, 0.0)


def test_waypoint_reached_advances_and_stops_for_tick(drive_bot):
    follower = alg.WaypointFollower([(0.1, 0.0), (1.0, 0.0)], tol=0.2)
    follower(drive_bot, 0)
    assert follower.i == 1
    assert drive_bot.commands[-1] == (0.0, 0.0)


def test_waypoint_loop_wraps_to_first(drive_bot):
    follower = alg.WaypointFollower([(0.0, 0.0)], tol=0.1, loop=True)
    follower(drive_bot, 0)
    assert follower.i == 0
    assert not follower.done


def test_waypoint_turns_before_driving_when_goal_is_behind(drive_bot):
    follower = alg.WaypointFollower([(-1.0, 0.0)], v_max=0.5, w_max=0.8)
    follower(drive_bot, 0)
    v, w = drive_bot.commands[-1]
    assert v == 0.0
    assert abs(w) == pytest.approx(0.8)


def test_waypoint_scales_speed_near_goal(drive_bot):
    follower = alg.WaypointFollower([(0.3, 0.0)], v_max=0.6, tol=0.01)
    follower(drive_bot, 0)
    assert drive_bot.commands[-1] == pytest.approx((0.3, 0.0))


def test_waypoint_clamps_turn_rate(drive_bot):
    follower = alg.WaypointFollower([(0.0, 2.0)], w_max=0.4, heading_gain=10)
    follower(drive_bot, 0)
    assert drive_bot.commands[-1][1] == pytest.approx(0.4)


def test_obstacle_sense_splits_image_and_ignores_nonfinite(drive_bot):
    depth = np.full((4, 6), np.inf)
    depth[1, 0] = np.nan
    depth[2, 1] = 1.2
    depth[1, 4] = 2.3
    drive_bot.depth = lambda *_args, **_kwargs: depth
    avoid = alg.ObstacleAvoider(width=6, height=4, band=(0, 1))
    assert avoid.sense(drive_bot) == pytest.approx((1.2, 2.3))
    assert avoid.last_depth is depth


def test_obstacle_clear_path_cruises(drive_bot):
    avoid = alg.ObstacleAvoider(v_cruise=0.4, clear_range=2.0, stop_range=1.0)
    avoid.sense = lambda _bot: (4.0, 4.0)
    avoid(drive_bot, 0.0)
    assert drive_bot.commands[-1] == pytest.approx((0.4, 0.0))


@pytest.mark.parametrize(
    ("depths", "turn_sign"), [((0.5, 1.5), -1), ((1.5, 0.5), 1), ((0.5, 0.5), -1)]
)
def test_obstacle_too_close_rotates_toward_open_side(drive_bot, depths, turn_sign):
    avoid = alg.ObstacleAvoider(w_max=1.2, stop_range=1.0)
    avoid.sense = lambda _bot: depths
    avoid(drive_bot, 0.0)
    assert drive_bot.commands[-1] == pytest.approx((0.0, turn_sign * 1.2))


def test_obstacle_urgent_path_slows_and_steers(drive_bot):
    avoid = alg.ObstacleAvoider(v_cruise=1.0, w_max=1.0,
                                clear_range=2.0, stop_range=1.0)
    avoid.sense = lambda _bot: (1.5, 1.2)
    avoid(drive_bot, 0.0)
    v, w = drive_bot.commands[-1]
    assert 0.3 < v < 1.0
    assert w > 0


def test_obstacle_respects_period_and_reuses_command(drive_bot):
    avoid = alg.ObstacleAvoider(period=1.0)
    calls = []
    avoid.sense = lambda _bot: calls.append(1) or (4.0, 4.0)
    avoid(drive_bot, 5.0)
    avoid(drive_bot, 5.5)
    assert len(calls) == 1
    assert drive_bot.commands[-1] == drive_bot.commands[-2]


def test_obstacle_clock_reset_forces_new_sense(drive_bot):
    avoid = alg.ObstacleAvoider(period=10.0)
    calls = []
    avoid.sense = lambda _bot: calls.append(1) or (4.0, 4.0)
    avoid(drive_bot, 5.0)
    avoid(drive_bot, 5.1)
    avoid(drive_bot, 0.0)
    assert len(calls) == 2


def test_sequence_advances_and_then_stops(drive_bot):
    seq = alg.Sequence(alg.Drive(1, 0, 0), alg.Drive(0, 1, 0))
    assert seq.current is seq.steps[0]
    seq(drive_bot, 0)
    assert seq.i == 1
    seq(drive_bot, 0)
    assert seq.done and seq.current is None
    seq(drive_bot, 1)
    assert drive_bot.commands[-1] == (0.0, 0.0)


def test_square_patrol_shape_and_options():
    patrol = alg.square_patrol(side=2.5, v_max=0.2)
    assert patrol.loop
    assert patrol.v_max == 0.2
    assert np.array_equal(patrol.waypoints,
                          np.array([(2.5, 0), (2.5, 2.5), (0, 2.5), (0, 0)]))


def test_local_map_empty_shapes(drive_bot):
    local = alg.LocalMap()
    assert local.points().shape == (0, 2)
    bearings, ranges = local.scan(drive_bot, n_bins=7)
    assert bearings.shape == ranges.shape == (7,)
    assert np.isinf(ranges).all()


def test_local_map_update_filters_height_and_self(drive_bot):
    drive_bot.cloud = np.array([
        [1.02, 0.02, 0.5], [1.04, 0.01, 0.6],
        [0.1, 0.0, 0.5], [2.0, 0.0, 0.01], [3.0, 0.0, 2.0],
    ])
    local = alg.LocalMap(cell=0.1, self_radius=0.3)
    local.update(drive_bot, 4.0)
    assert len(local.cells) == 1
    assert (10, 0) in local.cells
    assert local.cells[(10, 0)] == 4.0


def test_local_map_prunes_old_and_far_cells(drive_bot):
    local = alg.LocalMap(cell=1.0, max_age=5.0, radius=3.0)
    local.cells = {(1, 0): 0.0, (2, 0): 9.0, (4, 0): 9.0}
    local._prune(drive_bot, 10.0)
    assert local.cells == {(2, 0): 9.0}


def test_local_map_scan_uses_robot_frame_and_nearest(drive_bot):
    local = alg.LocalMap(cell=1.0)
    local.cells = {(1, 0): 0, (2, 0): 0, (0, 2): 0}
    bearings, ranges = local.scan(drive_bot, n_bins=5, half_fov=np.pi / 2)
    assert np.nanmin(ranges) == pytest.approx(1.0)
    drive_bot.yaw = np.pi / 2
    _, turned = local.scan(drive_bot, n_bins=5, half_fov=np.pi / 2)
    assert np.isfinite(turned).any()


def test_navigate_plan_points_at_goal_with_empty_map(drive_bot):
    nav = alg.NavigateTo((2, 1), n_candidates=81)
    nav.map.update = lambda *_args: None
    nav.map.points = lambda: np.empty((0, 2))
    nav._plan(drive_bot, np.arctan2(1, 2))
    assert nav._steer == pytest.approx(np.arctan2(1, 2), abs=0.05)
    assert nav._ahead == pytest.approx(nav.lookahead)
    assert not nav._goal_blocked


def test_navigate_plan_steers_around_blocked_goal_corridor(drive_bot):
    nav = alg.NavigateTo((2, 0), n_candidates=81)
    nav.map.update = lambda *_args: None
    nav.map.points = lambda: np.array([[0.6, 0.0]])
    nav._plan(drive_bot, 0.0)
    assert abs(nav._steer) > 0.1
    assert nav._goal_blocked


def test_navigate_position_goal_arrival(drive_bot):
    nav = alg.NavigateTo((0.01, 0), standoff_tol=0.05)
    nav(drive_bot, 0)
    assert nav.done
    assert drive_bot.commands[-1] == (0.0, 0.0)


def test_navigate_cruise_commands_forward(drive_bot):
    nav = alg.NavigateTo((2, 0), v_max=0.3)
    nav.map.update = lambda *_args: None
    nav.map.points = lambda: np.empty((0, 2))
    nav(drive_bot, 0)
    assert drive_bot.commands[-1] == pytest.approx((0.3, 0), abs=0.05)


def test_navigate_clock_reset_resets_schedule_and_stuck_timer(drive_bot):
    nav = alg.NavigateTo((2, 0), period=10)
    nav.map.update = lambda *_args: None
    nav.map.points = lambda: np.empty((0, 2))
    nav(drive_bot, 5)
    nav._stuck_since = 4
    nav(drive_bot, 0)
    assert nav._next == pytest.approx(10)
    assert nav._stuck_since == 0


def test_navigate_pose_entry_point():
    nav = alg.NavigateTo((2, 3), goal_yaw=np.pi / 2)
    assert nav._entry_point() == pytest.approx((2, 3 - nav.entry_offset))


def test_navigate_pose_transitions_to_turn(drive_bot):
    nav = alg.NavigateTo((1, 0), goal_yaw=0)
    drive_bot.position[:2] = nav._entry_point()
    nav(drive_bot, 0)
    assert nav.stage == "turn"


def test_navigate_pose_arrival_requires_low_speed(drive_bot):
    nav = alg.NavigateTo((0, 0), goal_yaw=0, standoff_tol=0.1)
    nav.stage = "turn"
    drive_bot.ground_speed = 0.1
    nav(drive_bot, 0)
    assert not nav.done
    drive_bot.ground_speed = 0.0
    nav(drive_bot, 0.1)
    assert nav.done


def test_navigate_pose_at_position_rotates_only(drive_bot):
    nav = alg.NavigateTo((0, 0), goal_yaw=0.3, standoff_tol=0.1)
    nav.stage = "turn"
    nav(drive_bot, 0)
    assert drive_bot.commands[-1] == pytest.approx((0, 0.4))


def test_navigate_pose_recruises_if_displaced(drive_bot):
    nav = alg.NavigateTo((2, 0), goal_yaw=0)
    nav.stage = "turn"
    nav(drive_bot, 0)
    assert nav.stage == "cruise"
    assert drive_bot.commands[-1] == (0, 0)


def test_navigate_pose_can_reverse_toward_goal_behind(drive_bot):
    nav = alg.NavigateTo((-0.4, 0), goal_yaw=0)
    nav.stage = "turn"
    nav(drive_bot, 0)
    assert drive_bot.commands[-1][0] < 0


def test_navigate_cruise_stops_at_close_obstacle(drive_bot):
    nav = alg.NavigateTo((2, 0), stop_range=0.5)
    nav._next = 100
    nav._ahead = 0.2
    nav._steer = 0.4
    nav._cruise(drive_bot, 1, np.array([2.0, 0]), 2)
    assert drive_bot.commands[-1][0] == 0
    assert drive_bot.commands[-1][1] > 0


def test_navigate_stuck_recovery_backs_up(drive_bot):
    nav = alg.NavigateTo((2, 0))
    nav._next = 100
    nav._ahead = nav.lookahead
    nav._steer = 0.2
    nav._stuck_since = 0.0
    nav._cruise(drive_bot, 2.0, np.array([2.0, 0]), 2)
    assert drive_bot.commands[-1] == pytest.approx((-0.22, nav.w_max))
