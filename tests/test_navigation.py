from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vision_sim import navigation, perception


class FakeGrid:
    resolution = 1.0
    origin = (0.0, 0.0)
    size = (10, 10)

    def __init__(self):
        self.integrations = []
        self.blocked = np.zeros(self.size, bool)
        self.soft = np.zeros(self.size, float)
        self.radius = None

    def integrate(self, camera, hits, free):
        self.integrations.append((camera.copy(), hits.copy(), free.copy()))

    def costmap(self, robot_radius):
        self.radius = robot_radius
        return self.blocked.copy(), self.soft.copy()

    def to_cell(self, xy):
        return np.atleast_2d(np.floor(xy).astype(int))

    def to_world(self, cell):
        return np.atleast_2d(np.asarray(cell, float) + 0.5)

    def inside(self, cell):
        cell = np.atleast_2d(cell)
        return ((cell[:, 0] >= 0) & (cell[:, 0] < self.size[0]) &
                (cell[:, 1] >= 0) & (cell[:, 1] < self.size[1]))


class Bot:
    def __init__(self, xy=(1, 1), yaw=0):
        self.position = np.array([xy[0], xy[1], 0.0], float)
        self.yaw = yaw
        self.commands = []

    def drive(self, v, w):
        self.commands.append((float(v), float(w)))


def observation(cam=(0, 0, 1)):
    return perception.Observation(
        rgb=np.zeros((2, 2, 3), np.uint8), depth=np.ones((2, 2), np.float32),
        points=np.ones((2, 2, 3)), valid=np.ones((2, 2), bool),
        cam_pos=np.asarray(cam, float), cam_mat=np.eye(3),
        intrinsics=perception.Intrinsics(1, 1, 1, 1, 2, 2),
        robot_xy=np.zeros(2),
    )


def detection(label="target", pos=(4, 0, .5), pixels=10, distance=2):
    return perception.Detection(label, np.asarray(pos, float), distance, 0,
                                np.ones(3), pixels)


def test_initial_state_and_done_property():
    nav = navigation.VisualNavigator(grid=FakeGrid())
    assert nav.state == nav.SCAN and not nav.done
    nav.state = nav.ARRIVED
    assert nav.done
    nav.state = nav.STUCK
    assert nav.done


def test_sense_runs_pipeline_updates_map_and_best_frame(monkeypatch):
    obs = observation()
    dets = [detection(), detection("pillar", pixels=20)]
    monkeypatch.setattr(perception, "observe", lambda *_a, **_k: obs)
    monkeypatch.setattr(perception, "obstacle_points", lambda _o, **_k: np.array([[1, 2, 3]]))
    monkeypatch.setattr(perception, "floor_points", lambda _o, **_k: np.array([[4, 5, 0]]))
    calls = []
    # `t` is passed so time-gated detectors can pace themselves on sim time.
    detector = lambda frame, robot_yaw, t=None: calls.append((frame, robot_yaw)) or dets
    grid = FakeGrid()
    bot = Bot(yaw=0.4)
    nav = navigation.VisualNavigator(grid=grid, detector=detector, width=4, height=3)
    nav.sense(bot, 7.0)
    assert nav.obs is obs and nav.detections == dets
    assert calls == [(obs, 0.4)]
    assert len(grid.integrations) == 1
    assert nav.best_obs is obs and nav.best_detections == dets and nav.best_time == 7


def test_sense_goal_uses_largest_detection_and_nudges_from_camera(monkeypatch):
    obs = observation(cam=(0, 0, 1))
    dets = [detection(pos=(2, 0, .5), pixels=5), detection(pos=(0, 3, .5), pixels=20)]
    monkeypatch.setattr(perception, "observe", lambda *_a, **_k: obs)
    monkeypatch.setattr(perception, "obstacle_points", lambda _o, **_k: np.empty((0, 3)))
    monkeypatch.setattr(perception, "floor_points", lambda _o, **_k: np.empty((0, 3)))
    nav = navigation.VisualNavigator(grid=FakeGrid(), detector=lambda *_a, **_k: dets)
    nav.sense(Bot(), 0)
    assert nav.goal_xy == pytest.approx([0, 3.2])


def test_sense_smooths_existing_goal(monkeypatch):
    obs = observation()
    monkeypatch.setattr(perception, "observe", lambda *_a, **_k: obs)
    monkeypatch.setattr(perception, "obstacle_points", lambda _o, **_k: np.empty((0, 3)))
    monkeypatch.setattr(perception, "floor_points", lambda _o, **_k: np.empty((0, 3)))
    nav = navigation.VisualNavigator(grid=FakeGrid(),
        detector=lambda *_a, **_k: [detection(pos=(2, 0, .5))])
    # Within relocate_distance of the sighting, so the estimate is averaged in.
    # A sighting further off than that is treated as the goal having moved and
    # replaces it outright instead of crawling toward the midpoint.
    nav.goal_xy = np.array([2.5, 0.0])
    nav.sense(Bot(), 0)
    assert nav.goal_xy == pytest.approx(0.7 * np.array([2.5, 0]) + 0.3 * np.array([2.2, 0]))


def test_sense_does_not_replace_best_frame_with_fewer_detections(monkeypatch):
    obs = observation()
    monkeypatch.setattr(perception, "observe", lambda *_a, **_k: obs)
    monkeypatch.setattr(perception, "obstacle_points", lambda _o, **_k: np.empty((0, 3)))
    monkeypatch.setattr(perception, "floor_points", lambda _o, **_k: np.empty((0, 3)))
    nav = navigation.VisualNavigator(grid=FakeGrid(), detector=lambda *_a, **_k: [])
    old = observation((1, 1, 1))
    nav.best_obs, nav.best_detections, nav.best_time = old, [detection()], 2
    nav.sense(Bot(), 3)
    assert nav.best_obs is old and nav.best_time == 2


def test_replan_requires_goal():
    assert not navigation.VisualNavigator(grid=FakeGrid()).replan(Bot())


def test_replan_rejects_robot_exactly_at_goal():
    nav = navigation.VisualNavigator(grid=FakeGrid())
    nav.goal_xy = np.array([1.0, 1.0])
    assert not nav.replan(Bot((1, 1)))


def test_replan_success_shortcuts_and_converts_cells(monkeypatch):
    grid = FakeGrid()
    nav = navigation.VisualNavigator(grid=grid, stop_distance=1)
    nav.goal_xy = np.array([8.0, 1.0])
    monkeypatch.setattr(navigation.planning, "astar",
                        lambda blocked, start, goal, soft, soft_weight=None:
                        [(1, 1), (3, 1), (7, 1)])
    monkeypatch.setattr(navigation.planning, "shortcut", lambda blocked, cells: cells)
    assert nav.replan(Bot((1, 1)))
    assert len(nav.path) == 2
    assert nav.path[0] == pytest.approx([3.5, 1.5])
    assert nav._wp == 0 and nav._plan_failures == 0
    assert nav.blocked is not None and grid.radius == nav.robot_radius


def test_replan_frees_area_around_blocked_start(monkeypatch):
    grid = FakeGrid()
    grid.blocked[:] = True
    nav = navigation.VisualNavigator(grid=grid)
    nav.goal_xy = np.array([8.0, 8.0])
    captured = {}

    def astar(blocked, start, goal, soft, soft_weight=None):
        captured["blocked"] = blocked
        return [tuple(start), tuple(goal)]

    monkeypatch.setattr(navigation.planning, "astar", astar)
    monkeypatch.setattr(navigation.planning, "shortcut", lambda _b, p: p)
    assert nav.replan(Bot((1, 1)))
    assert not captured["blocked"][0:4, 0:4].any()


def test_replan_fallback_path_when_astar_returns_only_start(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), stop_distance=1)
    nav.goal_xy = np.array([5.0, 1.0])
    monkeypatch.setattr(navigation.planning, "astar", lambda *_a, **_k: [(1, 1)])
    monkeypatch.setattr(navigation.planning, "shortcut", lambda _b, p: p)
    nav.replan(Bot())
    assert nav.path == [pytest.approx([4.0, 1.0])]


def test_replan_eight_failures_marks_stuck(monkeypatch):
    # A fixed goal is a coordinate the caller asked for, so there is nothing to
    # re-search: running out of plans is terminal. With a detected goal the
    # navigator looks again instead, which the explore/re-search tests cover.
    nav = navigation.VisualNavigator(grid=FakeGrid(), goal=(8.0, 8.0))
    nav.goal_xy = np.array([8.0, 8.0])
    monkeypatch.setattr(navigation.planning, "astar", lambda *_a, **_k: None)
    for _ in range(8):
        assert not nav.replan(Bot())
    assert nav.state == nav.STUCK and nav.done


def test_displaced_uses_hysteresis():
    # Displacement is movement from where the robot settled, not distance to
    # the goal: a robot stopped at an unreachable goal is not displaced.
    nav = navigation.VisualNavigator(grid=FakeGrid(), stop_distance=1,
                                     re_engage_margin=.75)
    nav.goal_xy = np.array([0.0, 0.0])
    nav._settle(Bot((1.2, 0)), nav.ARRIVED)
    assert not nav._displaced(Bot((1.8, 0)))
    assert nav._displaced(Bot((2.0, 0)))
    # Never settled anywhere, so there is no reference to be displaced from.
    nav._settled_at = None
    assert not nav._displaced(Bot((99, 99)))


def test_follow_advances_close_intermediate_waypoint():
    nav = navigation.VisualNavigator(grid=FakeGrid(), waypoint_tol=.3)
    nav.path = [np.array([1.1, 1.0]), np.array([2.0, 1.0])]
    v, w = nav._follow(Bot((1, 1)))
    assert nav._wp == 1 and v > 0 and w == pytest.approx(0)


def test_follow_turns_before_translating_for_goal_behind():
    nav = navigation.VisualNavigator(grid=FakeGrid())
    nav.path = [np.array([-1.0, 0.0])]
    v, w = nav._follow(Bot((0, 0)))
    assert v == 0 and abs(w) == nav.w_max


def test_scan_initialises_and_commands_spin(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    bot = Bot(yaw=0.2)
    nav(bot, 0)
    assert nav._scan_start == 0 and nav._last_yaw == .2
    assert bot.commands[-1] == (0, nav.scan_rate)


def test_scan_wraps_yaw_increment_across_pi(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    bot = Bot(yaw=np.pi - .1)
    nav(bot, 0)
    bot.yaw = -np.pi + .1
    nav(bot, .1)
    assert nav._scan_yaw == pytest.approx(.2)


def test_completed_scan_without_goal_restarts(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    nav._scan_start, nav._last_yaw = 0, 0
    nav._scan_yaw = 2 * np.pi
    nav(Bot(), 1)
    assert nav._scan_yaw == 0 and nav.state == nav.SCAN


def test_completed_scan_with_goal_replans_and_navigates(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    monkeypatch.setattr(nav, "replan", lambda _bot: True)
    nav.goal_xy = np.array([3.0, 0])
    nav._scan_start, nav._last_yaw = 0, 0
    nav._scan_yaw = 2 * np.pi
    nav(Bot(), 1)
    assert nav.state == nav.NAVIGATE and "scan done" in nav.log[-1]


def test_navigate_arrival_stops_and_logs(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    nav.state = nav.NAVIGATE
    nav.goal_xy = np.array([1.1, 1])
    bot = Bot((1, 1))
    nav(bot, 0)
    assert nav.state == nav.ARRIVED
    assert bot.commands[-1] == (0, 0) and "arrived" in nav.log[-1]


def test_navigate_periodic_replan_and_follow(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10, plan_period=2)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    calls = []
    monkeypatch.setattr(nav, "replan", lambda _b: calls.append(1) or True)
    monkeypatch.setattr(nav, "_follow", lambda _b: (.2, -.3))
    nav.state = nav.NAVIGATE
    nav.goal_xy = np.array([9., 9.])
    nav.path = [np.array([2., 2.])]
    bot = Bot()
    nav(bot, 0)
    nav(bot, 1)
    assert len(calls) == 1 and bot.commands[-1] == (.2, -.3)


@pytest.mark.parametrize("terminal", [navigation.VisualNavigator.ARRIVED,
                                      navigation.VisualNavigator.STUCK])
def test_terminal_state_reengages_when_displaced(monkeypatch, terminal):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10)
    monkeypatch.setattr(nav, "sense", lambda *_a: None)
    # Settle at the origin so the move to (0, 0) below is measured from
    # somewhere: displacement is relative to the settle point, not the goal.
    nav._settle(Bot((5, 5)), terminal)
    nav.goal_xy = np.array([9., 9.])
    monkeypatch.setattr(nav, "replan", lambda _bot: False)
    bot = Bot((0, 0))
    nav(bot, 3)
    assert nav.state == nav.NAVIGATE and nav._plan_failures == 0
    assert "re-engaging" in nav.log[-1]


def test_clock_reset_only_when_time_moves_backward(monkeypatch):
    nav = navigation.VisualNavigator(grid=FakeGrid(), sense_period=10, plan_period=10)
    calls = []
    monkeypatch.setattr(nav, "sense", lambda _b, t: calls.append(t))
    nav(Bot(), 5)
    nav(Bot(), 5.1)
    nav(Bot(), 0)
    assert calls == [5, 0]


def test_note_logs_and_optionally_prints(capsys):
    quiet = navigation.VisualNavigator(grid=FakeGrid(), verbose=False)
    quiet._note("quiet")
    assert capsys.readouterr().out == ""
    loud = navigation.VisualNavigator(grid=FakeGrid(), verbose=True)
    loud._note("hello")
    assert capsys.readouterr().out.strip() == "hello"
