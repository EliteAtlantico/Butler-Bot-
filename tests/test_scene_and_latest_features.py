from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from vision_sim import navigation, perception
from vision_sim.occupancy import OccupancyGrid
from vision_sim.scene import (SceneInfo, _floor_height, _geom_halfextent,
                              _robot_extent, _robot_geoms, _robot_root,
                              _scenery_bounds, pick_camera)
from vision_sim.yolo_detector import YoloDetector


ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "comp_vision_sim"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_nav = load_script("latest_run_navigation", VISION / "run_navigation.py")
tour = load_script("demo_tour_tests", VISION / "demo_tour.py")


def make_obs(points, depth=None, robot_xy=(0, 0), cam_pos=(0, 0, 1)):
    points = np.asarray(points, float)
    h, w = points.shape[:2]
    depth = np.ones((h, w), np.float32) if depth is None else np.asarray(depth, np.float32)
    return perception.Observation(
        rgb=np.zeros((h, w, 3), np.uint8), depth=depth, points=points,
        valid=np.isfinite(depth), cam_pos=np.asarray(cam_pos, float), cam_mat=np.eye(3),
        intrinsics=perception.Intrinsics(1, 1, w / 2, h / 2, w, h),
        robot_xy=np.asarray(robot_xy, float))


def test_scene_info_span_including_is_immutable_and_describes():
    info = SceneInfo((0, 1, 10, 6), floor_z=.5, robot_radius=.3,
                     robot_top=1.6, camera="front", map_range=8)
    assert info.span == (10, 5)
    same = info.including((5, 3), margin=2)
    wider = info.including((-4, 20), margin=1)
    assert same.bounds == info.bounds
    assert wider.bounds == (-5, 1, 10, 21)
    assert info.bounds == (0, 1, 10, 6)
    text = info.describe()
    assert "x[0.0,10.0]" in text and "floor z=0.50" in text
    assert "camera='front'" in text and "map_range=8.0m" in text


def test_occupancy_covering_rounds_up_and_enforces_one_cell():
    g = OccupancyGrid.covering((-1.2, 3.0, 1.21, 3.01), resolution=.5, hit=1.2)
    assert g.origin == (-1.2, 3.0) and g.size == (5, 1)
    assert g.hit == 1.2 and g.logodds.shape == (5, 1)


def test_pick_camera_prefers_named_fixed_camera():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><camera name='side'/><camera name='my_depth'/></worldbody></mujoco>")
    assert pick_camera(model) == "my_depth"
    assert pick_camera(model, prefer=("side",)) == "side"


def test_pick_camera_falls_back_and_rejects_camera_less_model():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody><camera name='odd'/></worldbody></mujoco>")
    assert pick_camera(model) == "odd"
    empty = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    with pytest.raises(ValueError, match="no cameras"):
        pick_camera(empty)


def robot_and_scenery_model():
    xml = """<mujoco><worldbody>
      <geom name='floor' type='plane' pos='0 0 .5' size='0 0 .1'/>
      <body name='wall' pos='4 2 1.5'><geom name='wall_geom' type='box' size='1 2 1'/></body>
      <body name='robot' pos='1 -1 .8'><freejoint/><geom name='robot_geom' type='sphere' size='.3'/>
        <camera name='front_depth'/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    return model, data


def test_robot_root_and_geom_masks_distinguish_static_scenery():
    model, _ = robot_and_scenery_model()
    root = _robot_root(model)
    assert mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root) == "robot"
    mask = _robot_geoms(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
             for i in np.where(mask)[0]}
    assert names == {"robot_geom"}
    static = mujoco.MjModel.from_xml_string("<mujoco><worldbody><geom type='sphere' size='.1'/></worldbody></mujoco>")
    assert _robot_root(static) == 0 and not _robot_geoms(static).any()


def test_scene_measurement_helpers_find_floor_robot_and_bounds():
    model, data = robot_and_scenery_model()
    robot = _robot_geoms(model); static = ~robot
    floor = _floor_height(model, data, static)
    radius, top = _robot_extent(model, data, robot, floor)
    bounds = _scenery_bounds(model, data, static, floor)
    assert floor == pytest.approx(.5)
    assert radius == pytest.approx(.3, rel=.02)
    assert top == pytest.approx(.6, rel=.02)
    assert bounds == pytest.approx((3, 0, 5, 4))


def test_scene_helper_fallbacks_without_floor_robot_or_scenery():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    empty = np.zeros(model.ngeom, bool)
    assert _floor_height(model, data, empty) == 0
    assert _robot_extent(model, data, empty, 0) == (.3, 1.)
    assert _scenery_bounds(model, data, empty, 0) == (-1, -1, 1, 1)


def test_floor_height_falls_back_to_lowest_static_bound():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom type='box' pos='0 0 2' size='1 1 .5'/></worldbody></mujoco>")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    expected = 2.0 - float(model.geom_rbound[0])
    assert _floor_height(model, data, np.ones(model.ngeom, bool)) == pytest.approx(expected)


def test_geom_halfextent_respects_rotation_and_fallback_bound():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><geom name='box' type='box' size='1 2 3' euler='0 0 90'/></worldbody></mujoco>")
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")
    assert _geom_halfextent(model, data, gid) == pytest.approx([2, 1, 3], abs=1e-6)
    fake_model = SimpleNamespace(geom_aabb=np.zeros((1, 6)), geom_rbound=np.array([2.]))
    fake_data = SimpleNamespace(geom_xmat=np.eye(3).reshape(1, 9))
    assert _geom_halfextent(fake_model, fake_data, 0) == pytest.approx([2, 2, 2])


def test_scene_info_from_model_measures_and_limits_range():
    model, data = robot_and_scenery_model()
    info = SceneInfo.from_model(model, data, margin=1, max_map_range=2)
    assert info.camera == "front_depth" and info.map_range == 2
    assert info.bounds[0] <= 0 and info.bounds[1] <= -2
    assert info.robot_radius > 0 and info.robot_top > 0


@pytest.mark.integration
@pytest.mark.parametrize("filename", ["obstacle_course.xml", "test_arena.xml",
                                       "apartment.xml", "moving_obstacle.xml"])
def test_every_vision_scene_compiles_and_is_auto_discovered(filename):
    model = mujoco.MjModel.from_xml_path(str(VISION / filename))
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    info = SceneInfo.from_model(model, data)
    assert info.span[0] > 0 and info.span[1] > 0
    assert info.robot_radius > 0 and info.robot_top > 0 and info.camera
    assert info.map_range > 0


def test_geometric_detector_empty_and_filtered_returns():
    pts = np.zeros((2, 2, 3)); pts[..., 2] = .5
    obs = make_obs(pts, depth=np.full((2, 2), np.inf))
    detector = perception.GeometricDetector(min_points=1)
    assert detector(obs) == []


def test_geometric_detector_clusters_sorts_and_uses_relative_floor():
    pts = np.array([[[1., 0, 1.5], [1.1, 0, 1.5],
                     [3., 0, 1.5], [3.1, 0, 1.5]]])
    obs = make_obs(pts, cam_pos=(0, 0, 1))
    detector = perception.GeometricDetector(label="thing", min_points=2,
        cluster_res=.3, self_radius=.1, floor_z=1.0, min_height=.1, max_height=1)
    detections = detector(obs, robot_yaw=.1)
    assert [d.label for d in detections] == ["thing", "thing"]
    assert detections[0].distance < detections[1].distance
    assert detections[0].pixels == 2 and detections[0].bbox == (0, 0, 1, 0)
    assert detections[0].bearing == pytest.approx(-.1)


def test_geometric_detector_drops_clusters_below_minimum():
    pts = np.array([[[1., 0, .5], [3., 0, .5]]])
    obs = make_obs(pts)
    detector = perception.GeometricDetector(min_points=2, cluster_res=.1, self_radius=.1)
    assert detector(obs) == []


def test_obstacle_and_floor_points_honor_nonzero_floor():
    pts = np.array([[[1, 0, 2.05], [2, 0, 2.5]]], float)
    obs = make_obs(pts)
    floor = perception.floor_points(obs, floor_z=2)
    obstacle = perception.obstacle_points(obs, floor_z=2, self_radius=.1)
    assert floor.shape == (1, 3) and floor[0, 2] == pytest.approx(2.05)
    assert obstacle.shape == (1, 3) and obstacle[0, 2] == pytest.approx(2.5)


def test_observe_auto_picks_camera(monkeypatch):
    model, data = robot_and_scenery_model()
    bot = SimpleNamespace(model=model, data=data, position=np.zeros(3),
        camera=lambda name, w, h: np.zeros((h, w, 3), np.uint8),
        depth=lambda name, w, h, max_range: np.ones((h, w), np.float32))
    obs = perception.observe(bot, camera=None, width=2, height=2)
    assert obs.intrinsics.width == 2 and obs.valid.all()


def test_yolo_converts_rgb_to_contiguous_bgr_before_prediction():
    seen = []
    detector = object.__new__(YoloDetector)
    detector.model = SimpleNamespace(predict=lambda image, **kw:
                                     seen.append((image, kw)) or [SimpleNamespace(boxes=[])])
    detector.names = {}; detector.conf=.3; detector.iou=.5; detector.imgsz=4
    detector.device="cpu"
    obs = make_obs(np.zeros((1, 1, 3)))
    obs.rgb = np.array([[[10, 20, 30]]], np.uint8)
    assert detector(obs) == []
    assert np.array_equal(seen[0][0], [[[30, 20, 10]]])
    assert seen[0][0].flags.c_contiguous


def test_visual_navigator_scene_configuration_and_fixed_goal():
    scene = SceneInfo((-2, -1, 8, 5), 1.0, .4, 1.8, "front", 4)
    nav = navigation.VisualNavigator(scene=scene, goal=(7, 2), max_range=12,
                                     resolution=.5)
    assert nav.camera == "front" and nav.floor_z == 1
    assert nav.obstacle_ceiling == pytest.approx(1.9)
    # The inflation margin is derived from the grid resolution (0.75 of a cell)
    # rather than a fixed 0.05, so it still covers the grid's own quantisation
    # if either value is retuned: .4 + .75 * .5 = .775.
    assert nav.robot_radius == pytest.approx(.775) and nav.self_radius == pytest.approx(.65)
    assert nav.max_range == 6 and nav.grid.origin == (-2., -1.)
    assert nav.grid.size == (20, 12) and np.array_equal(nav.fixed_goal, [7, 2])


def test_visual_navigator_label_goal_overrides_default():
    nav = navigation.VisualNavigator(goal="beacon")
    assert nav.fixed_goal is None and nav.goal_label == "beacon"


def test_visual_navigator_for_bot_expands_scene_for_coordinate(monkeypatch):
    info = SceneInfo((0, 0, 2, 2), 0, .3, 1, "cam", 3)
    monkeypatch.setattr(navigation.SceneInfo, "from_model", lambda *_a, **_k: info)
    bot = SimpleNamespace(model=object(), data=object())
    nav = navigation.VisualNavigator.for_bot(bot, goal=(10, 0), resolution=1)
    assert nav.grid.inside(nav.grid.to_cell((10, 0)))[0]
    assert np.array_equal(nav.fixed_goal, [10, 0])


def test_visual_navigator_fixed_goal_wins_over_detections(monkeypatch):
    obs = make_obs(np.array([[[2., 0, .5]]]))
    monkeypatch.setattr(perception, "observe", lambda *_a, **_k: obs)
    monkeypatch.setattr(perception, "obstacle_points", lambda *_a, **_k: np.empty((0, 3)))
    monkeypatch.setattr(perception, "floor_points", lambda *_a, **_k: np.empty((0, 3)))
    nav = navigation.VisualNavigator(goal=(8, 9), detector=lambda *_a, **_k: [])
    bot = SimpleNamespace(yaw=0., position=np.zeros(3))
    nav.sense(bot)
    assert nav.goal_xy is nav.fixed_goal and np.array_equal(nav.goal_xy, [8, 9])


def test_replan_off_grid_warns_once_and_eventually_sticks():
    grid = OccupancyGrid.covering((0, 0, 2, 2), resolution=1)
    # Fixed goal: an off-map coordinate the caller asked for is terminal, with
    # nothing to re-search. A detected goal would be looked for again instead.
    nav = navigation.VisualNavigator(grid=grid, verbose=False, goal=(20., 20.))
    nav.goal_xy = np.array([20., 20.])
    bot = SimpleNamespace(position=np.array([.5, .5, 0.]))
    for _ in range(8):
        assert not nav.replan(bot)
    assert nav.state == nav.STUCK and len(nav.log) == 1
    assert "outside the map" in nav.log[0] and "widen the grid" in nav.log[0]


def test_scene_has_truth_only_for_original_course():
    assert run_nav.scene_has_truth("x/obstacle_course.xml")
    assert not run_nav.scene_has_truth("x/apartment.xml")


def test_build_geometric_detector_uses_scene_geometry(capsys):
    info = SceneInfo((0, 0, 1, 1), 2.0, .4, 1.5, "cam", 4)
    args = SimpleNamespace(detector="geometric")
    det = run_nav.build_detector(args, info)
    assert isinstance(det, perception.GeometricDetector)
    assert det.floor_z == 2 and det.self_radius == pytest.approx(.65)
    assert "colour-free" in capsys.readouterr().out


def test_navigation_parse_coordinate_geometric_and_mover_args(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_navigation.py", "--goal", "4, 5",
                                      "--detector", "geometric", "--camera", "front",
                                      "--animate-movers", ".8"])
    args = run_nav.parse_args()
    assert args.goal == "4, 5" and args.detector == "geometric"
    assert args.camera == "front" and args.animate_movers == .8


def test_tour_banner_and_list(monkeypatch, capsys):
    tour.banner(1, tour.SCENARIOS[0])
    assert "[1/5]" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["demo_tour.py", "--list"])
    tour.main()
    output = capsys.readouterr().out
    assert "1. Colour detection" in output and "5. Moving obstacle" in output


def test_tour_runs_selected_scenario_headless(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(sys, "argv", ["demo_tour.py", "--only", "3", "--headless"])
    monkeypatch.setattr(tour.subprocess, "run", lambda cmd, cwd: calls.append((cmd, cwd)))
    tour.main()
    cmd, cwd = calls[0]
    assert "test_arena.xml" in " ".join(cmd)
    assert "--viewer" not in cmd and "tour_3.png" in " ".join(cmd)
    assert cwd == str(VISION) and "tour complete" in capsys.readouterr().out


def test_tour_default_adds_viewer_to_every_scenario(monkeypatch):
    calls = []
    monkeypatch.setattr(sys, "argv", ["demo_tour.py"])
    monkeypatch.setattr(tour, "banner", lambda *_a: None)
    monkeypatch.setattr(tour.subprocess, "run", lambda cmd, cwd: calls.append(cmd))
    tour.main()
    assert len(calls) == len(tour.SCENARIOS)
    assert all("--viewer" in cmd for cmd in calls)


def test_demo_tour_help_entrypoint():
    result = subprocess.run([sys.executable, str(VISION / "demo_tour.py"), "--help"],
                            cwd=VISION, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0 and "Watch every navigation capability" in result.stdout
