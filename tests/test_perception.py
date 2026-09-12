from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from vision_sim import perception


def make_obs(rgb, depth=None, points=None, robot_xy=(0.0, 0.0), cam_pos=(0, 0, 1)):
    rgb = np.asarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    depth = np.ones((h, w), np.float32) if depth is None else np.asarray(depth, np.float32)
    if points is None:
        points = np.zeros((h, w, 3), float)
        points[..., 0] = 2.0
        points[..., 2] = 0.5
    return perception.Observation(
        rgb=rgb, depth=depth, points=np.asarray(points, float),
        valid=np.isfinite(depth), cam_pos=np.asarray(cam_pos, float),
        cam_mat=np.eye(3),
        intrinsics=perception.Intrinsics(1, 1, w / 2, h / 2, w, h),
        robot_xy=np.asarray(robot_xy, float),
    )


def test_intrinsics_from_vertical_fov():
    model = SimpleNamespace(cam_fovy=np.array([90.0]))
    intr = perception.Intrinsics.from_model(model, 0, width=200, height=100)
    assert intr.fx == pytest.approx(50.0)
    assert intr.fy == pytest.approx(50.0)
    assert (intr.cx, intr.cy, intr.width, intr.height) == (100, 50, 200, 100)


@pytest.mark.parametrize(
    ("rgb", "hsv"),
    [([255, 0, 0], [0, 1, 1]), ([0, 255, 0], [120, 1, 1]),
     ([0, 0, 255], [240, 1, 1]), ([255, 255, 0], [60, 1, 1]),
     ([255, 255, 255], [0, 0, 1]), ([0, 0, 0], [0, 0, 0]),
     ([128, 128, 128], [0, 0, 128 / 255])],
)
def test_rgb_to_hsv_known_colours(rgb, hsv):
    actual = perception.rgb_to_hsv(np.array([[rgb]], np.uint8))[0, 0]
    assert actual == pytest.approx(hsv, abs=1e-5)


def test_rgb_to_hsv_preserves_image_shape_and_float_type():
    out = perception.rgb_to_hsv(np.zeros((3, 4, 3), np.uint8))
    assert out.shape == (3, 4, 3)
    assert np.issubdtype(out.dtype, np.floating)


def test_object_class_regular_hue_window_and_thresholds():
    cls = perception.ObjectClass("green", (100, 140), sat_min=0.5, val_min=0.2)
    hsv = np.array([[[120, 0.7, 0.8], [90, 0.7, 0.8],
                     [120, 0.4, 0.8], [120, 0.7, 0.1]]])
    assert np.array_equal(cls.mask(hsv), [[True, False, False, False]])


def test_object_class_wrapped_hue_window():
    cls = perception.ObjectClass("red", (345, 20))
    hsv = np.array([[[350, 1, 1], [10, 1, 1], [180, 1, 1]]])
    assert np.array_equal(cls.mask(hsv), [[True, True, False]])


def test_deproject_centre_and_corners():
    intr = perception.Intrinsics(2, 2, 1, 1, 3, 3)
    depth = np.full((3, 3), 2.0)
    pts = perception.deproject(depth, intr)
    assert pts[1, 1] == pytest.approx([0, 0, -2])
    assert pts[0, 0] == pytest.approx([-1, 1, -2])


def test_deproject_nonfinite_depth_becomes_all_nan():
    intr = perception.Intrinsics(1, 1, 0, 0, 2, 1)
    pts = perception.deproject(np.array([[np.inf, np.nan]]), intr)
    assert np.isnan(pts).all()


def test_observe_rejects_missing_camera():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    bot = SimpleNamespace(model=model)
    with pytest.raises(ValueError, match="no camera"):
        perception.observe(bot, camera="missing")


def test_observe_renders_registered_frame_and_transforms_world_points():
    xml = """<mujoco><worldbody><camera name='cam' pos='1 2 3' fovy='90'/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    class Bot:
        position = np.array([4.0, 5.0, 0.0])
        camera_calls = []
        depth_calls = []

        def camera(self, name, width, height):
            self.camera_calls.append((name, width, height))
            return np.zeros((height, width, 3), np.uint8)

        def depth(self, name, width, height, max_range):
            self.depth_calls.append((name, width, height, max_range))
            return np.ones((height, width), np.float32)

    bot = Bot()
    bot.model, bot.data = model, data
    obs = perception.observe(bot, "cam", width=2, height=2, max_range=7)
    assert obs.rgb.shape == (2, 2, 3) and obs.points.shape == (2, 2, 3)
    assert obs.valid.all()
    assert np.array_equal(obs.robot_xy, [4, 5])
    assert bot.camera_calls == [("cam", 2, 2)]
    assert bot.depth_calls == [("cam", 2, 2, 7)]
    expected = perception.deproject(obs.depth, obs.intrinsics) @ obs.cam_mat.T + obs.cam_pos
    assert obs.points == pytest.approx(expected)


def test_cluster_empty():
    assert perception._cluster_xy(np.empty((0, 2)), 0.2).shape == (0,)


def test_cluster_connects_neighbouring_cells_and_separates_gaps():
    xy = np.array([[0.01, 0.01], [0.11, 0.11], [0.21, 0.21], [2.0, 2.0]])
    labels = perception._cluster_xy(xy, 0.1)
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] != labels[0]


def test_detection_repr_contains_label_pose_range_and_bearing():
    d = perception.Detection("target", np.array([1, 2, 3]), 4.25,
                             np.pi / 2, np.ones(3), 50)
    text = repr(d)
    assert "target" in text and "(1.00, 2.00, 3.00)" in text
    assert "4.25m" in text and "+90deg" in text and "50px" in text


def test_detect_empty_when_class_pixel_threshold_not_met():
    obs = make_obs(np.full((2, 2, 3), [255, 0, 0], np.uint8))
    cls = perception.ObjectClass("red", (350, 10), min_pixels=5)
    assert perception.detect(obs, classes=[cls]) == []


def test_detect_builds_bbox_centroid_extent_distance_and_bearing():
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[1:3, 1:4] = [255, 0, 0]
    pts = np.zeros((4, 5, 3), float)
    pts[..., 0] = 2
    pts[..., 2] = 0.5
    pts[1:3, 1:4, 1] = np.linspace(1, 1.5, 6).reshape(2, 3)
    obs = make_obs(rgb, points=pts, cam_pos=(0, 0, 0))
    cls = perception.ObjectClass("red", (350, 10), min_pixels=3)
    dets = perception.detect(obs, classes=[cls], self_radius=0.1,
                             cluster_res=1.0, robot_yaw=0.2)
    assert len(dets) == 1
    d = dets[0]
    assert d.label == "red" and d.pixels == 6 and d.bbox == (1, 1, 3, 2)
    assert d.position == pytest.approx([2, 1.25, 0.5])
    assert d.extent == pytest.approx([0, 0.5, 0])
    assert d.distance == pytest.approx(np.linalg.norm(d.position))
    assert d.bearing == pytest.approx(np.arctan2(1.25, 2) - 0.2)


def test_detect_filters_height_range_validity_and_robot_self_returns():
    rgb = np.full((2, 3, 3), [255, 0, 0], np.uint8)
    pts = np.array([[[0.1, 0, 0.5], [2, 0, 0.05], [2, 0, 3]],
                    [[2, 0, 0.5], [2, 0, 0.5], [2, 0, 0.5]]], float)
    depth = np.array([[1, 1, 1], [0.2, np.inf, 1]], float)
    obs = make_obs(rgb, depth=depth, points=pts)
    cls = perception.ObjectClass("red", (350, 10), min_pixels=1)
    dets = perception.detect(obs, classes=[cls], self_radius=0.5,
                             min_range=0.4, min_height=0.1, max_height=2)
    assert len(dets) == 1 and dets[0].pixels == 1


def test_detect_sorts_nearest_first():
    rgb = np.zeros((1, 4, 3), np.uint8)
    rgb[0, :2] = [255, 0, 0]
    rgb[0, 2:] = [0, 255, 0]
    pts = np.array([[[1, 0, .5], [1, .01, .5], [3, 0, .5], [3, .01, .5]]])
    obs = make_obs(rgb, points=pts, cam_pos=(0, 0, 0))
    classes = [perception.ObjectClass("far", (110, 130), min_pixels=1),
               perception.ObjectClass("near", (350, 10), min_pixels=1)]
    dets = perception.detect(obs, classes=classes, self_radius=0.1)
    assert [d.label for d in dets] == ["near", "far"]


def test_obstacle_points_filters_geometry_and_self():
    pts = np.array([[[1, 0, .5], [.1, 0, .5], [2, 0, .05], [2, 0, 2.1]]])
    obs = make_obs(np.zeros((1, 4, 3), np.uint8),
                   depth=np.array([[1, 1, 1, 1]]), points=pts)
    out = perception.obstacle_points(obs, self_radius=0.5)
    assert out.shape == (1, 3) and out[0] == pytest.approx([1, 0, .5])


def test_floor_points_filters_height_and_range():
    pts = np.array([[[1, 0, .05], [2, 0, .2], [3, 0, .0], [4, 0, .0]]])
    obs = make_obs(np.zeros((1, 4, 3), np.uint8),
                   depth=np.array([[1, 1, .2, 9]]), points=pts)
    out = perception.floor_points(obs, max_height=.1, min_range=.4, max_range=8)
    assert out.shape == (1, 3) and out[0] == pytest.approx([1, 0, .05])
