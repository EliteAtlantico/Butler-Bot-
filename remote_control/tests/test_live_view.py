"""The live view shows the room: a scene camera, and head views that look down at it."""

from __future__ import annotations

import copy
import math
import os
import unittest

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from remote_control.robot_adapter import DEFAULT_SCENE, USER_CAMERAS, display_camera  # noqa: E402


def view(camera_id: str) -> dict:
    return next(camera for camera in USER_CAMERAS if camera["id"] == camera_id)


def sky_fraction(rgb) -> float:
    """Share of pixels that are the blue skybox rather than the room."""
    rgb = np.asarray(rgb, dtype=np.int16)
    return float((rgb[..., 2] > rgb[..., 0] + 20).mean())


class LiveViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
        cls.data = mujoco.MjData(cls.model)
        mujoco.mj_forward(cls.model, cls.data)

    def camera_id(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)

    def test_every_view_uses_cameras_the_robot_model_has(self):
        for camera in USER_CAMERAS:
            for name in (camera["model_name"], camera.get("tilt_from")):
                if name:
                    self.assertGreaterEqual(self.camera_id(name), 0, name)
        self.assertEqual(USER_CAMERAS[0]["id"], "scene")

    def test_views_without_a_tilt_render_their_model_camera(self):
        self.assertEqual(display_camera(self.model, self.data, view("scene")), "chase")
        self.assertEqual(display_camera(self.model, self.data, view("wrist-left")), "wrist_cam_left")

    def test_head_views_look_down_like_the_depth_sensor_from_their_own_position(self):
        depth = -self.data.cam_xmat[self.camera_id("head_depth")].reshape(3, 3)[:, 2]
        for camera_id, model_name in (("head-left", "head_stereo_left"),
                                      ("head-right", "head_stereo_right")):
            shown = display_camera(self.model, self.data, view(camera_id))
            self.assertEqual(shown.type, mujoco.mjtCamera.mjCAMERA_FREE)
            azimuth, elevation = math.radians(shown.azimuth), math.radians(shown.elevation)
            forward = np.array([math.cos(elevation) * math.cos(azimuth),
                                math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
            np.testing.assert_allclose(forward, depth, atol=1e-6)
            self.assertLess(shown.elevation, -15.0)          # the depth sensor's ~22 deg down
            eye = np.asarray(shown.lookat) - shown.distance * forward
            np.testing.assert_allclose(eye, self.data.cam_xpos[self.camera_id(model_name)], atol=1e-6)

    def test_the_head_view_shows_the_table_in_front_instead_of_sky(self):
        display = copy.deepcopy(self.model)
        display.vis.global_.fovy = float(self.model.cam_fovy[self.camera_id("head_stereo_left")])
        with mujoco.Renderer(self.model, 240, 320) as level:
            level.update_scene(self.data, camera="head_stereo_left")
            before = sky_fraction(level.render())
        with mujoco.Renderer(display, 240, 320) as tilted:
            tilted.update_scene(self.data, camera=display_camera(self.model, self.data, view("head-left")))
            after = sky_fraction(tilted.render())
        self.assertGreater(before, 0.4)       # the level camera: mostly sky over an empty plane
        self.assertLess(after, 0.3)           # the shown view: the floor and the coffee table


if __name__ == "__main__":
    unittest.main()
