from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bracketbot_sim import robot as robot_module
from bracketbot_sim.lqr import PlantParams
from bracketbot_sim.robot import BalanceController, BracketBot, ODriveSim


@pytest.fixture
def controller(monkeypatch):
    gains = [np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])[:2],
             np.full((2, 6), 2.0)]
    monkeypatch.setattr(robot_module, "LQR_gains", lambda *_a, **_k: gains.pop(0))
    return BalanceController(PlantParams(trim=.02))


def test_balance_initial_state_and_gain_switch(controller):
    assert not controller.enabled
    assert controller.K is controller.K_drive
    controller.station_gains()
    assert controller.K is controller.K_station
    controller.station_gains(False)
    assert controller.K is controller.K_drive


def test_balance_disabled_returns_zero_without_moving_references(controller):
    controller.v_cmd, controller.w_cmd = 2, 3
    assert controller(np.zeros(6), .5) == (0, 0)
    assert controller.x_ref == 0 and controller.yaw_ref == 0


def test_balance_enable_seeds_and_disable(controller):
    state = np.array([1, 2, 3, 4, .5, 6], float)
    controller.enable(state)
    assert controller.enabled and controller.x_ref == 1 and controller.yaw_ref == .5
    controller.disable()
    assert not controller.enabled


def test_balance_enable_without_state_preserves_references(controller):
    controller.x_ref, controller.yaw_ref = 4, 5
    controller.enable()
    assert (controller.x_ref, controller.yaw_ref) == (4, 5)


def test_balance_reset_reference(controller):
    controller.reset_reference([3, 0, 0, 0, -2, 0])
    assert controller.x_ref == 3 and controller.yaw_ref == -2


def test_balance_reference_governors_limit_lead(controller):
    controller.enable(np.zeros(6))
    controller.v_cmd, controller.w_cmd = 100, 100
    controller(np.zeros(6), 1)
    assert controller.x_ref == pytest.approx(controller.max_lead)
    assert abs(controller.yaw_ref) == pytest.approx(controller.max_yaw_lead)


def test_balance_yaw_governor_handles_wrap(controller):
    state = np.array([0, 0, 0, 0, np.pi - .1, 0])
    controller.enable(state)
    controller.w_cmd = 1
    controller(state, 1)
    assert abs(robot_module._wrap(controller.yaw_ref - state[4])) <= controller.max_yaw_lead + 1e-9


def test_balance_feedforward_uses_live_com(controller):
    controller.enable(np.zeros(6))
    bot = SimpleNamespace(com_lean=.1)
    controller(np.zeros(6), .01, bot)
    assert controller._base_trim == pytest.approx(-np.arctan2(.1, controller.plant.L))
    controller.trim_feedforward = False
    controller(np.zeros(6), .01, bot)
    assert controller._base_trim == controller.plant.trim


def test_balance_auto_trim_integrates_and_clamps(controller):
    controller.enable(np.zeros(6))
    controller.x_ref = 100
    for _ in range(10):
        controller(np.zeros(6), 100)
    assert abs(controller.trim_integral) == controller.trim_limit
    assert controller.pitch_trim == pytest.approx(controller._base_trim + controller.trim_integral)


def test_balance_auto_trim_can_be_disabled(controller):
    controller.enable(np.zeros(6))
    controller.auto_trim = False
    controller.x_ref = .2
    controller(np.zeros(6), 1)
    assert controller.trim_integral == 0


def test_balance_output_is_float_and_torque_limited(controller):
    controller.enable(np.zeros(6))
    controller.K[:] = 1e6
    pitch, yaw = controller(np.ones(6) * 10, 0)
    assert isinstance(pitch, float) and isinstance(yaw, float)
    assert abs(pitch) <= controller.max_torque
    assert abs(yaw) <= controller.max_yaw_torque


def test_robot_wrap_vectorized_and_scalar():
    assert robot_module._wrap(3 * np.pi) == pytest.approx(-np.pi)
    assert robot_module._wrap(np.array([0, 2 * np.pi])) == pytest.approx([0, 0])


@pytest.mark.integration
def test_robot_core_contract_and_state_shapes(robot):
    robot.reset()
    assert robot.dt == pytest.approx(robot.model.opt.timestep)
    assert robot.time == 0
    assert robot.rotation.shape == (3, 3)
    assert robot.gravity_body.shape == (3,)
    assert robot.angular_velocity.shape == (3,)
    assert robot.wheel_angles.shape == robot.wheel_rates.shape == (2,)
    assert robot.wheel_speeds_mps.shape == (2,)
    assert robot.position.shape == (3,) and robot.state.shape == (6,)
    assert np.isfinite(robot.state).all()
    assert isinstance(robot.pitch, float) and isinstance(robot.roll, float)
    assert isinstance(robot.yaw, float) and isinstance(robot.com_lean, float)
    assert isinstance(robot.ground_speed, float) and not robot.fallen


@pytest.mark.integration
def test_robot_joint_and_camera_name_inventory(robot):
    assert "rj0" in robot.arm_joints and "lj0" in robot.arm_joints
    assert "left_wheel_joint" not in robot.arm_joints
    assert set(robot.camera_names) == {"head_rgb", "head_depth", "head_stereo_left",
                                       "head_stereo_right", "wrist_cam_right",
                                       "wrist_cam_left", "chase"}
    assert robot._act_id("left_wheel") >= 0 and robot._jnt_id("rj0") >= 0


@pytest.mark.integration
def test_robot_reset_applies_arm_pose_and_reanchors_odometry(robot):
    robot.reset({"rj0": .02, "lj0": .03})
    assert robot.joint_position("rj0") == pytest.approx(.02)
    assert robot.joint_position("lj0") == pytest.approx(.03)
    assert robot.data.ctrl[robot._act_id("act_rj0")] == pytest.approx(.02)
    assert robot.odometry == pytest.approx(0)


@pytest.mark.integration
def test_robot_reset_reseeds_existing_balance_reference(robot):
    robot.balance.x_ref = 99
    robot.balance.yaw_ref = 99
    robot.reset()
    assert robot.balance.x_ref == pytest.approx(robot.state[0])
    assert robot.balance.yaw_ref == pytest.approx(robot.state[4])


@pytest.mark.integration
def test_robot_drive_and_raw_actuator_commands(robot):
    robot.reset()
    robot.drive(.3, -.4)
    assert robot.balance.v_cmd == .3 and robot.balance.w_cmd == -.4
    robot.set_wheel_torque(999, -999)
    assert robot.data.ctrl[robot._wheel_act] == pytest.approx([15, -15])
    robot.set_arm_target("rj1", .25)
    assert robot.data.ctrl[robot._act_id("act_rj1")] == pytest.approx(.25)
    robot.set_arm_pose({"rj2": .1, "lj2": -.1})
    assert robot.data.ctrl[robot._act_id("act_rj2")] == pytest.approx(.1)
    assert robot.data.ctrl[robot._act_id("act_lj2")] == pytest.approx(-.1)


@pytest.mark.integration
def test_robot_joint_body_site_queries(robot):
    robot.reset()
    assert isinstance(robot.joint_position("rj1"), float)
    assert isinstance(robot.joint_velocity("rj1"), float)
    assert robot.body_position("chassis").shape == (3,)
    assert robot.site_position("right_grasp").shape == (3,)
    with pytest.raises(KeyError, match="no body"):
        robot.body_position("does_not_exist")


@pytest.mark.integration
def test_robot_step_default_duration_and_controller_count(robot):
    robot.reset()
    calls = []
    state = robot.step(controller=lambda bot, t: calls.append((bot, t)))
    assert len(calls) == 1 and robot.time == pytest.approx(robot.dt)
    assert state.shape == (6,)
    robot.reset()
    robot.step(robot.dt * 5.2, controller=lambda *_: calls.append(1))
    assert len(calls) == 1 + 5


@pytest.mark.integration
def test_robot_step_always_advances_at_least_one_step(robot):
    robot.reset()
    robot.step(0)
    assert robot.time == pytest.approx(robot.dt)


@pytest.mark.integration
def test_robot_pose_angles_are_yaw_invariant(robot):
    robot.reset()
    robot.data.qpos[3:7] = [np.cos(.6), 0, 0, np.sin(.6)]
    mujoco.mj_forward(robot.model, robot.data)
    assert robot.yaw == pytest.approx(1.2, abs=1e-6)
    assert robot.pitch == pytest.approx(0, abs=1e-6)
    assert robot.roll == pytest.approx(0, abs=1e-6)


def test_point_cloud_geometry_with_mock_depth(robot, monkeypatch):
    depth = np.array([[np.inf, 2.0], [1.0, 2.0]], np.float32)
    monkeypatch.setattr(robot, "depth", lambda *_a, **_k: depth)
    pts = robot.point_cloud("head_depth", width=2, height=2)
    assert pts.shape == (3, 3)
    assert np.array_equal(pts[:, 2], [-2, -1, -2])
    assert np.isfinite(pts).all()


def test_point_cloud_world_applies_camera_pose(robot, monkeypatch):
    monkeypatch.setattr(robot, "point_cloud", lambda *_a, **_k: np.array([[1., 0, 0]]))
    monkeypatch.setattr(robot, "camera_pose", lambda _n: (np.array([2., 3, 4]),
                                                           np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])))
    assert np.allclose(robot.point_cloud_world(), [[2, 4, 4]])


def test_renderer_cache_depth_mode_and_close(monkeypatch, robot):
    made = []

    class Renderer:
        def __init__(self, model, height, width):
            self.args = model, height, width
            self.depth = False
            self.closed = False
            made.append(self)

        def enable_depth_rendering(self): self.depth = True
        def close(self): self.closed = True

    monkeypatch.setattr(robot_module.mujoco, "Renderer", Renderer)
    cache = {}
    a = robot._renderer(cache, 10, 20, False)
    b = robot._renderer(cache, 10, 20, True)
    c = robot._renderer(cache, 11, 20, True)
    assert a is b and c is not a
    assert not a.depth and c.depth
    robot._renderers, robot._depth_renderers = cache, {}
    robot.close()
    assert all(r.closed for r in made) and not cache
    robot.close()


def test_context_manager_closes(monkeypatch, model_path):
    bot = BracketBot(model_path)
    calls = []
    monkeypatch.setattr(bot, "close", lambda: calls.append(1))
    with bot as entered:
        assert entered is bot
    assert calls == [1]


class ODriveBot:
    def __init__(self):
        self.wheel_angles = np.array([2 * np.pi, -np.pi])
        self.wheel_speeds_mps = np.array([.1, -.2])
        self.torques = []

    def set_wheel_torque(self, left, right):
        self.torques.append((float(left), float(right)))


def test_odrive_lifecycle_commands_and_measurements():
    bot = ODriveBot()
    odrive = ODriveSim(bot)
    odrive.start_left(); odrive.start_right()
    assert odrive.active
    odrive.enable_velocity_mode_left(); odrive.enable_velocity_mode_right()
    odrive.disable_watchdog_left(); odrive.disable_watchdog_right()
    odrive.set_speed_mps_left(.5); odrive.set_speed_mps_right(-.5)
    assert odrive.target == pytest.approx([.5, -.5])
    assert odrive.get_position_turns_left() == pytest.approx(1)
    assert odrive.get_position_turns_right() == pytest.approx(-.5)
    assert odrive.get_speed_mps_left() == pytest.approx(.1)
    assert odrive.get_speed_mps_right() == pytest.approx(-.2)


def test_odrive_pi_update_integrates_and_clips():
    bot = ODriveBot()
    odrive = ODriveSim(bot, kp=10, ki=20, torque_limit=2)
    odrive.target[:] = [10, -10]
    odrive.update(1)
    assert odrive._integral == pytest.approx([2, -2])
    assert bot.torques[-1] == pytest.approx((2, -2))
    odrive.clear_errors_left()
    assert np.array_equal(odrive._integral, [0, 0])
    odrive._integral[:] = 1
    odrive.clear_errors_right()
    assert np.array_equal(odrive._integral, [0, 0])
