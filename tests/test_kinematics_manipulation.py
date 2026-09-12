from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from bracketbot_sim.kinematics import ARM_JOINTS, GRASP_SITE, GRIPPER_JOINTS, ArmIK, rot_z
from bracketbot_sim.manipulation import (GRIP_GAP_A, GRIP_GAP_B, ArmController,
                                         PickCube, StationKeeper, grip_for_gap)
from bracketbot_sim.robot import BracketBot


@pytest.mark.parametrize(("side", "prefix"), [("right", "rj"), ("left", "lj")])
def test_arm_constant_contract(side, prefix):
    assert ARM_JOINTS[side] == [f"{prefix}{i}" for i in range(7)]
    assert len(GRIPPER_JOINTS[side]) == 2
    assert GRASP_SITE[side] == f"{side}_grasp"


@pytest.mark.integration
@pytest.mark.parametrize("side", ["right", "left"])
def test_arm_ik_initialises_model_addresses_and_limits(robot, side):
    ik = ArmIK(robot.model, side)
    assert len(ik.joints) == len(ik.jids) == len(ik.qadr) == len(ik.dofs) == 7
    assert np.all(np.asarray(ik.jids) >= 0) and ik.site >= 0
    assert np.all(ik.lo <= ik.hi)
    assert np.array_equal(ik.rest, np.zeros(7))


@pytest.mark.integration
def test_arm_ik_site_pose_and_forward_match_live_pose(robot):
    robot.reset()
    ik = ArmIK(robot.model, "right")
    q = robot.data.qpos[ik.qadr].copy()
    p_live, R_live = ik.site_pose(robot.data)
    p_fwd, R_fwd = ik.forward(q, robot.data)
    assert p_live.shape == (3,) and R_live.shape == (3, 3)
    assert p_fwd == pytest.approx(p_live, abs=1e-8)
    assert R_fwd == pytest.approx(R_live, abs=1e-8)


@pytest.mark.integration
def test_arm_ik_solves_current_position_immediately(robot):
    robot.reset()
    ik = ArmIK(robot.model, "right")
    target, target_R = ik.site_pose(robot.data)
    source_before = robot.data.qpos.copy()
    q, pe, re = ik.solve(robot.data, target, target_R)
    assert pe < 1.5e-3 and re < .02
    assert q == pytest.approx(robot.data.qpos[ik.qadr])
    assert robot.data.qpos == pytest.approx(source_before)


@pytest.mark.integration
def test_arm_ik_position_only_and_zero_iterations(robot):
    robot.reset()
    ik = ArmIK(robot.model, "left", posture_weight=0)
    target, _ = ik.site_pose(robot.data)
    q, pe, re = ik.solve(robot.data, target, target_mat=None, q_init=np.zeros(7))
    assert q.shape == (7,) and re == 0 and pe < .01
    q0, pe0, re0 = ik.solve(robot.data, target, iters=0)
    assert q0.shape == (7,) and np.isinf(pe0) and np.isinf(re0)


@pytest.mark.integration
def test_arm_ik_respects_joint_limits_for_unreachable_target(robot):
    ik = ArmIK(robot.model, "right")
    q, pe, _ = ik.solve(robot.data, [100, 100, 100], iters=10, step_limit=.05)
    assert np.all(q >= ik.lo) and np.all(q <= ik.hi)
    assert pe > 1


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(0, 0), (GRIP_GAP_A, 0), (GRIP_GAP_A + GRIP_GAP_B / 2, .5),
     (GRIP_GAP_A + GRIP_GAP_B, 1), (99, 1)],
)
def test_grip_for_gap_clamps_and_converts(gap, expected):
    assert grip_for_gap(gap) == pytest.approx(expected)


def test_station_keeper_projects_position_on_heading_and_clips():
    commands = []
    bot = SimpleNamespace(position=np.array([0., 0., 0.]), yaw=0.,
                          drive=lambda v, w: commands.append((v, w)))
    keeper = StationKeeper((10, 5), np.pi, kp=2, kyaw=2, v_limit=.3, w_limit=.4)
    errors = keeper(bot)
    assert commands[-1] == pytest.approx((.3, .4))
    assert errors == pytest.approx((10, np.pi))
    bot.yaw = np.pi / 2
    forward, _ = keeper(bot)
    assert forward == pytest.approx(5)


@pytest.mark.integration
def test_arm_controller_initial_state_gap_and_pose(robot):
    robot.reset()
    arm = ArmController(robot, "right")
    assert arm.q_cmd.shape == arm.speeds.shape == (7,)
    assert arm.speeds[0] < arm.speeds[1]
    assert arm.grasp_pose[0].shape == (3,)
    assert arm.gap() == pytest.approx(GRIP_GAP_A + GRIP_GAP_B * arm.grip_cmd)


@pytest.mark.integration
def test_arm_controller_gripper_commands_apply_to_both_fingers(robot):
    robot.reset()
    arm = ArmController(robot, "right")
    arm.set_gripper(2)
    assert arm.grip_cmd == 1
    arm.apply()
    for joint in arm.grippers:
        assert robot.data.ctrl[robot._act_id(f"act_{joint}")] == pytest.approx(1)
    arm.open(.08)
    assert arm.grip_cmd == pytest.approx(grip_for_gap(.08))
    arm.close(.03)
    assert arm.grip_cmd == pytest.approx(grip_for_gap(.03))


@pytest.mark.integration
def test_arm_controller_track_rate_limits_and_records_errors(robot, monkeypatch):
    robot.reset()
    arm = ArmController(robot, "right", joint_speed=.5, mast_speed=.2)
    goal = arm.q_cmd + 10
    monkeypatch.setattr(arm.ik, "solve", lambda *_a, **_k: (goal, .123, .456))
    before = arm.q_cmd.copy()
    assert arm.track([0, 0, 0], np.eye(3), .1) == .123
    assert arm.q_cmd - before == pytest.approx([.02] + [.05] * 6)
    assert arm.last_pos_err == .123 and arm.last_rot_err == .456
    arm.hold()


@pytest.mark.integration
def test_arm_controller_at_uses_world_distance(robot, monkeypatch):
    arm = ArmController(robot, "left")
    monkeypatch.setattr(type(arm), "grasp_pose", property(lambda self: (np.array([1, 2, 3]), np.eye(3))))
    assert arm.at([1.005, 2, 3], tol=.01)
    assert not arm.at([1.02, 2, 3], tol=.01)


@pytest.fixture
def table_bot(table_model_path):
    bot = BracketBot(table_model_path)
    yield bot
    bot.close()


@pytest.mark.integration
def test_pick_cube_initial_geometry_and_standoff(table_bot):
    picker = PickCube(table_bot, verbose=False)
    base, yaw = picker.standoff_pose()
    assert base.shape == (2,) and np.isfinite(base).all() and np.isfinite(yaw)
    assert picker.phase == "navigate" and not picker.done
    assert picker.cube_start_z == pytest.approx(table_bot.body_position("cube")[2])
    assert picker.required_standoff > 0 and picker.standoff <= picker.standoff_used <= picker.max_reach


@pytest.mark.integration
def test_pick_cube_geometry_helpers(table_bot):
    picker = PickCube(table_bot, verbose=False)
    assert picker._table_half_extent([1, 0]) > 0
    assert picker._table_half_extent([0, 1]) > 0
    assert np.isfinite(picker.grasp_lateral_offset())
    target = picker.cube_target(.2)
    assert target[:2] == pytest.approx(table_bot.body_position("cube")[:2])
    assert target[2] == pytest.approx(table_bot.body_position("cube")[2] + .2)


def test_pick_table_extent_returns_zero_without_geom():
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    picker = object.__new__(PickCube)
    picker.bot = SimpleNamespace(model=model)
    assert picker._table_half_extent([1, 0]) == 0


def test_pick_enter_elapsed_done_and_verbose(capsys):
    picker = object.__new__(PickCube)
    picker.phase = "navigate"
    picker._phase_t0 = None
    picker.verbose = True
    picker._enter("align", 3.5)
    assert picker.phase == "align" and picker._elapsed(5) == pytest.approx(1.5)
    assert "align" in capsys.readouterr().out
    picker._enter("done", 6)
    assert picker.done


def test_pick_settled_and_grasped_predicates():
    picker = object.__new__(PickCube)
    picker.hold_xy = np.array([1., 2.])
    picker.bot = SimpleNamespace(position=np.array([1.01, 2., 0]), ground_speed=.01,
                                 body_position=lambda _n: np.array([1., 2., .6]))
    picker.arm = SimpleNamespace(grasp_pose=(np.array([1., 2., .61]), np.eye(3)))
    picker.cube_body = "cube"
    picker.cube_start_z = .5
    assert picker.settled() and picker.grasped()
    picker.bot.ground_speed = 1
    assert not picker.settled()
    picker.cube_start_z = .59
    assert not picker.grasped()


class FakeBalance:
    def __init__(self):
        self.max_lead = .3
        self.station = []
        self.reset = []

    def station_gains(self, on=True): self.station.append(on)
    def reset_reference(self, state): self.reset.append(state)


class FakeArm:
    def __init__(self):
        self.calls = []
        self.grasp_pose = (np.array([0., 0., .5]), np.eye(3))

    def hold(self): self.calls.append("hold")
    def open(self, gap): self.calls.append(("open", gap))
    def close(self, gap): self.calls.append(("close", gap))
    def track(self, p, m, dt): self.calls.append(("track", np.asarray(p), dt)); return 0
    def at(self, *_a, **_k): return True


def fake_picker(phase):
    p = object.__new__(PickCube)
    p.phase = phase; p._phase_t0 = None; p.verbose = False
    p.arm = FakeArm(); p.nav = SimpleNamespace(done=False)
    p.bot = SimpleNamespace(dt=.01, balance=FakeBalance(), position=np.array([0., 0., 0]),
                            ground_speed=0., state=np.zeros(6), yaw=0.,
                            drive=lambda *_a: None,
                            body_position=lambda _n: np.array([0., 0., .5]))
    p.manip_max_lead=.15; p._saved_max_lead=.3; p.hold_xy=np.zeros(2)
    p.settle_time=1.; p.open_gap=.08; p.close_gap=.03; p.grasp_yaw=0.
    p.approach_height=.14; p.grasp_offset=.005; p.lift_height=.22
    p.cube_body="cube"; p.cube_start_z=.5
    return p


def test_pick_navigate_transition_configures_station_controller():
    p = fake_picker("navigate")
    p.nav = lambda bot, t: None
    p.nav.done = True
    PickCube.__call__(p, p.bot, 2)
    assert p.phase == "align"
    assert p.bot.balance.max_lead == p.manip_max_lead
    assert p.bot.balance.station == [True] and len(p.bot.balance.reset) == 1
    assert p.arm.calls == ["hold"]


def test_pick_align_waits_then_enters_pregrasp():
    p = fake_picker("align")
    p.settled = lambda *_a: True
    PickCube.__call__(p, p.bot, 0)
    assert p.phase == "align"
    PickCube.__call__(p, p.bot, 2)
    assert p.phase == "pregrasp" and p.grasp_yaw == 0


@pytest.mark.parametrize(("phase", "next_phase"),
                         [("pregrasp", "descend"), ("descend", "grasp")])
def test_pick_reach_phases_advance_when_at_target_and_settled(phase, next_phase):
    p = fake_picker(phase)
    p.settled = lambda *_a: True
    PickCube.__call__(p, p.bot, 1)
    assert p.phase == next_phase
    assert p.arm.calls[0][0] == "track"


def test_pick_grasp_closes_then_enters_lift_after_delay():
    p = fake_picker("grasp")
    PickCube.__call__(p, p.bot, 0)
    assert p.phase == "grasp" and p.arm.calls[0][0] == "close"
    PickCube.__call__(p, p.bot, 2)
    assert p.phase == "lift"


def test_pick_lift_finishes_and_restores_balance_settings():
    p = fake_picker("lift")
    p.grasped = lambda: True
    PickCube.__call__(p, p.bot, 0)
    assert p.phase == "done"
    assert p.bot.balance.max_lead == p._saved_max_lead
    assert p.bot.balance.station == [False]


def test_pick_done_holds_arm():
    p = fake_picker("done")
    PickCube.__call__(p, p.bot, 0)
    assert p.arm.calls == ["hold"]
