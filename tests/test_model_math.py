from __future__ import annotations

import importlib.util
from pathlib import Path

import mujoco
import numpy as np
import pytest

from bracketbot_sim import lqr, plant
from bracketbot_sim.kinematics import rot_z


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_mujoco"


def load_builder():
    spec = importlib.util.spec_from_file_location("build_dynamic_model", MAIN / "build_dynamic_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_builder()


def test_plant_params_defaults_and_description():
    p = lqr.PlantParams()
    assert p.Mr == 2.2 and p.R > 0 and p.D > 0 and p.L > 0
    desc = p.describe()
    for text in ("Mp=", "L=", "Jpth=", "Jpd=", "Mr=", "Jr=", "R=", "D=", "trim="):
        assert text in desc
    assert "deg" in desc


def test_state_space_shapes_structure_and_finiteness():
    A, B = lqr.state_space(lqr.PlantParams())
    assert A.shape == (6, 6) and B.shape == (6, 2)
    assert np.isfinite(A).all() and np.isfinite(B).all()
    assert A[0, 1] == 1 and A[2, 3] == 1 and A[4, 5] == 1
    assert np.count_nonzero(B[:, 0]) == 2
    assert np.count_nonzero(B[:, 1]) == 1


def test_state_space_changes_with_physical_parameters():
    A1, B1 = lqr.state_space(lqr.PlantParams())
    A2, B2 = lqr.state_space(lqr.PlantParams(Mp=5.0, L=.5, D=.7))
    assert not np.allclose(A1, A2)
    assert not np.allclose(B1, B2)


def test_lqr_gains_shape_finite_and_stabilize_controllable_dynamics():
    p = lqr.PlantParams()
    A, B = lqr.state_space(p)
    K = lqr.LQR_gains([60, 30, 260, 20, 40, 10], [18, 1], p)
    assert K.shape == (2, 6) and np.isfinite(K).all()
    eig = np.linalg.eigvals(A - B @ K)
    assert np.max(eig.real) < 1e-6


def test_lqr_default_plant_matches_explicit_default():
    q, r = [60, 30, 260, 20, 40, 10], [18, 1]
    assert lqr.LQR_gains(q, r) == pytest.approx(lqr.LQR_gains(q, r, lqr.PlantParams()))


@pytest.mark.parametrize("theta", [0, np.pi / 2, -np.pi / 2, np.pi])
def test_rot_z_is_proper_rotation(theta):
    R = rot_z(theta)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1)
    assert R @ np.array([0, 0, 1]) == pytest.approx([0, 0, 1])


def test_builder_fmt_scalar_vector_and_precision():
    assert builder.fmt(1.25) == "1.25"
    assert builder.fmt([1 / 3, 2], prec=3) == "0.333 2"


def test_mat2quat_identity_and_rotation_round_trip():
    for R in (np.eye(3), rot_z(.7)):
        q = builder.mat2quat(R)
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, q)
        assert out.reshape(3, 3) == pytest.approx(R, abs=1e-7)


def test_inertia_to_mjcf_diagonal_and_clips_nonpositive_values():
    q, diag = builder.inertia_to_mjcf(np.diag([3.0, 2.0, -1.0]))
    assert np.linalg.norm(q) == pytest.approx(1)
    assert np.all(diag > 0)
    assert diag[0] == pytest.approx(1e-8)


def test_camera_xyaxes_are_orthonormal_and_face_requested_direction():
    axes = builder.camera_xyaxes([1, 1, -1], [0, 0, 1])
    x, y = axes[:3], axes[3:]
    f = np.cross(y, x)  # camera looks down local -z
    target = np.array([1, 1, -1], float) / np.sqrt(3)
    assert np.linalg.norm(x) == pytest.approx(1)
    assert np.linalg.norm(y) == pytest.approx(1)
    assert x @ y == pytest.approx(0, abs=1e-12)
    assert f == pytest.approx(target)


def test_world_to_local_identity_and_rotated_body():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b' pos='1 2 3' euler='0 0 90'/></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "b")
    axes_w = np.array([1, 0, 0, 0, 1, 0], float)
    pos, axes = builder.world_to_local(data, bid, [1, 3, 3], axes_w)
    assert pos == pytest.approx([1, 0, 0], abs=1e-6)
    assert axes[:3] == pytest.approx([0, -1, 0], abs=1e-6)


def test_subtree_helpers_find_only_descendants():
    xml = """<mujoco><worldbody>
      <body name='root'><body name='child'><body name='grand'/></body></body>
      <body name='other'/></worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    ids = builder.subtree_bodies(model, "root")
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in ids}
    assert names == {"root", "child", "grand"}
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "root")
    assert plant._subtree(model, root) == ids


def test_group_inertia_and_builder_combine_inertia_parallel_axis():
    xml = """<mujoco><worldbody>
      <body name='a' pos='-1 0 0'><inertial pos='0 0 0' mass='2' diaginertia='1 1 1'/></body>
      <body name='b' pos='1 0 0'><inertial pos='0 0 0' mass='2' diaginertia='1 1 1'/></body>
    </worldbody></mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in ("a", "b")]
    mass, com, I = plant._group_inertia(model, data, ids)
    assert mass == pytest.approx(4) and com == pytest.approx([0, 0, 0])
    assert np.diag(I) == pytest.approx([2, 6, 6])
    mass2, com2, I2 = builder.combine_inertia(model, data, ids, .5)
    assert mass2 == pytest.approx(2) and com2 == pytest.approx(com)
    assert I2 == pytest.approx(I * .5)


@pytest.mark.integration
def test_measure_plant_matches_compiled_robot(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    p = plant.measure_plant(model, data)
    assert p.Mr == pytest.approx(2.2, rel=.01)
    assert p.Jr == pytest.approx(.018, rel=.01)
    assert p.R == pytest.approx(.0846, rel=.01)
    assert p.D == pytest.approx(.3222, rel=.01)
    assert p.Mp > 5 and p.L > .2
    assert abs(p.trim) < .2


@pytest.mark.integration
def test_sprung_body_helpers_exclude_wheels(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ids = plant.sprung_bodies(model)
    names = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(i)) for i in ids}
    assert "chassis" in names
    assert "left_wheel" not in names and "right_wheel" not in names
    com = plant.sprung_com(model, data, ids)
    assert com.shape == (3,) and np.isfinite(com).all() and com[2] > 0


@pytest.mark.integration
@pytest.mark.parametrize("filename", ["chopped_dynamic.xml", "scene.xml", "scene_dynamic.xml",
                                       "scene_flat.xml", "scene_table.xml"])
def test_all_mjcf_program_models_compile_and_step(filename):
    model = mujoco.MjModel.from_xml_path(str(MAIN / filename))
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert data.time == pytest.approx(model.opt.timestep)
    assert model.nbody > 1 and model.ngeom > 0


@pytest.mark.integration
def test_compiled_robot_contract(model_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    assert model.nq > 20 and model.nv > 20 and model.nu >= 20
    assert model.ncam == 7 and model.nsensor >= 8
    for name in ("chassis", "left_wheel", "right_wheel"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
    for name in ("left_wheel", "right_wheel", "act_rj0", "act_lj0"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) >= 0
    for name in ("head_rgb", "head_depth", "head_stereo_left", "head_stereo_right",
                 "wrist_cam_left", "wrist_cam_right", "chase"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0
    for name in ("right_grasp", "left_grasp", "imu"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0
