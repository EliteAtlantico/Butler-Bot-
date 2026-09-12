"""Measure the balance-plant parameters from the compiled MuJoCo model.

The hard-coded numbers in BracketBot's `lib/lqr.py` describe the v1 hardware.
This model is v2 with two arms bolted on, so rather than trusting those
constants we read the equivalent quantities straight out of the simulated
bodies. The LQR is then derived from the plant it will actually control --
including whatever pose the arms are parked in, which moves the CoM.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .lqr import PlantParams

WHEEL_BODIES = ("left_wheel", "right_wheel")


def _body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _group_inertia(model, data, body_ids):
    """(mass, CoM, inertia about CoM) in world axes for a set of bodies."""
    m = np.array([model.body_mass[b] for b in body_ids])
    c = np.array([data.xipos[b] for b in body_ids])
    total = m.sum()
    com = (m[:, None] * c).sum(0) / total
    I = np.zeros((3, 3))
    for b, mi, ci in zip(body_ids, m, c):
        R = data.ximat[b].reshape(3, 3)
        Ib = R @ np.diag(model.body_inertia[b]) @ R.T
        r = ci - com
        I += Ib + mi * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    return total, com, I


def _subtree(model, root_id):
    """Body ids in the subtree rooted at `root_id`, inclusive."""
    out = {root_id}
    for b in range(root_id + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


def measure_plant(model, data) -> PlantParams:
    """Derive PlantParams from the model in its current configuration.

    Only the robot's own subtree is measured -- the scene may contain walls,
    pillars and other static bodies that are emphatically not part of the
    inverted pendulum.
    """
    mujoco.mj_forward(model, data)

    wheel_ids = [_body_id(model, n) for n in WHEEL_BODIES]
    robot_ids = _subtree(model, _body_id(model, "chassis"))
    sprung_ids = [b for b in sorted(robot_ids) if b not in wheel_ids]

    # wheel geometry: axle height and half-track come from the body frames
    axle_z = float(np.mean([data.xpos[b][2] for b in wheel_ids]))
    track = float(abs(data.xpos[wheel_ids[0]][1] - data.xpos[wheel_ids[1]][1]))

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "left_tire")
    radius = float(model.geom_size[gid][0])

    wheel_mass = float(model.body_mass[wheel_ids[0]])
    # the hinge is the wheel's y axis, so the spin inertia is the y component
    Jr = float(model.body_inertia[wheel_ids[0]][1])

    Mp, com, I = _group_inertia(model, data, sprung_ids)
    L = float(com[2] - axle_z)

    # The CoM is not exactly over the axle (the mast leans, the arms hang
    # forward), so "upright" is not pitch == 0. Balancing at pitch 0 would make
    # the robot creep to keep catching itself; this is the pitch that actually
    # puts the CoM over the contact patch.
    axle_x = float(np.mean([data.xpos[b][0] for b in wheel_ids]))
    # pitch > 0 leans forward, so a CoM ahead of the axle needs a lean back
    trim = float(-np.arctan2(com[0] - axle_x, L))

    return PlantParams(
        Jr=Jr, Mr=wheel_mass,
        Jpth=float(I[1, 1]),   # pitch: rotation about y
        Jpd=float(I[2, 2]),    # yaw:   rotation about z
        Mp=float(Mp), R=radius, D=track, L=L, trim=trim,
    )
