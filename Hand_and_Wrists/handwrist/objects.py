"""Household objects: how each one should be held, and where it is.

`ObjectSpec` is the per-class grasp knowledge a person would have ("pick a
remote up across its narrow side", "grab a bottle from the side"). It is
deliberately tiny -- dimensions are NOT duplicated here, they are read from
whatever produced the estimate.

`ObjectEstimate` is what the planner consumes: where the part to grip is, how
big it is, and which way it points. `truth_estimate` fills one in from the
simulator's ground truth; the perception stack will fill in the same fields
from the cameras, so the grasp code never knows the difference.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ObjectSpec:
    name: str
    grasp_geom: str               # the geom the fingers close on
    grasp: str = "top"            # "top": from above, "side": horizontally
    aligned: bool = False         # wrist yaw must follow the object's long axis
    handle_geom: str | None = None  # keep the fingers clear of this part
    grip_at_center: bool = False  # close at mid-height (round things)
    squeeze: float = 0.02         # m the fingers are commanded past the width


CATALOGUE = {
    "mug": ObjectSpec("mug", "mug_body", handle_geom="mug_handle"),
    "can": ObjectSpec("can", "can_body"),
    "bottle": ObjectSpec("bottle", "bottle_body", grasp="side"),
    "remote": ObjectSpec("remote", "remote_body", aligned=True),
    "keys": ObjectSpec("keys", "keys_body", aligned=True),
    "ball": ObjectSpec("ball", "ball_body", grip_at_center=True, squeeze=0.015),
    "box": ObjectSpec("box", "box_body", aligned=True),
}


@dataclass
class ObjectEstimate:
    name: str
    center: np.ndarray        # world xyz centre of the part to grip
    width: float              # gripping width across the fingers (m)
    height: float             # vertical extent (m)
    length: float             # extent along axis_yaw (m); == width if round
    axis_yaw: float | None    # world yaw the gripper's finger-free axis must
                              # follow; None if any wrist yaw will do
    body_id: int = -1         # sim body, -1 if unknown (perception)
    source: str = "truth"

    @property
    def top_z(self):
        return float(self.center[2] + self.height / 2)

    @property
    def bottom_z(self):
        return float(self.center[2] - self.height / 2)


def _yaw_of(v):
    return float(np.arctan2(v[1], v[0]))


def truth_estimate(model, data, spec: ObjectSpec) -> ObjectEstimate:
    """Ground-truth estimate straight from the simulator."""
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.grasp_geom)
    if gid < 0:
        raise KeyError(f"scene has no geom {spec.grasp_geom!r} for {spec.name!r}")
    R = data.geom_xmat[gid].reshape(3, 3)
    size = model.geom_size[gid]
    gtype = model.geom_type[gid]
    center = data.geom_xpos[gid].copy()

    if gtype == mujoco.mjtGeom.mjGEOM_BOX:
        # long axis is whichever horizontal box axis is bigger
        long_ax = 0 if size[0] >= size[1] else 1
        length, width = 2 * size[long_ax], 2 * size[1 - long_ax]
        height = 2 * size[2]
        axis_yaw = _yaw_of(R[:, long_ax]) if spec.aligned else None
    elif gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
        width = length = 2 * size[0]
        height = 2 * size[1]
        axis_yaw = None
    elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
        width = length = height = 2 * size[0]
        axis_yaw = None
    else:
        raise ValueError(f"unsupported grasp geom type {gtype} on {spec.name}")

    if spec.handle_geom is not None:
        hid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.handle_geom)
        # the handle sticks out along the finger-free axis, between the pads'
        # outer edges rather than under one of them
        axis_yaw = _yaw_of(data.geom_xpos[hid] - center)

    return ObjectEstimate(name=spec.name, center=center, width=float(width),
                          height=float(height), length=float(length),
                          axis_yaw=axis_yaw, body_id=int(model.geom_bodyid[gid]))


def set_object_pose(model, data, name, pos, yaw=0.0):
    """Teleport a free object so its body origin (its base) sits at `pos`."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_free")
    if jid < 0:
        raise KeyError(f"no free joint {name}_free")
    a = model.jnt_qposadr[jid]
    v = model.jnt_dofadr[jid]
    data.qpos[a:a + 3] = pos
    data.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    data.qvel[v:v + 6] = 0.0
