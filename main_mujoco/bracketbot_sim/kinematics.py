"""Inverse kinematics for the BracketBot arms.

Damped least squares on the grasp site, solved on a scratch `MjData` so it
never disturbs the running simulation. Seven joints per arm (the mast prismatic
`?j0` plus six revolute) against a 6-DOF target, so there is redundancy; a
posture bias keeps the solver from wandering into folded-up configurations.

The grasp sites are built world-axis-aligned at the home pose, which makes the
target orientation easy to think about: identity means "gripper pointing
straight down, fingers straddling along world y", and `Rz(theta)` spins the
grasp about the vertical.
"""
from __future__ import annotations

import mujoco
import numpy as np

ARM_JOINTS = {
    "right": ["rj0", "rj1", "rj2", "rj3", "rj4", "rj5", "rj6"],
    "left": ["lj0", "lj1", "lj2", "lj3", "lj4", "lj5", "lj6"],
}
GRIPPER_JOINTS = {
    "right": ["right_left_gripper", "right_right_gripper"],
    "left": ["left_left_gripper", "left_right_gripper"],
}
GRASP_SITE = {"right": "right_grasp", "left": "left_grasp"}


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ArmIK:
    def __init__(self, model, side="right", posture_weight=0.02):
        self.model = model
        self.side = side
        self.data = mujoco.MjData(model)
        self.joints = ARM_JOINTS[side]
        self.jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                     for j in self.joints]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dofs = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo = np.array([model.jnt_range[j][0] for j in self.jids])
        self.hi = np.array([model.jnt_range[j][1] for j in self.jids])
        self.site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      GRASP_SITE[side])
        self.posture_weight = posture_weight
        self.rest = np.zeros(len(self.joints))

    def site_pose(self, data):
        """(position, rotation matrix) of the grasp site in world coords."""
        return data.site_xpos[self.site].copy(), data.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, data_src, target_pos, target_mat=None, q_init=None,
              iters=200, pos_tol=1.5e-3, rot_tol=0.02, damping=0.12,
              rot_weight=0.35, step_limit=0.25):
        """Joint values that put the grasp site at (target_pos, target_mat).

        Seeded from the live state, so the solution stays near the arm's
        current configuration instead of teleporting through the robot.
        Returns (q, pos_err, rot_err); check the errors, it does not raise.
        """
        d = self.data
        d.qpos[:] = data_src.qpos
        d.qvel[:] = 0.0
        if q_init is not None:
            d.qpos[self.qadr] = q_init
        target_pos = np.asarray(target_pos, float)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        q_err = np.zeros(4)
        q_cur = np.zeros(4)
        q_des = np.zeros(4)
        vel = np.zeros(3)

        pos_err = rot_err = np.inf
        for _ in range(iters):
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            p, R = self.site_pose(d)

            e_pos = target_pos - p
            pos_err = float(np.linalg.norm(e_pos))
            if target_mat is None:
                err = e_pos
                rot_err = 0.0
            else:
                mujoco.mju_mat2Quat(q_cur, R.flatten())
                mujoco.mju_mat2Quat(q_des, np.asarray(target_mat, float).flatten())
                mujoco.mju_negQuat(q_err, q_cur)
                mujoco.mju_mulQuat(q_err, q_des, q_err)
                mujoco.mju_quat2Vel(vel, q_err, 1.0)
                rot_err = float(np.linalg.norm(vel))
                err = np.concatenate([e_pos, vel * rot_weight])

            if pos_err < pos_tol and rot_err < rot_tol:
                break

            mujoco.mj_jacSite(self.model, d, jacp, jacr, self.site)
            J = jacp[:, self.dofs] if target_mat is None else np.vstack(
                [jacp[:, self.dofs], jacr[:, self.dofs] * rot_weight])

            # damped least squares, plus a weak pull toward the rest posture in
            # the nullspace so the redundant DOF does not drift
            JJt = J @ J.T + (damping ** 2) * np.eye(J.shape[0])
            dq = J.T @ np.linalg.solve(JJt, err)
            if self.posture_weight:
                q_now = d.qpos[self.qadr]
                null = (np.eye(len(self.joints)) - J.T @ np.linalg.solve(JJt, J))
                dq += self.posture_weight * (null @ (self.rest - q_now))

            n = np.linalg.norm(dq)
            if n > step_limit:
                dq *= step_limit / n
            d.qpos[self.qadr] = np.clip(d.qpos[self.qadr] + dq, self.lo, self.hi)

        return d.qpos[self.qadr].copy(), pos_err, rot_err

    def forward(self, q, data_src):
        """Grasp-site pose for a given arm configuration."""
        d = self.data
        d.qpos[:] = data_src.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        mujoco.mj_comPos(self.model, d)
        return self.site_pose(d)
