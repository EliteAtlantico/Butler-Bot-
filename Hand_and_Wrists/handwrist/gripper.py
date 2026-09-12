"""The gripper: finger-gap calibration and touch sensing.

The gap is calibrated by sweeping the finger joints on a scratch copy of the
model and measuring the distance between the pads' inner faces, rather than
trusting a hand-fitted constant -- the pads swing on hinges, so the mapping is
not linear and it changes if anyone moves the pads in build_dynamic_model.py.

"Am I holding it?" is answered the way the real robot could: both pads report
contact with the same object AND the fingers stalled open short of the
commanded close, i.e. something is in the way. No peeking at where the object
actually is.
"""
from __future__ import annotations

import mujoco
import numpy as np

from bracketbot_sim.kinematics import GRIPPER_JOINTS

PAD_GEOMS = {
    "right": ("pad_left_finger__left_finger", "pad_right_finger__right_finger"),
    "left": ("pad_l_left_finger__left_finger", "pad_l_right_finger__right_finger"),
}


def _subtree(model, root):
    out = {root}
    for b in range(root + 1, model.nbody):
        if model.body_parentid[b] in out:
            out.add(b)
    return out


class Gripper:
    def __init__(self, bot, side="right"):
        self.bot = bot
        self.side = side
        m = bot.model
        self.joints = GRIPPER_JOINTS[side]
        self.qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                     for j in self.joints]
        self.dofadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                       for j in self.joints]
        self.pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                     for g in PAD_GEOMS[side]]
        chassis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.robot_bodies = _subtree(m, chassis)
        self._calibrate()

    # ------------------------------------------------------------ calibration
    def _calibrate(self):
        m = self.bot.model
        d = mujoco.MjData(m)
        d.qpos[:] = self.bot.data.qpos
        half_thick = m.geom_size[self.pads[0]][1]
        self.q_table = np.linspace(0.0, 1.0, 41)
        gaps = []
        for q in self.q_table:
            for a in self.qadr:
                d.qpos[a] = q
            mujoco.mj_kinematics(m, d)
            pa, pb = d.geom_xpos[self.pads[0]], d.geom_xpos[self.pads[1]]
            gaps.append(np.linalg.norm(pa - pb) - 2 * half_thick)
        self.gap_table = np.maximum.accumulate(np.array(gaps))
        self.max_gap = float(self.gap_table[-1])
        self.min_gap = float(self.gap_table[0])

    def q_for_gap(self, gap):
        """Finger joint command that puts the pad inner faces `gap` m apart."""
        return float(np.interp(gap, self.gap_table, self.q_table))

    # ------------------------------------------------------------------ state
    @property
    def q(self):
        return float(self.bot.data.qpos[self.qadr[0]])

    @property
    def qdot(self):
        return float(self.bot.data.qvel[self.dofadr[0]])

    @property
    def gap(self):
        return float(np.interp(self.q, self.q_table, self.gap_table))

    def pad_contacts(self):
        """(bodies touching pad A, bodies touching pad B), robot excluded."""
        m, d = self.bot.model, self.bot.data
        touching = (set(), set())
        for i in range(d.ncon):
            c = d.contact[i]
            for k, pad in enumerate(self.pads):
                other = c.geom2 if c.geom1 == pad else c.geom1 if c.geom2 == pad else -1
                if other < 0:
                    continue
                b = int(m.geom_bodyid[other])
                if b not in self.robot_bodies:
                    touching[k].add(b)
        return touching

    def pinched_body(self):
        """Body id held between both pads, or -1."""
        a, b = self.pad_contacts()
        both = a & b
        return next(iter(both)) if both else -1

    def stalled(self, q_cmd, tol=0.02, still=0.05):
        """Fingers have stopped short of the command: something is between them."""
        return self.q - q_cmd > tol and abs(self.qdot) < still

    def holding(self, q_cmd):
        """Both pads on the same object, fingers blocked open by it."""
        return self.pinched_body() >= 0 and self.stalled(q_cmd)
