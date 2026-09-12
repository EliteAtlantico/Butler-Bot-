"""Grasp planning: object estimate -> which arm, wrist angle, grasp pose, and
where the base should park.

A grasp is only as good as the place the robot stands to make it. The arm
reaches ~0.36 m forward with the gripper pointing down (see the reach map in
the README), and the base hull sticks out 0.094 m in front of the mast, so an
item more than ~0.2 m in from a table edge simply cannot be grasped from
above. The planner therefore enumerates candidates -- every edge of the
supporting table (or any heading on open floor), each arm, each equivalent
wrist yaw -- checks each one with the arm's IK from the base pose it implies,
and ranks the survivors.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bracketbot_sim.kinematics import ArmIK, rot_z

from .gripper import Gripper
from .objects import ObjectEstimate, ObjectSpec

PAD_HALF_Z = 0.018        # fingertip pad half-height (build_dynamic_model.py)
SURFACE_CLEAR = 0.005     # keep the pad bottoms this far off the surface
BELOW_TOP = 0.045         # tall items: pads centred this far below the top,
                          # which keeps the palm (65 mm above the pads) clear
APPROACH_H = 0.12         # pre-grasp hover above a top grasp
SIDE_BACKOFF = 0.12       # pre-grasp stand-off for a side grasp
LIFT_H = 0.12


def rot_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


@dataclass
class Support:
    """What the object is resting on."""
    kind: str                     # "floor" or "table"
    top_z: float
    center: np.ndarray | None = None
    R: np.ndarray | None = None
    half: np.ndarray | None = None


@dataclass
class GraspPlan:
    name: str
    side: str                     # which arm
    kind: str                     # "top" or "side"
    base_xy: np.ndarray           # where to park the base
    base_yaw: float
    grasp_pos: np.ndarray         # grasp-site targets, world frame
    grasp_mat: np.ndarray
    pregrasp_pos: np.ndarray
    lift_pos: np.ndarray
    open_gap: float               # pad gap on approach (m)
    close_gap: float              # pad gap commanded when closing (m)
    wrist_yaw: float              # wrist yaw relative to the base (rad)
    reach: float                  # grasp point distance ahead of the base (m)
    ik_err: float
    cost: float

    def describe(self):
        return (f"{self.side} arm, {self.kind} grasp, wrist {np.rad2deg(self.wrist_yaw):+.0f} deg, "
                f"park at ({self.base_xy[0]:.2f}, {self.base_xy[1]:.2f}) "
                f"facing {np.rad2deg(self.base_yaw):+.0f} deg, reach {self.reach:.2f} m, "
                f"gap {self.open_gap * 1000:.0f}->{self.close_gap * 1000:.0f} mm")


class GraspPlanner:
    MIN_REACH = 0.22                        # hand must clear the hull and wheels
    MAX_REACH = {"top": 0.36, "side": 0.44}
    FLOOR_REACH = 0.28
    CLEARANCE = 0.09                        # hull front to furniture edge; the
                                            # base parks to a few cm
    MAX_POS_ERR = 0.008
    MAX_ROT_ERR = 0.06

    def __init__(self, bot):
        self.bot = bot
        m = bot.model
        self.scratch = mujoco.MjData(m)
        self.ik = {s: ArmIK(m, s) for s in ("right", "left")}
        self.max_gap = Gripper(bot, "right").max_gap

        hull = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "col_base_hull")
        self.hull_front = float(m.geom_pos[hull][0] + m.geom_size[hull][0])

        # grasp-site offset from the base at the home pose, in the base frame:
        # parking so the arm reaches straight ahead keeps the most reach in hand
        d = self.scratch
        d.qpos[:] = m.qpos0
        mujoco.mj_kinematics(m, d)
        chassis = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self.lateral = {}
        for s, ik in self.ik.items():
            p, _ = ik.site_pose(d)
            self.lateral[s] = float(p[1] - d.xpos[chassis][1])

        self._ray_group = np.array([1, 0, 0, 0, 0, 0], np.uint8)
        self._ray_hit = np.zeros(1, np.int32)

    # ---------------------------------------------------------------- support
    def support(self, est: ObjectEstimate) -> Support:
        """Ray-cast straight down from the object to find what holds it up."""
        m, d = self.bot.model, self.bot.data
        start = np.array([est.center[0], est.center[1], est.bottom_z + 0.01])
        dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]),
                             self._ray_group, 1, est.body_id, self._ray_hit)
        g = int(self._ray_hit[0])
        if dist < 0 or g < 0 or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            return Support("floor", 0.0)
        return Support("table", float(start[2] - dist), center=d.geom_xpos[g].copy(),
                       R=d.geom_xmat[g].reshape(3, 3).copy(), half=m.geom_size[g].copy())

    def _headings(self, est, support, kind):
        """(heading, reach) pairs the base could approach from."""
        c = est.center[:2]
        if support.kind == "table":
            out = []
            for k in (0, 1):
                axis = support.R[:2, k]
                axis = axis / np.linalg.norm(axis)
                for sgn in (1.0, -1.0):
                    n = sgn * axis                       # outward edge normal
                    inset = support.half[k] - float((c - support.center[:2]) @ n)
                    reach = max(inset + self.hull_front + self.CLEARANCE, self.MIN_REACH)
                    if reach <= self.MAX_REACH[kind]:
                        out.append((float(np.arctan2(-n[1], -n[0])), reach))
            return out
        # open floor: head straight at it, or line the body up with its axis --
        # but only from the robot's own side, or the route to the parking spot
        # runs straight over the thing it is trying to pick up
        here = self.bot.position[:2]
        direct = float(np.arctan2(c[1] - here[1], c[0] - here[0]))
        headings = [direct]
        if est.axis_yaw is not None:
            for h in (est.axis_yaw, est.axis_yaw + np.pi):
                if abs(_wrap(h - direct)) < 1.2:
                    headings.append(_wrap(h))
        return [(h, self.FLOOR_REACH) for h in headings]

    # ------------------------------------------------------------------- plan
    def plan(self, est: ObjectEstimate, spec: ObjectSpec, sides=("right", "left"),
             max_plans=None):
        """All feasible grasp plans for `est`, best first. Empty if none."""
        bot = self.bot
        support = self.support(est)
        kind = spec.grasp
        if est.width + 0.02 > self.max_gap:
            return []
        open_gap = min(est.width + 0.045, self.max_gap - 0.005)
        close_gap = max(est.width - spec.squeeze, 0.0)

        # grasp height
        floor_z = support.top_z + PAD_HALF_Z + SURFACE_CLEAR
        if kind == "side" or spec.grip_at_center:
            z = max(float(est.center[2]), floor_z)
        else:
            z = max(est.top_z - BELOW_TOP, floor_z)
        grasp = np.array([est.center[0], est.center[1], z])

        here, yaw_now = bot.position[:2], bot.yaw
        plans = []
        for heading, reach in self._headings(est, support, kind):
            hvec = np.array([np.cos(heading), np.sin(heading)])
            lvec = np.array([-hvec[1], hvec[0]])
            if kind == "top":
                if est.axis_yaw is None:
                    wrist = 0.0
                else:
                    # the grasp is symmetric under a half turn: take the
                    # representative closest to straight ahead
                    wrist = _wrap(est.axis_yaw - heading)
                    if wrist > np.pi / 2:
                        wrist -= np.pi
                    elif wrist <= -np.pi / 2:
                        wrist += np.pi
                mat = rot_z(heading + wrist)
                pre = grasp + np.array([0.0, 0.0, APPROACH_H])
            else:
                wrist = 0.0
                mat = rot_z(heading) @ rot_y(-np.pi / 2)
                pre = grasp - SIDE_BACKOFF * np.array([hvec[0], hvec[1], 0.0])
            lift = grasp + np.array([0.0, 0.0, LIFT_H])

            for side in sides:
                base_xy = grasp[:2] - hvec * reach - lvec * self.lateral[side]
                err = self._check_ik(side, base_xy, heading, (pre, grasp, lift), mat)
                if err is None:
                    continue
                # Turning is what a differential drive pays for: turn to face
                # the parking spot, drive, turn to the grasp heading. Scoring
                # only the final heading against the current one prefers spots
                # that need a U-turn on arrival. A long reach costs balance.
                dist = float(np.linalg.norm(base_xy - here))
                if dist > 0.15:
                    bearing = float(np.arctan2(*(base_xy - here)[::-1]))
                    turn = abs(_wrap(bearing - yaw_now)) + abs(_wrap(heading - bearing))
                else:
                    turn = abs(_wrap(heading - yaw_now))
                cost = (dist + 0.35 * turn + 1.5 * reach + 0.3 * abs(wrist)
                        + 20.0 * err + (0.02 if side == "left" else 0.0))
                plans.append(GraspPlan(
                    name=est.name, side=side, kind=kind, base_xy=base_xy,
                    base_yaw=heading, grasp_pos=grasp, grasp_mat=mat,
                    pregrasp_pos=pre, lift_pos=lift, open_gap=open_gap,
                    close_gap=close_gap, wrist_yaw=wrist, reach=reach,
                    ik_err=err, cost=cost))
        plans.sort(key=lambda p: p.cost)
        return plans[:max_plans] if max_plans else plans

    def reachable_from_here(self, plan):
        """Can the plan's arm reach every target from where the base is now?"""
        return self._check_ik(plan.side, self.bot.position[:2], self.bot.yaw,
                              (plan.pregrasp_pos, plan.grasp_pos, plan.lift_pos),
                              plan.grasp_mat) is not None

    def _check_ik(self, side, base_xy, heading, targets, mat):
        """Worst IK position error over `targets` from a hypothetical base
        pose, or None if any target is out of reach."""
        m, d = self.bot.model, self.scratch
        d.qpos[:] = self.bot.data.qpos
        d.qpos[0:2] = base_xy
        d.qpos[3:7] = [np.cos(heading / 2), 0.0, 0.0, np.sin(heading / 2)]
        ik = self.ik[side]
        q = None
        worst = 0.0
        for target in targets:
            q, pe, re = ik.solve(d, target, mat, q_init=q)
            if pe > self.MAX_POS_ERR or re > self.MAX_ROT_ERR:
                return None
            worst = max(worst, pe)
        return worst


