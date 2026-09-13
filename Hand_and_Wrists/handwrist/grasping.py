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
from .places import is_look

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
    # Reaching down to the floor shifts the CoM forward and the base creeps
    # ~7 cm toward the item before the balance loop catches it. From 0.28 m
    # that left the item too close for the folded arm to get down to it
    # (floor picks 32/36); from 0.31 m the creep lands inside the arm's
    # comfortable envelope (35/36).
    FLOOR_REACH = 0.31
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
        self.low_obstacles = self._low_obstacles()

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

        # Scenery is group 0, and group 3 holds the collision primitives the
        # household looks hide (the person's palm among them). Group 2 -- the
        # looks, and the robot's visual meshes -- is never a support.
        self._ray_group = np.array([1, 0, 0, 1, 0, 0], np.uint8)
        self._ray_hit = np.zeros(1, np.int32)

    # ---------------------------------------------------------------- support
    def support(self, est: ObjectEstimate) -> Support:
        """Ray-cast straight down from the object to find what holds it up."""
        m, d = self.bot.model, self.bot.data
        start = np.array([est.center[0], est.center[1], est.bottom_z + 0.01])
        down = np.array([0.0, 0.0, -1.0])
        for _ in range(4):
            dist = mujoco.mj_ray(m, d, start, down, self._ray_group, 1,
                                 est.body_id, self._ray_hit)
            g = int(self._ray_hit[0])
            if dist < 0 or g < 0:
                return Support("floor", 0.0)
            # A camera estimate carries no body id, and its bottom can sit a
            # little low -- so the ray may hit the item itself or another loose
            # one. Anything that moves (loose items, the robot's own collision
            # geoms, also in group 3) is never furniture: look through it.
            b = m.geom_bodyid[g]
            if m.body_weldid[b] != 0:
                start = start + down * (dist + 0.002)
                continue
            break
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
            return Support("floor", 0.0)
        # The surface height is the TOP of what was hit, not where the ray
        # hit it: stepping through a loose item can restart the ray inside the
        # tabletop, which then reports its underside -- 3 cm too low, and the
        # fingertips were driven into the table.
        R = d.geom_xmat[g].reshape(3, 3)
        aabb_c, aabb_h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        top = float(d.geom_xpos[g][2] + (R @ aabb_c)[2] + (np.abs(R) @ aabb_h)[2])
        return Support("table", top, center=d.geom_xpos[g].copy(),
                       R=R.copy(), half=m.geom_size[g].copy())

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
        candidates = []
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
                natural = grasp[:2] - hvec * reach - lvec * self.lateral[side]
                for shift in self.LATERAL_SHIFTS:
                    base_xy = natural + lvec * shift
                    # Turning is what a differential drive pays for: turn to
                    # face the parking spot, drive, turn to the grasp heading.
                    # Scoring only the final heading against the current one
                    # prefers spots that need a U-turn on arrival. A long
                    # reach costs balance; a sideways shift costs reach.
                    dist = float(np.linalg.norm(base_xy - here))
                    if dist > 0.15:
                        bearing = float(np.arctan2(*(base_xy - here)[::-1]))
                        turn = abs(_wrap(bearing - yaw_now)) + abs(_wrap(heading - bearing))
                    else:
                        turn = abs(_wrap(heading - yaw_now))
                    cost = (dist + 0.35 * turn + 1.5 * reach + 0.3 * abs(wrist)
                            + 1.0 * abs(shift) + (0.02 if side == "left" else 0.0)
                            + (0.0 if self.creep_clear(base_xy, heading) else self.BLOCKED_COST))
                    candidates.append((cost, side, heading, reach, base_xy, mat,
                                       pre, lift, wrist))

        # IK is the expensive check: run it cheapest-first and stop early
        candidates.sort(key=lambda c: c[0])
        plans = []
        for cost, side, heading, reach, base_xy, mat, pre, lift, wrist in candidates:
            err = self._check_ik(side, base_xy, heading, (pre, grasp, lift), mat)
            if err is None:
                continue
            plans.append(GraspPlan(
                name=est.name, side=side, kind=kind, base_xy=base_xy,
                base_yaw=heading, grasp_pos=grasp, grasp_mat=mat,
                pregrasp_pos=pre, lift_pos=lift, open_gap=open_gap,
                close_gap=close_gap, wrist_yaw=wrist, reach=reach,
                ik_err=err, cost=cost + 20.0 * err))
            if len(plans) >= self.MAX_FEASIBLE:
                break
        plans.sort(key=lambda p: p.cost)
        return plans[:max_plans] if max_plans else plans

    # ------------------------------------------------------- base creep room
    # While the arm works the base creeps forward -- 8-20 cm measured on table
    # picks, as the balance loop answers the arm's CoM shift -- and it runs
    # under a tabletop (0.37 m up, above the 0.29 m hull) until it meets a
    # LEG. Pressed against a leg, a balancing robot cannot back off (it has to
    # roll forward to lean back, and the leg is in the way), so everything
    # after the pick failed. Parking spots whose creep corridor holds
    # anything low are penalised, and sideways-shifted spots are offered so
    # the base can sit clear of a corner leg while the arm reaches across.
    CREEP = 0.20            # m of forward creep to leave room for
    HULL_TOP = 0.30         # m: scenery lower than this can hit the base
    LATERAL_SHIFTS = (0.0, 0.06, -0.06, 0.12, -0.12)
    BLOCKED_COST = 5.0
    MAX_FEASIBLE = 4        # stop IK-checking once this many plans pass

    def _low_obstacles(self):
        """(lo_xy, hi_xy, body) of every fixed geom low enough to hit the base."""
        m, d = self.bot.model, self.bot.data
        out = []
        for g in range(m.ngeom):
            b = m.geom_bodyid[g]
            if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or m.body_weldid[b] != 0
                    or is_look(m, g)):
                continue
            R = d.geom_xmat[g].reshape(3, 3)
            c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
            h = np.abs(R) @ m.geom_aabb[g, 3:]
            if c[2] - h[2] < self.HULL_TOP and c[2] + h[2] > 0.01:
                out.append(((c - h)[:2], (c + h)[:2], int(b)))
        return out

    def creep_clear(self, base_xy, heading, exclude_body=-1):
        """Is the strip the base could creep into, parked here, free of low
        scenery? Strip: the hull's width, from its back to CREEP past its nose."""
        c, s = np.cos(heading), np.sin(heading)
        x0, x1 = -0.12, self.hull_front + self.CREEP
        y0, y1 = -0.20, 0.20
        R = np.array([[c, -s], [s, c]])
        corners = [np.asarray(base_xy) + R @ np.array(p)
                   for p in ((x0, y0), (x0, y1), (x1, y0), (x1, y1))]
        for lo, hi, b in self.low_obstacles:
            if b == exclude_body:
                continue
            pts = [lo, hi, np.array([lo[0], hi[1]]), np.array([hi[0], lo[1]]), (lo + hi) / 2]
            for p in pts:
                q = R.T @ (p - base_xy)
                if x0 <= q[0] <= x1 and y0 <= q[1] <= y1:
                    return False
            for p in corners:
                if np.all(p >= lo) and np.all(p <= hi):
                    return False
        return True

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


