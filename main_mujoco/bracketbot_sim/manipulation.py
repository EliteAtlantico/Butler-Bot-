"""Arm control and a pick-the-cube-off-the-table behaviour.

`ArmController` turns a world-frame grasp pose into rate-limited joint commands.
It re-solves IK every control tick against the world target rather than
solving once and replaying the trajectory, because the base does not hold
still: BracketBot balances, and extending a 1.2 kg arm forward moves the CoM,
so the balance loop drives the wheels to compensate. Closed-loop-in-world is
the only version that lands on the cube.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .kinematics import ARM_JOINTS, GRIPPER_JOINTS, ArmIK, rot_z

# gripper joint angle -> fingertip gap is very nearly linear over the range
GRIP_GAP_A = 0.0131     # m, gap at q = 0
GRIP_GAP_B = 0.1815     # m per rad


def grip_for_gap(gap):
    """Gripper command that opens the fingertips to `gap` metres."""
    return float(np.clip((gap - GRIP_GAP_A) / GRIP_GAP_B, 0.0, 1.0))


class StationKeeper:
    """Hold the base on a world pose with an outer position loop.

    NOTE: PickCube no longer uses this. Wrapping a second position loop around
    the balance controller -- which already regulates position through its own
    x reference -- gives two nested position loops with different dynamics, and
    the pair limit-cycles: the base swung +/-0.16 m indefinitely and never
    settled enough to start a descent. Re-seeding the balance loop's own
    reference and commanding zero velocity holds far better. Kept because an
    outer loop is still the right tool for holding a pose the balance loop was
    never told about.
    """

    def __init__(self, goal_xy, goal_yaw, kp=1.6, kyaw=1.8,
                 v_limit=0.24, w_limit=0.5):
        self.goal = np.asarray(goal_xy, float)[:2]
        self.goal_yaw = float(goal_yaw)
        self.kp, self.kyaw = kp, kyaw
        self.v_limit, self.w_limit = v_limit, w_limit

    def __call__(self, bot):
        delta = self.goal - bot.position[:2]
        heading = np.array([np.cos(bot.yaw), np.sin(bot.yaw)])
        forward_err = float(delta @ heading)
        yaw_err = float(np.arctan2(np.sin(self.goal_yaw - bot.yaw),
                                   np.cos(self.goal_yaw - bot.yaw)))
        bot.drive(float(np.clip(self.kp * forward_err, -self.v_limit, self.v_limit)),
                  float(np.clip(self.kyaw * yaw_err, -self.w_limit, self.w_limit)))
        return abs(forward_err), abs(yaw_err)


class ArmController:
    def __init__(self, bot, side="right", joint_speed=0.6, mast_speed=0.35):
        self.bot = bot
        self.side = side
        self.ik = ArmIK(bot.model, side)
        self.joints = ARM_JOINTS[side]
        self.grippers = GRIPPER_JOINTS[side]
        self.q_cmd = np.array([bot.joint_position(j) for j in self.joints])
        self.grip_cmd = bot.joint_position(self.grippers[0])
        # the mast carries the whole arm, so it moves at its own slower rate
        self.speeds = np.array([mast_speed] + [joint_speed] * (len(self.joints) - 1))
        self.last_pos_err = np.inf
        self.last_rot_err = np.inf

    # ------------------------------------------------------------------ state
    @property
    def grasp_pose(self):
        return self.ik.site_pose(self.bot.data)

    def gap(self):
        return GRIP_GAP_A + GRIP_GAP_B * self.bot.joint_position(self.grippers[0])

    # --------------------------------------------------------------- commands
    def set_gripper(self, q):
        self.grip_cmd = float(np.clip(q, 0.0, 1.0))

    def open(self, gap=0.09):
        self.set_gripper(grip_for_gap(gap))

    def close(self, gap=0.028):
        self.set_gripper(grip_for_gap(gap))

    def track(self, target_pos, target_mat, dt):
        """One control tick toward a world-frame grasp pose. Returns pos error."""
        q, pe, re = self.ik.solve(self.bot.data, target_pos, target_mat,
                                  q_init=self.q_cmd)
        self.last_pos_err, self.last_rot_err = pe, re
        step = self.speeds * dt
        self.q_cmd = self.q_cmd + np.clip(q - self.q_cmd, -step, step)
        self.apply()
        return pe

    def hold(self):
        self.apply()

    def apply(self):
        for name, val in zip(self.joints, self.q_cmd):
            self.bot.set_arm_target(name, val)
        for g in self.grippers:
            self.bot.set_arm_target(g, self.grip_cmd)

    def at(self, target_pos, tol=0.012):
        p, _ = self.grasp_pose
        return float(np.linalg.norm(np.asarray(target_pos) - p)) < tol


class PickCube:
    """Navigate to a table, then pick the cube off it.

    Phases: drive there (avoiding obstacles) -> line up -> open gripper and
    reach above the cube -> descend -> close -> lift. Each phase reports its
    own progress, so the whole thing is one `f(bot, t)` you can hand to
    `bot.step`.
    """

    PHASES = ("navigate", "align", "pregrasp", "descend", "grasp", "lift", "done")

    def __init__(self, bot, cube_body="cube", table_body="table", side="right",
                 standoff=0.34, max_reach=0.40, hull_clearance=0.22,
                 base_half=0.094, approach_height=0.14, grasp_offset=0.005,
                 lift_height=0.22, open_gap=0.085, close_gap=0.030,
                 settle_time=1.0, verbose=True):
        from .algorithms import NavigateTo

        self.bot = bot
        self.arm = ArmController(bot, side)
        self.side = side
        self.cube_body = cube_body
        self.table_body = table_body
        self.standoff = standoff
        self.max_reach = max_reach
        self.hull_clearance = hull_clearance
        self.base_half = base_half
        self.out_of_envelope = False
        self.approach_height = approach_height
        self.grasp_offset = grasp_offset
        self.lift_height = lift_height
        self.open_gap, self.close_gap = open_gap, close_gap
        self.settle_time = settle_time
        self.verbose = verbose

        self.phase = "navigate"
        self._phase_t0 = None
        # While the arm is out, clamp how far the balance loop's position
        # reference may lead the measured base. The arm's CoM shift otherwise
        # sends the base on a ~0.25 m excursion before the position term
        # catches it, which is plenty to put the robot's nose into the table
        # it is reaching over.
        self.manip_max_lead = 0.15
        self._saved_max_lead = bot.balance.max_lead
        self.grasp_yaw = 0.0
        self.cube_start_z = bot.body_position(cube_body)[2]

        base_goal, goal_yaw = self.standoff_pose()
        self.nav = NavigateTo(base_goal, goal_yaw)
        self.hold_xy = base_goal

    # ---------------------------------------------------------------- geometry
    def standoff_pose(self):
        """Where to park so the grasp site can reach the cube.

        Approach along the outward normal from the table centre through the
        cube, and offset sideways by the grasp site's lateral offset in the
        base frame so the gripper -- not the mast -- ends up over the cube.

        The stand-off distance is derived from the table's own geometry rather
        than fixed, so the base hull keeps `hull_clearance` from the table's
        near face whatever the cube's position. A fixed number works for one
        cube placement and drives the robot into the table for the rest.
        """
        cube = self.bot.body_position(self.cube_body)[:2]
        table = self.bot.body_position(self.table_body)[:2]
        a = cube - table
        n = np.linalg.norm(a)
        a = a / n if n > 1e-6 else np.array([-1.0, 0.0])

        half = self._table_half_extent(a)
        d = float((cube - table) @ a)
        # clearance = d + standoff - base_half - half  >=  hull_clearance
        needed = self.hull_clearance + self.base_half + half - d
        self.required_standoff = float(needed)
        standoff = float(np.clip(needed, self.standoff, self.max_reach))
        self.standoff_used = standoff
        self.out_of_envelope = needed > self.max_reach
        if self.out_of_envelope and self.verbose:
            print(f"    NOTE: keeping {self.hull_clearance:.2f} m off the table "
                  f"needs a {needed:.2f} m reach, arm maxes out at "
                  f"{self.max_reach:.2f} m -- standing closer than ideal")

        heading = -a
        left = np.array([-heading[1], heading[0]])
        base = cube + a * standoff - left * self.grasp_lateral_offset()
        return base, float(np.arctan2(heading[1], heading[0]))

    def _table_half_extent(self, direction):
        """Half-size of the table box along a world direction."""
        gid = mujoco.mj_name2id(self.bot.model, mujoco.mjtObj.mjOBJ_GEOM,
                                "table_top")
        if gid < 0:
            return 0.0
        size = self.bot.model.geom_size[gid]
        return float(abs(direction[0]) * size[0] + abs(direction[1]) * size[1])

    def grasp_lateral_offset(self):
        """Grasp site offset along the base's +y, at the home pose."""
        p, _ = self.arm.grasp_pose
        R = self.bot.rotation
        rel = R.T @ (p - self.bot.position)
        return float(rel[1])

    def cube_target(self, dz):
        c = self.bot.body_position(self.cube_body)
        return np.array([c[0], c[1], c[2] + dz])

    # ------------------------------------------------------------------- logic
    def _enter(self, phase, t):
        self.phase = phase
        self._phase_t0 = t
        if self.verbose:
            print(f"    [{t:6.1f}s] -> {phase}")

    def _elapsed(self, t):
        return t - (self._phase_t0 if self._phase_t0 is not None else t)

    @property
    def done(self):
        return self.phase == "done"

    def settled(self, pos_tol=0.040, speed_tol=0.040):
        """Is the base actually parked, not just commanded to be?

        Extending the arm moves the CoM and the balance loop answers by
        running the base 0.15 m forward and back. Starting the descent during
        that excursion means the gripper is chasing a world target from a
        moving base with rate-limited joints -- it arrives late, clips the
        cube, and knocks it off the table. Waiting costs a second.
        """
        err = float(np.linalg.norm(self.bot.position[:2] - self.hold_xy))
        return err < pos_tol and self.bot.ground_speed < speed_tol

    def grasped(self):
        """Is the cube actually off the table and between the fingers?"""
        c = self.bot.body_position(self.cube_body)
        p, _ = self.arm.grasp_pose
        return (c[2] > self.cube_start_z + 0.04
                and float(np.linalg.norm(c - p)) < 0.07)

    def __call__(self, bot, t):
        if self._phase_t0 is None:
            self._phase_t0 = t
        dt = bot.dt

        if self.phase == "navigate":
            self.nav(bot, t)
            self.arm.hold()
            if self.nav.done:
                bot.balance.max_lead = self.manip_max_lead
                bot.balance.station_gains(True)
                bot.balance.reset_reference(bot.state)
                self.hold_xy = bot.position[:2].copy()
                self._enter("align", t)
            return

        # From here the base holds station through the balance loop's own
        # position reference, re-seeded on arrival. Commanding zero velocity is
        # the instruction "stay where the reference says", not "coast".
        bot.drive(0.0, 0.0)

        if self.phase == "align":
            # let the base settle before reaching -- the arm is heavy enough
            # that reaching while still rolling throws the grasp off
            self.arm.open(self.open_gap)
            self.arm.hold()
            if (self._elapsed(t) > self.settle_time and self.settled()) \
                    or self._elapsed(t) > 8.0:
                self.grasp_yaw = bot.yaw
                self._enter("pregrasp", t)
            return

        mat = rot_z(self.grasp_yaw)

        if self.phase == "pregrasp":
            target = self.cube_target(self.approach_height)
            self.arm.track(target, mat, dt)
            ready = self.arm.at(target, 0.02) and self.settled()
            if ready or self._elapsed(t) > 16.0:
                self._enter("descend", t)
            return

        if self.phase == "descend":
            target = self.cube_target(self.grasp_offset)
            self.arm.track(target, mat, dt)
            ready = self.arm.at(target, 0.012) and self.settled(0.035, 0.05)
            if ready or self._elapsed(t) > 12.0:
                self._enter("grasp", t)
            return

        if self.phase == "grasp":
            self.arm.close(self.close_gap)
            self.arm.track(self.cube_target(self.grasp_offset), mat, dt)
            if self._elapsed(t) > 1.5:
                self._enter("lift", t)
            return

        if self.phase == "lift":
            target = self.cube_target(0.0) if self.grasped() else None
            # lift straight up from where the grasp happened
            p, _ = self.arm.grasp_pose
            if not hasattr(self, "_lift_from"):
                self._lift_from = p.copy()
            goal = self._lift_from + np.array([0.0, 0.0, self.lift_height])
            self.arm.track(goal, mat, dt)
            if self.arm.at(goal, 0.02) or self._elapsed(t) > 10.0:
                if self.verbose:
                    print(f"    grasp {'HELD' if self.grasped() else 'FAILED'}")
                bot.balance.max_lead = self._saved_max_lead
                bot.balance.station_gains(False)
                self._enter("done", t)
            return

        self.arm.hold()
