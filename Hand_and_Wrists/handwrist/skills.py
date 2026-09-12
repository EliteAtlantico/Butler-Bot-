"""Pick: find where to stand, drive there, reach, grip, check, lift, stow.

Runs as an ordinary `f(bot, t)` controller, like everything in
main_mujoco/bracketbot_sim/algorithms.py:

    pick = Pick(bot, "remote")
    while not pick.done:
        bot.step(0.1, controller=pick)
    print(pick.succeeded, pick.failure)

Arm motion goes through the team's `ArmController`, which re-solves IK every
tick against the WORLD target. That matters because the base never holds
still: it balances, and swinging a 1.2 kg arm forward shifts the CoM enough to
roll the wheels a few centimetres.

What the hand does differs by object -- the planner picks the arm, grasp
type, wrist yaw and finger gap -- but the sequence is the same for all of
them. Every step that can fail has a check, and a failed grasp backs off,
re-estimates the object (it may have been nudged) and tries again.
"""
from __future__ import annotations

import mujoco
import numpy as np

from bracketbot_sim.kinematics import rot_z
from bracketbot_sim.manipulation import ArmController

from .grasping import GraspPlanner
from .gripper import Gripper
from .objects import CATALOGUE, truth_estimate

STOW_FWD = 0.22      # carry the item this far ahead of the base...
STOW_Z = 0.72        # ...at this height, where every wrist yaw is reachable
HAND_SPEED = 0.20    # m/s, straight-line hand speed
DESCEND_SPEED = 0.10  # m/s, the last few cm onto the object
WRIST_SPEED = 1.2    # rad/s
MAX_HAND_LEAD = 0.05  # m the commanded hand may run ahead of the real one


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def _rotate_toward(Ra, Rb, max_angle):
    """Ra turned toward Rb by at most `max_angle`, about the shortest axis."""
    qa, qb, qi, qd = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qa, Ra.flatten())
    mujoco.mju_mat2Quat(qb, Rb.flatten())
    mujoco.mju_negQuat(qi, qa)
    mujoco.mju_mulQuat(qd, qb, qi)
    if qd[0] < 0:
        qd = -qd
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, qd, 1.0)
    angle = float(np.linalg.norm(vel))
    if angle <= max_angle:
        return Rb.copy()
    qs, q = np.zeros(4), np.zeros(4)
    mujoco.mju_axisAngle2Quat(qs, vel / angle, max_angle)
    mujoco.mju_mulQuat(q, qs, qa)
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R.reshape(3, 3)


def truth_estimator(bot, spec):
    return truth_estimate(bot.model, bot.data, spec)


class ApproachPose:
    """Drive the base to a pose on open floor. No obstacle avoidance: this is
    the last-metre controller, meant to take over from NavigateTo once the
    route is clear.

    Designed against the balance loop's measured behaviour (scene_flat, from
    standstill), not against how a wheeled base "should" behave:

      * Stopping is slow. A balancing robot has to lean back before it can
        brake, so pinning the position reference at 0.2 m/s runs 0.24 m past
        it, at 0.3 m/s 0.36 m -- about 1.2 s x speed -- and drifts back over
        several seconds. Proportional speed-toward-goal loops on top of that
        limit-cycle by +/-0.2 m. So: cruise at a fixed speed, and start
        braking when the distance left is 1.2 s x the MEASURED speed, pinning
        the reference on the goal. Creep at 0.1 m/s to trim.
      * Turning on the spot has a dead band: the tyres scrub, and the yaw
        loop's proportional torque stalls 5-8 deg short. Commanding a yaw
        rate with a 0.15 rad/s floor, then pinning the yaw reference inside
        0.03 rad, lands within 2 deg in 2-3 s.

    Parking is good to a few cm. That is enough because the arm tracks WORLD
    targets, and Pick re-checks reachability from wherever the base stopped.

    Stages -- a differential drive cannot fix a sideways error on the spot,
    so it lines up from behind first:

      turn1  face a point ENTRY m short of the goal on the goal heading
      leg1   drive to it
      turn2  turn to the goal heading
      leg2   drive in along that line, steering out lateral error
      turn3  square up the final heading
    """

    ENTRY = 0.35
    BRAKE_TIME = 1.4     # s: measured stopping distance / speed is 1.2 s at
                         # steady speed; more while still accelerating
    CRUISE, CREEP = 0.25, 0.10

    def __init__(self, goal_xy, goal_yaw, pos_tol=0.04, yaw_tol=0.05):
        self.goal = np.asarray(goal_xy, float)
        self.goal_yaw = float(goal_yaw)
        self.h = np.array([np.cos(goal_yaw), np.sin(goal_yaw)])
        self.pos_tol, self.yaw_tol = pos_tol, yaw_tol
        self.stage = "turn1"
        self._x_hold = None
        self._brake_ref = None
        self._still_since = None
        self._brakes = 0

    def error(self, bot):
        return (float(np.linalg.norm(self.goal - bot.position[:2])),
                abs(_wrap(self.goal_yaw - bot.yaw)))

    # -- primitives -----------------------------------------------------------
    def _turn(self, bot, target):
        """Turn on the spot to `target`. True when there."""
        if self._x_hold is None:
            self._x_hold = bot.odometry          # don't wander while turning
        bot.balance.x_ref = self._x_hold
        err = _wrap(target - bot.yaw)
        if abs(err) < 0.03:
            bot.drive(0.0, 0.0)
            bot.balance.yaw_ref = bot.yaw
            return abs(bot.state[5]) < 0.1
        bot.drive(0.0, float(np.sign(err) * np.clip(1.2 * abs(err), 0.15, 0.8)))
        return False

    def _drive(self, bot, point, line_yaw=None, tol=None, max_speed=None):
        """Drive to `point`, heading at it -- or, given `line_yaw`, along that
        line with the lateral error steered out. True once stopped on it."""
        tol = self.pos_tol if tol is None else tol
        self._x_hold = None
        here = bot.position[:2]
        d = point - here
        e = float(d @ np.array([np.cos(bot.yaw), np.sin(bot.yaw)]))  # along the nose

        if self._brake_ref is not None:
            bot.drive(0.0, 0.0)
            bot.balance.x_ref = self._brake_ref
            # Stopped means STAYING stopped: the base passes through zero
            # speed at the top of every overshoot on its way back, and
            # treating that instant as "parked" re-triggers a creep that
            # overshoots again.
            if bot.ground_speed > 0.03:
                self._still_since = None
                return False
            if self._still_since is None:
                self._still_since = bot.time
            if bot.time - self._still_since < 0.6:
                return False
            self._brake_ref = None
            self._still_since = None
            self._brakes += 1
            # good enough, or out of patience after several trims
            return abs(e) < tol or (self._brakes >= 4 and abs(e) < 2 * tol)

        sgn = 1.0 if e >= 0 else -1.0
        if abs(e) <= self.BRAKE_TIME * max(sgn * bot.forward_velocity, 0.0) + 0.015:
            self._brake_ref = bot.odometry + e
            bot.drive(0.0, 0.0)
            bot.balance.x_ref = self._brake_ref
            return False

        if line_yaw is None:
            want = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 0.15 else bot.yaw
        else:
            left = np.array([-np.sin(line_yaw), np.cos(line_yaw)])
            lat = float((here - point) @ left)
            want = line_yaw - float(np.clip(2.5 * lat, -0.3, 0.3)) * sgn
        speed = self.CRUISE if abs(e) > 0.4 else self.CREEP
        if max_speed is not None:
            speed = min(speed, max_speed)
        bot.drive(sgn * speed, float(np.clip(1.8 * _wrap(want - bot.yaw), -0.6, 0.6)))
        return False

    def __call__(self, bot):
        """One tick. Returns True once parked."""
        here = bot.position[:2]
        entry = self.goal - self.ENTRY * self.h

        if self.stage == "turn1":
            d = entry - here
            rel = here - self.goal
            s = float(rel @ self.h)
            lat = float(rel @ np.array([-self.h[1], self.h[0]]))
            if np.linalg.norm(d) < 0.08:
                self.stage = "turn2"
            elif (-self.ENTRY - 0.25 < s < -0.05 and abs(lat) < 0.04
                  and abs(_wrap(self.goal_yaw - bot.yaw)) < 0.15):
                self.stage = "leg2"                 # already lined up
            else:
                if self._turn(bot, float(np.arctan2(d[1], d[0]))):
                    self.stage = "leg1"
                return False

        if self.stage == "leg1":
            d = entry - here
            # the entry point only has to be roughly right -- leg2 lines up
            # from wherever this stops -- so don't burn time trimming it
            if self._drive(bot, entry, tol=0.10):
                self.stage = "turn2"
            elif (self._brake_ref is None and np.linalg.norm(d) > 0.25 and
                  abs(_wrap(float(np.arctan2(d[1], d[0])) - bot.yaw)) > 0.8):
                self.stage = "turn1"                # knocked off course
            return False

        if self.stage == "turn2":
            if self._turn(bot, self.goal_yaw):
                self.stage = "leg2"
            return False

        if self.stage == "leg2":
            # always creep the last leg: it usually ends facing furniture, and
            # starting it at cruise speed overshot into the table's edge
            if self._drive(bot, self.goal, self.goal_yaw, max_speed=self.CREEP):
                self.stage = "turn3"
            return False

        # turn3
        if self._turn(bot, self.goal_yaw):
            if np.linalg.norm(self.goal - here) > 0.08:
                self.stage = "leg2"
                return False
            bot.drive(0.0, 0.0)
            return True
        return False


class Pick:
    PHASES = ("plan", "approach", "settle", "pregrasp", "insert", "close",
              "lift", "stow", "verify", "backoff", "done", "failed")

    def __init__(self, bot, name, estimator=truth_estimator, sides=("right", "left"),
                 max_retries=2, control_period=0.02, verbose=True):
        if name not in CATALOGUE:
            raise KeyError(f"unknown object {name!r}; know {sorted(CATALOGUE)}")
        self.bot = bot
        self.spec = CATALOGUE[name]
        self.estimator = estimator
        self.sides = sides
        self.planner = GraspPlanner(bot)
        self.arms = {s: ArmController(bot, s) for s in ("right", "left")}
        self.grippers = {s: Gripper(bot, s) for s in ("right", "left")}
        self.max_retries = max_retries
        self.control_period = control_period
        self.verbose = verbose

        self.phase = "plan"
        self.plan = None
        self.estimate = None
        self.failure = None
        self.retries = 0
        self._reparks = 0
        self.history = []            # (time, phase)
        self._t0 = None
        self._next_tick = -np.inf
        self._saved_max_lead = bot.balance.max_lead
        self._manip = False
        self._cmd = None             # straight-line hand target (pos, mat)

    # ------------------------------------------------------------- helpers
    @property
    def done(self):
        return self.phase in ("done", "failed")

    @property
    def succeeded(self):
        return self.phase == "done"

    @property
    def arm(self):
        return self.arms[self.plan.side]

    @property
    def gripper(self):
        return self.grippers[self.plan.side]

    def _enter(self, phase, t, note=""):
        self.phase = phase
        self._t0 = t
        self.history.append((t, phase))
        if self.verbose:
            print(f"    [{t:6.1f}s] {phase:9s} {note}")

    def _elapsed(self, t):
        return t - self._t0

    def _manipulating(self, on):
        """Stiffen the balance loop's position hold while the arm works."""
        b = self.bot.balance
        if on and not self._manip:
            b.max_lead = 0.15
            b.station_gains(True)
            b.reset_reference(self.bot.state)
        elif not on and self._manip:
            b.max_lead = self._saved_max_lead
            b.station_gains(False)
            b.reset_reference(self.bot.state)
        self._manip = on

    def _still(self, speed=0.04):
        return self.bot.ground_speed < speed

    def _fail(self, reason, t):
        self.failure = reason
        self._manipulating(False)
        self._enter("failed", t, reason)

    def _retry(self, reason, t):
        self.retries += 1
        if self.retries > self.max_retries:
            self._fail(reason, t)
            return
        self.arm.set_gripper(self.gripper.q_for_gap(self.plan.open_gap))
        self._backoff_to = self.arm.grasp_pose[0] + np.array([0.0, 0.0, 0.10])
        self._enter("backoff", t, f"{reason}, retry {self.retries}/{self.max_retries}")

    def _track(self, target, mat, dt, speed=HAND_SPEED):
        """Move the hand toward a world pose along a straight line.

        Handing the final pose straight to the IK makes every joint slew at
        its own rate limit, and the hand sweeps a curve that dipped 8 cm below
        the target on the way in -- enough to clip the mug with a fingertip,
        after which the 800-gain arm servos shoved the whole robot back 40 cm.
        Walking an intermediate target along the line keeps the path where
        the planner checked it.
        """
        p_now, R_now = self.arm.grasp_pose
        if self._cmd is None:
            self._cmd = (p_now, R_now)
        p, R = self._cmd
        d = np.asarray(target, float) - p
        n = float(np.linalg.norm(d))
        # Wait for the real hand only if it is lagging ALONG the path. A lag
        # measured as plain distance also counts the base drifting sideways,
        # and freezing the target then leaves the hand parked on whatever it
        # was touching while the base rolls away.
        lead = float((p - p_now) @ d) / n if n > 1e-9 else 0.0
        if lead < MAX_HAND_LEAD:
            step = speed * dt
            p = np.asarray(target, float).copy() if n <= step else p + d * (step / n)
            R = _rotate_toward(R, mat, WRIST_SPEED * dt)
        self._cmd = (p, R)
        self.arm.track(p, R, dt)

    def _arms_home(self, dt):
        """Rate-limit both arms back to the home pose. True once there."""
        home = True
        for a in self.arms.values():
            step = a.speeds * dt
            a.q_cmd = a.q_cmd + np.clip(-a.q_cmd, -step, step)
            a.apply()
            q = np.array([self.bot.joint_position(j) for j in a.joints])
            home &= bool(np.all(np.abs(q) < 0.08))
        return home

    def _stow_target(self):
        bot = self.bot
        h = np.array([np.cos(bot.yaw), np.sin(bot.yaw)])
        l = np.array([-h[1], h[0]])
        xy = bot.position[:2] + STOW_FWD * h + self.planner.lateral[self.plan.side] * l
        mat = rot_z(_wrap(bot.yaw - self.plan.base_yaw)) @ self.plan.grasp_mat
        # never carry it lower than it was lifted: off a 0.6 m side table the
        # lift already ends above STOW_Z, and stowing would put it back down
        z = max(STOW_Z, float(self.plan.lift_pos[2]))
        return np.array([xy[0], xy[1], z]), mat

    # ---------------------------------------------------------------- tick
    def __call__(self, bot, t):
        if t < self._next_tick - self.control_period:   # sim was reset
            self._next_tick = t
        if t < self._next_tick:
            return
        self._next_tick = t + self.control_period
        dt = self.control_period
        if self._t0 is None:
            self._t0 = t

        if bot.fallen and not self.done:
            self._fail("fell over", t)
        getattr(self, f"_{self.phase}")(bot, t, dt)

    # --------------------------------------------------------------- phases
    def _plan(self, bot, t, dt):
        self.estimate = self.estimator(bot, self.spec)
        plans = self.planner.plan(self.estimate, self.spec, self.sides)
        if not plans:
            self._fail("no reachable grasp", t)
            return
        self.plan = plans[0]
        self.approach = ApproachPose(self.plan.base_xy, self.plan.base_yaw)
        rho, dyaw = self.approach.error(bot)
        if (self._manip and rho < 0.08 and dyaw < 0.08
                and self.planner.reachable_from_here(self.plan)):
            # a retry from where we already stand: no need to drive
            self._enter("settle", t, self.plan.describe())
        else:
            self._manipulating(False)
            self._enter("approach", t, self.plan.describe())

    def _approach(self, bot, t, dt):
        # Never drive with an arm out: it moves the CoM enough that the base
        # wobbles the whole way and parks badly. After a failed grasp the arm
        # is still over the table, so bring it home first.
        if not self._arms_home(dt):
            bot.drive(0.0, 0.0)
            return
        self._cmd = None
        if self.approach(bot):
            self._manipulating(True)
            self._enter("settle", t)
        elif self._elapsed(t) > 40.0:
            self._fail("could not reach the parking spot", t)

    def _settle(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self.arm.set_gripper(self.gripper.q_for_gap(self.plan.open_gap))
        self.arm.hold()
        if (self._elapsed(t) > 0.8 and self._still()) or self._elapsed(t) > 6.0:
            # the base parks to a few cm, not exactly: make sure the grasp is
            # still in reach from where it actually stopped
            if self.planner.reachable_from_here(self.plan):
                self._cmd = None
                self._enter("pregrasp", t)
            elif self._reparks < 2:
                self._reparks += 1
                self._manipulating(False)
                self._enter("plan", t, "parked out of reach, re-planning")
            else:
                self._fail("parked out of reach", t)

    def _pregrasp(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p = self.plan.pregrasp_pos
        self._track(p, self.plan.grasp_mat, dt)
        if self.arm.at(p, 0.02) and self._still(0.05):
            self._enter("insert", t)
        elif self._elapsed(t) > 14.0:
            if self.arm.at(p, 0.05):
                self._enter("insert", t, "(pre-grasp loose)")
            else:
                self._retry("arm could not reach the pre-grasp", t)

    def _insert(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p = self.plan.grasp_pos
        self._track(p, self.plan.grasp_mat, dt, DESCEND_SPEED)
        if self.arm.at(p, 0.010) and self._still(0.05):
            self._enter("close", t)
        elif self._elapsed(t) > 8.0:
            if self.arm.at(p, 0.025):
                self._enter("close", t, "(grasp pose loose)")
            else:
                self._retry("arm could not reach the grasp", t)

    def _close(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        q_close = self.gripper.q_for_gap(self.plan.close_gap)
        self.arm.set_gripper(q_close)
        self._track(self.plan.grasp_pos, self.plan.grasp_mat, dt, DESCEND_SPEED)
        held = self.gripper.pinched_body()
        right_thing = self.estimate.body_id < 0 or held == self.estimate.body_id
        if self._elapsed(t) > 0.3 and self.gripper.holding(q_close) and right_thing:
            self._enter("lift", t, f"pads closed at {self.gripper.gap * 1000:.0f} mm")
        elif self._elapsed(t) > 1.8:
            self._retry("fingers closed on nothing", t)

    def _lift(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p = self.plan.lift_pos
        # Lift by handing the IK the final pose, NOT along a straight line.
        # Walking the target up in small steps off the floor let the solver
        # creep into a configuration against joint 4's limit where "hand up"
        # came out as "mast down": the hand pressed into the floor, propped
        # the robot off its wheels, and it fell (floor picks 20/24 vs 23/24).
        # Straight lines are for moving TOWARD things; away from a surface a
        # curved path does no harm.
        self.arm.track(p, self.plan.grasp_mat, dt)
        self._cmd = None
        if self.arm.at(p, 0.02) or self._elapsed(t) > 6.0:
            if self.gripper.pinched_body() >= 0:
                self._enter("stow", t)
            else:
                self._retry("dropped it while lifting", t)

    def _stow(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p, mat = self._stow_target()
        self._track(p, mat, dt)
        if self.arm.at(p, 0.03) or self._elapsed(t) > 6.0:
            self._enter("verify", t)

    def _verify(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p, mat = self._stow_target()
        self._track(p, mat, dt)
        if self.gripper.pinched_body() < 0:
            self._retry("dropped it while stowing", t)
        elif self._elapsed(t) > 1.0:
            self._manipulating(False)
            self._enter("done", t, f"holding the {self.spec.name}")

    def _backoff(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self._track(self._backoff_to, self.plan.grasp_mat, dt)
        if self.arm.at(self._backoff_to, 0.02) or self._elapsed(t) > 3.0:
            self._enter("plan", t)

    def _done(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p, mat = self._stow_target()
        self._track(p, mat, dt)

    def _failed(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        for a in self.arms.values():
            a.hold()
