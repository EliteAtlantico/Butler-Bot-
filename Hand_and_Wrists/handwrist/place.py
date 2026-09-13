"""Place: put down what a finished Pick is holding, at a named place.

    pick = Pick(bot, "keys", estimator=CameraEstimator())
    ...run until pick.done...
    give = Place(bot, pick, "person")        # a hand-over: onto the person's palm
    ...run until give.done...

Phases: backup -> plan -> approach -> settle -> over -> lower -> release ->
retreat -> home -> done.

It starts by reversing straight back BACKUP m. A pick leaves the base up
against whatever the item was on -- the balance loop creeps ~16 cm toward
the table while the arm is out -- and turning on the spot from there swings
the mast into the tabletop (it fell) or jams the hull on a table leg (it
stalled for 20 s and the item slipped out). It came in along that line, so
straight back is clear. It reuses the Pick's arm controllers (the held arm's joint
commands live there) and its grasp planner (reach limits and IK checks), and
keeps the hand's pick-up orientation relative to the base, so the item goes
down the way it came up.
"""
from __future__ import annotations

import numpy as np

from bracketbot_sim.kinematics import rot_z

from .places import PlacePlanner, resolve_place
from .skills import DESCEND_SPEED, ApproachPose, ArmSkill


class Place(ArmSkill):
    TAG = "place"
    PHASES = ("backup", "plan", "approach", "settle", "over", "lower", "release",
              "retreat", "home", "done", "failed")
    BACKUP = 0.45     # m: clears the mast and the stowed hand's swing

    def __init__(self, bot, holding, where, verbose=True, arms=None, keep=()):
        spec = resolve_place(where)      # a PLACES name, or a PlaceSpec found at run time
        if not holding.succeeded:
            raise ValueError("nothing to place: the pick did not succeed")
        item = holding.spec.name
        super().__init__(bot, arms or holding.arms, holding.grippers,
                         f"put the {item} {spec.preposition} the {spec.name}",
                         holding.control_period, verbose, keep=keep)
        self.item, self.where, self.place_spec = item, spec.name, spec
        self.side = holding.plan.side
        self.kind = holding.plan.kind
        self.rel_mat = rot_z(-holding.plan.base_yaw) @ holding.plan.grasp_mat
        self.grip_above_bottom = max(
            float(holding.plan.grasp_pos[2] - holding.estimate.bottom_z), 0.0)
        self.open_gap = holding.plan.open_gap
        self.planner = PlacePlanner(holding.planner)
        self.target = None
        self._replans = 0
        self._lost_ticks = 0
        self._backup_job = None
        self.phase = "backup"

    @property
    def arm(self):
        return self.arms[self.side]

    @property
    def gripper(self):
        return self.grippers[self.side]

    def _still_holding(self):
        """Pad contact flickers on a bump; call it dropped after 0.2 s."""
        self._lost_ticks = 0 if self.gripper.pinched_body() >= 0 else self._lost_ticks + 1
        return self._lost_ticks < 10

    # --------------------------------------------------------------- phases
    def _backup(self, bot, t, dt):
        self.arm.hold()
        if self._backup_job is None:
            self._manipulating(False)
            h = np.array([np.cos(bot.yaw), np.sin(bot.yaw)])
            goal = bot.position[:2] - h * self.BACKUP
            self._backup_job = (goal, ApproachPose(goal, bot.yaw))
        goal, mover = self._backup_job
        if not self._still_holding():
            self._fail(f"dropped the {self.item} backing away", t)
        elif mover._drive(bot, goal, mover.goal_yaw, tol=0.06) or self._elapsed(t) > 10.0:
            bot.drive(0.0, 0.0)
            self._enter("plan", t, "backed away")

    def _plan(self, bot, t, dt):
        self.arm.hold()
        targets = self.planner.plan(self.place_spec, self.side, self.rel_mat,
                                    self.grip_above_bottom, self.kind)
        if not targets:
            self._fail(f"can't reach the {self.where}", t)
            return
        self.target = targets[0]
        self.approach = ApproachPose(self.target.base_xy, self.target.base_yaw)
        self._manipulating(False)
        self._enter("approach", t, self.target.describe())

    def _approach(self, bot, t, dt):
        self.arm.hold()
        self._cmd = None
        if not self._still_holding():
            self._fail(f"dropped the {self.item} on the way", t)
        elif self.approach(bot):
            self._manipulating(True)
            self._enter("settle", t)
        elif self._elapsed(t) > 40.0:
            self._fail(f"could not reach the {self.where}", t)

    def _settle(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self.arm.hold()
        if (self._elapsed(t) > 0.8 and self._still()) or self._elapsed(t) > 6.0:
            if self.planner.reachable_from_here(self.target):
                self._cmd = None
                self._enter("over", t)
            elif self._replans < 2:
                self._replans += 1
                self._enter("plan", t, "parked out of reach, re-planning")
            else:
                self._fail(f"parked out of reach of the {self.where}", t)

    def _over(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        p = self.target.above
        self._track(p, self.target.mat, dt)
        if not self._still_holding():
            self._fail(f"dropped the {self.item} reaching over", t)
        elif self.arm.at(p, 0.02) and self._still(0.05):
            self._enter("lower", t)
        elif self._elapsed(t) > 14.0:
            if self.arm.at(p, 0.05):
                self._enter("lower", t, "(hover loose)")
            else:
                self._fail(f"could not get over the {self.where}", t)

    def _item_touching(self):
        """Is the held item resting on something other than the fingers?"""
        m, d = self.bot.model, self.bot.data
        held = self.gripper.pinched_body()
        if held < 0:
            return False
        robot = self.gripper.robot_bodies
        for i in range(d.ncon):
            b1 = m.geom_bodyid[d.contact[i].geom1]
            b2 = m.geom_bodyid[d.contact[i].geom2]
            if held in (b1, b2):
                other = b2 if b1 == held else b1
                if other not in robot:
                    return True
        return False

    def _lower(self, bot, t, dt):
        """Lower until the planned height OR the item touches down, whichever
        comes first. Pushing on after touchdown props the whole robot up on
        the item -- the base spun out and fell setting a box on a hand."""
        bot.drive(0.0, 0.0)
        p = self.target.point
        self._track(p, self.target.mat, dt, DESCEND_SPEED)
        if self._item_touching():
            self._cmd = (self.arm.grasp_pose[0], self.target.mat)   # stop here
            self._enter("release", t, "touched down")
        elif (self.arm.at(p, 0.012) and self._still(0.05)) or self._elapsed(t) > 6.0:
            self._enter("release", t)

    def _release(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self._track(self._cmd[0] if self._cmd is not None else self.target.point,
                    self.target.mat, dt, DESCEND_SPEED)      # hold where it stopped
        if self._elapsed(t) < dt * 1.5:
            self.arm.set_gripper(self.gripper.q_for_gap(self.open_gap))
        if self._elapsed(t) > 2.0:
            self.arm.set_gripper(1.0)            # stuck to a pad: open fully
        if (self._elapsed(t) > 0.6 and self.gripper.pinched_body() < 0) \
                or self._elapsed(t) > 3.0:
            if self.place_spec.mode == "drop":
                PlacePlanner.dropped(bot, self.place_spec)
            self._retreat_to = self.arm.grasp_pose[0] + np.array([0.0, 0.0, 0.12])
            self._enter("retreat", t, f"let go of the {self.item}")

    def _retreat(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self._track(self._retreat_to, self.target.mat, dt)
        if self.arm.at(self._retreat_to, 0.03) or self._elapsed(t) > 4.0:
            self._enter("home", t)

    def _home(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        if self._arms_home(dt):
            self._manipulating(False)
            prep = self.place_spec.preposition
            self._enter("done", t, f"the {self.item} is {prep} the {self.where}"
                        if prep != "to" else f"handed the {self.item} over")

    def _done(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        self._arms_home(dt)

    def _failed(self, bot, t, dt):
        bot.drive(0.0, 0.0)
        for a in self.arms.values():
            a.hold()
