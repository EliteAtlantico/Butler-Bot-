"""Movement algorithms for the simulated BracketBot.

An algorithm is any callable `f(bot, t)` that calls `bot.drive(v, w)`. It is
invoked once per physics step from `BracketBot.step(..., controller=f)`, and it
sits on top of the balance loop -- it asks for a velocity, the LQR works out
how to stay upright while delivering it.

    bot.step(30.0, controller=WaypointFollower([(2, 0), (2, 2)]))

`Sequence` chains them; `ObstacleAvoider` is the one that actually uses the
depth camera.
"""
from __future__ import annotations

import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Stand:
    """Do nothing but stay upright."""

    def __call__(self, bot, t):
        bot.drive(0.0, 0.0)

    @property
    def done(self):
        return False


class _Timed:
    """Base for algorithms that finish after a while."""

    def __init__(self, duration=np.inf):
        self.duration = duration
        self.t0 = None
        self._t = 0.0

    def _tick(self, t):
        if self.t0 is None:
            self.t0 = t
        self._t = t
        return t - self.t0

    @property
    def done(self):
        return self.t0 is not None and (self._t - self.t0) >= self.duration


class Drive(_Timed):
    """Hold a constant (v, w) for a fixed time, then stop."""

    def __init__(self, v=0.0, w=0.0, duration=3.0):
        super().__init__(duration)
        self.v, self.w = v, w

    def __call__(self, bot, t):
        self._tick(t)
        bot.drive(0.0, 0.0) if self.done else bot.drive(self.v, self.w)


class WaypointFollower:
    """Go-to-goal: turn toward the next waypoint, drive to it, repeat.

    Uses ground-truth pose. Swap `bot.position`/`bot.yaw` for an estimator if
    you want to make the localisation problem honest.
    """

    def __init__(self, waypoints, v_max=0.45, w_max=1.0, tol=0.18,
                 heading_gain=1.6, loop=False):
        self.waypoints = [np.asarray(p, float)[:2] for p in waypoints]
        self.v_max, self.w_max, self.tol = v_max, w_max, tol
        self.heading_gain = heading_gain
        self.loop = loop
        self.i = 0

    @property
    def done(self):
        return self.i >= len(self.waypoints)

    def __call__(self, bot, t):
        if self.done:
            bot.drive(0.0, 0.0)
            return
        goal = self.waypoints[self.i]
        here = bot.position[:2]
        delta = goal - here
        dist = float(np.linalg.norm(delta))
        if dist < self.tol:
            self.i += 1
            if self.loop and self.i >= len(self.waypoints):
                self.i = 0
            bot.drive(0.0, 0.0)
            return
        heading_err = _wrap(float(np.arctan2(delta[1], delta[0])) - bot.yaw)
        w = float(np.clip(self.heading_gain * heading_err, -self.w_max, self.w_max))
        # Slow down for big heading errors and when closing on the waypoint, so
        # the robot turns first instead of driving a wide arc past the goal.
        v = self.v_max * max(0.0, np.cos(heading_err)) * min(1.0, dist / 0.6)
        bot.drive(v, w)


class ObstacleAvoider:
    """Drive forward, steer away from whatever the depth camera sees.

    The head camera looks slightly down-range, so the floor fills the bottom of
    the frame; only the middle band is used, and anything nearer than
    `clear_range` counts as an obstacle. Steering is the difference between the
    nearest return on each side -- turn toward the open side.
    """

    def __init__(self, v_cruise=0.35, w_max=1.1, clear_range=2.2,
                 stop_range=1.1, width=64, height=48, band=(0.15, 0.7),
                 period=0.1):
        self.v_cruise, self.w_max = v_cruise, w_max
        self.clear_range, self.stop_range = clear_range, stop_range
        self.width, self.height, self.band = width, height, band
        self.period = period
        self._next = 0.0
        self._cmd = (0.0, 0.0)
        self.last_depth = None

    @property
    def done(self):
        return False

    def sense(self, bot):
        d = bot.depth("head_depth", self.width, self.height,
                      max_range=self.clear_range * 2)
        self.last_depth = d
        lo = int(self.band[0] * self.height)
        hi = int(self.band[1] * self.height)
        band = d[lo:hi]
        half = self.width // 2
        # Verified against a known-bearing obstacle: the image-left half is the
        # robot's left. (The camera's x axis points to the robot's right, which
        # makes it tempting to assume the opposite.)
        near = lambda a: float(np.nanmin(np.where(np.isfinite(a), a, np.inf)))
        return near(band[:, :half]), near(band[:, half:])   # (left, right)

    def __call__(self, bot, t):
        if t >= self._next:
            self._next = t + self.period
            d_left, d_right = self.sense(bot)
            closest = min(d_left, d_right)
            if closest < self.stop_range:
                # too close to drive past: rotate in place toward open space
                v = 0.0
                w = self.w_max * (1.0 if d_left > d_right else -1.0)
            else:
                urgency = np.clip(
                    (self.clear_range - closest) / max(self.clear_range - self.stop_range, 1e-6),
                    0.0, 1.0)
                v = self.v_cruise * (1.0 - 0.7 * urgency)
                bias = np.tanh((d_left - d_right) * 0.8)
                w = self.w_max * urgency * bias
            self._cmd = (float(v), float(w))
        bot.drive(*self._cmd)


class Sequence:
    """Run algorithms one after another, advancing when each reports done."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.i = 0

    @property
    def done(self):
        return self.i >= len(self.steps)

    @property
    def current(self):
        return None if self.done else self.steps[self.i]

    def __call__(self, bot, t):
        if self.done:
            bot.drive(0.0, 0.0)
            return
        step = self.steps[self.i]
        step(bot, t)
        if getattr(step, "done", False):
            self.i += 1


def square_patrol(side=1.6, **kw):
    """A closed square through the open floor, as a WaypointFollower."""
    return WaypointFollower(
        [(side, 0.0), (side, side), (0.0, side), (0.0, 0.0)], loop=True, **kw)
