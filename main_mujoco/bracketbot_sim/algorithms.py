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
        # A sim reset (the viewer's 'r' key, or BracketBot.reset) zeros the
        # clock, so t can jump backwards below _next. Re-seed the perception
        # clock in that case, or we'd sit on the last command with the depth
        # camera switched off and drive straight into the next obstacle.
        if t < self._next:
            self._next = t
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


class LocalMap:
    """A short-term memory of where obstacles were seen, in world coordinates.

    The head camera sits 1.54 m up and is tilted down, so an obstacle shorter
    than the robot slides out of the bottom of the frame as it gets close --
    at half a metre a 0.9 m pillar is completely invisible. Steering only on
    the current frame therefore drives confidently into things it saw clearly
    two seconds ago. Points are voxelised, stamped, and forgotten after
    `max_age` so the map cannot fossilise a stale reading.
    """

    def __init__(self, cell=0.08, max_age=60.0, radius=4.0,
                 z_min=0.10, z_max=1.45, self_radius=0.36):
        self.cell, self.max_age, self.radius = cell, max_age, radius
        self.z_min, self.z_max = z_min, z_max
        # A camera looking down from the mast sees the robot's own arms and
        # base. Without this the map is permanently full of obstacles 0.15 m
        # ahead and the robot refuses to move. Real stacks self-filter the same
        # way, using the known kinematics.
        self.self_radius = self_radius
        self.cells: dict[tuple, float] = {}

    def update(self, bot, t, camera="head_depth", width=64, height=48,
               max_range=3.5):
        pts = bot.point_cloud_world(camera, width, height, max_range)
        if len(pts):
            keep = (pts[:, 2] > self.z_min) & (pts[:, 2] < self.z_max)
            keep &= (np.linalg.norm(pts[:, :2] - bot.position[:2], axis=1)
                     > self.self_radius)
            for x, y in pts[keep][:, :2]:
                self.cells[(round(float(x) / self.cell),
                            round(float(y) / self.cell))] = t
        self._prune(bot, t)

    def _prune(self, bot, t, clear_margin=0.3, cam_half_fov=np.deg2rad(34.0)):
        """Forget cells that are out of range, ancient, or demonstrably empty.

        Ageing alone is not enough and is actively harmful: an obstacle the
        robot is pressed against cannot be re-observed (too close, below the
        tilted camera's view), so a short timeout makes the robot forget the
        very thing blocking it and drive into it forever. Instead, a cell is
        cleared only when the camera is looking through where it should be and
        sees something further away.
        """
        here = bot.position[:2]
        if not self.cells:
            return
        keys = list(self.cells)
        P = np.array(keys, float) * self.cell
        rel = P - here
        rng = np.linalg.norm(rel, axis=1)
        ang = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - bot.yaw)

        keep_range = rng < self.radius
        fresh = np.array([t - self.cells[k] < self.max_age for k in keys])

        # ray clearing: only for cells the camera can actually see through
        _, obs = self.scan(bot, n_bins=41, half_fov=cam_half_fov)
        edges = np.linspace(-cam_half_fov, cam_half_fov, 42)
        idx = np.digitize(ang, edges) - 1
        visible = (np.abs(ang) < cam_half_fov) & (rng > self.self_radius + 0.1)
        cleared = np.zeros(len(keys), bool)
        ok = visible & (idx >= 0) & (idx < 41)
        cleared[ok] = obs[idx[ok]] > rng[ok] + clear_margin

        alive = keep_range & fresh & ~cleared
        self.cells = {k: self.cells[k] for k, a in zip(keys, alive) if a}

    def points(self):
        if not self.cells:
            return np.zeros((0, 2))
        return np.array([[k[0], k[1]] for k in self.cells]) * self.cell

    def scan(self, bot, n_bins=41, half_fov=np.deg2rad(100.0), max_range=3.5):
        """Nearest remembered obstacle per bearing bin, in the robot frame."""
        bearings = np.linspace(-half_fov, half_fov, n_bins)
        ranges = np.full(n_bins, np.inf)
        P = self.points()
        if len(P) == 0:
            return bearings, ranges
        rel = P - bot.position[:2]
        rng = np.linalg.norm(rel, axis=1)
        ang = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - bot.yaw)
        edges = np.linspace(-half_fov, half_fov, n_bins + 1)
        idx = np.digitize(ang, edges) - 1
        ok = (idx >= 0) & (idx < n_bins) & (rng < max_range)
        np.minimum.at(ranges, idx[ok], rng[ok])
        return bearings, ranges


class NavigateTo:
    """Go to a goal pose, steering around whatever the depth camera sees.

    Vector-field-histogram style: score every candidate heading in the camera's
    field of view against both the goal direction and the measured clearance,
    then steer at the cheapest one. A left/right "which side is more open"
    rule is simpler but deadlocks on an obstacle dead ahead -- the two halves
    read the same, the bias comes out zero, and the robot drives straight into
    it.

    Inside `arrive_radius` obstacle costs are switched off, or the robot would
    refuse to approach the very table it was sent to.
    """

    def __init__(self, goal_xy, goal_yaw=None, standoff_tol=0.06,
                 yaw_tol=0.05, v_max=0.4, w_max=1.1, arrive_radius=0.45,
                 safe_range=1.25, stop_range=0.42, robot_half_width=0.24,
                 lookahead=1.2, n_candidates=81, max_scan=np.deg2rad(100.0),
                 heading_gain=1.9, commit_gain=0.35, period=0.1):
        self.goal = np.asarray(goal_xy, float)[:2]
        self.goal_yaw = goal_yaw
        self.standoff_tol, self.yaw_tol = standoff_tol, yaw_tol
        self.v_max, self.w_max = v_max, w_max
        self.arrive_radius = arrive_radius
        self.safe_range, self.stop_range = safe_range, stop_range
        self.robot_half_width = robot_half_width
        self.lookahead = lookahead
        self.n_candidates, self.max_scan = n_candidates, max_scan
        self.commit_gain = commit_gain
        self.heading_gain = heading_gain
        self._goal_blocked = False
        self.period = period
        self._next = 0.0
        self._steer = 0.0
        self._ahead = np.inf
        self.done = False
        self.map = LocalMap()
        self._stuck_since = None
        self.stage = "cruise"
        self.k_rho, self.k_alpha, self.k_beta = 0.55, 1.5, -0.55
        self.entry_offset = 0.45
        self.entry_tol = 0.13
        self.recruise_radius = 1.1

    def _plan(self, bot, goal_bearing, t=0.0):
        """Pick a heading by testing corridors, not by balancing gradients.

        For every candidate heading, sweep the robot's own width forward and
        find how far it gets before hitting a mapped point. Steer at the
        heading closest to the goal that is clear for the full lookahead.

        A weighted sum of "pull toward goal" and "push from obstacles" -- the
        obvious potential-field formulation -- is what this replaces. It hovers:
        in front of an obstacle the two terms balance, the robot creeps forward
        as the goal term wins slightly, and it ends up nose-against the thing it
        was avoiding. Asking "which way can I actually go?" gives a decisive
        answer at every step.
        """
        self.map.update(bot, t)
        bearings = np.linspace(-self.max_scan, self.max_scan, self.n_candidates)
        P = self.map.points()

        if len(P) == 0:
            clearance = np.full(self.n_candidates, self.lookahead)
        else:
            rel = P - bot.position[:2]
            r = np.linalg.norm(rel, axis=1)
            a = _wrap(np.arctan2(rel[:, 1], rel[:, 0]) - bot.yaw)
            d = _wrap(a[None, :] - bearings[:, None])
            fwd = r[None, :] * np.cos(d)
            lat = r[None, :] * np.sin(d)
            hit = (fwd > 0.0) & (fwd < self.lookahead) & \
                  (np.abs(lat) < self.robot_half_width)
            blocked = np.where(hit, fwd, np.inf)
            clearance = np.minimum(blocked.min(axis=1), self.lookahead)

        free = clearance >= self.lookahead - 1e-6
        cost = (np.abs(_wrap(bearings - goal_bearing))
                + self.commit_gain * np.abs(_wrap(bearings - self._steer)))
        if free.any():
            idx = int(np.argmin(np.where(free, cost, np.inf)))
        else:
            # nothing is clear all the way: take the roomiest direction,
            # still biased toward the goal and toward staying committed
            idx = int(np.argmin(-clearance / self.lookahead + 0.35 * cost))
        self._steer = float(bearings[idx])
        self._ahead = float(clearance[idx])
        self._goal_blocked = not free[int(np.argmin(np.abs(_wrap(bearings - goal_bearing))))]

    def _entry_point(self):
        """A point short of the goal, on the goal heading, to line up from."""
        d = np.array([np.cos(self.goal_yaw), np.sin(self.goal_yaw)])
        return self.goal - d * self.entry_offset

    def __call__(self, bot, t):
        here = bot.position[:2]
        heading = np.array([np.cos(bot.yaw), np.sin(bot.yaw)])

        if self.goal_yaw is None:
            delta = self.goal - here
            dist = float(np.linalg.norm(delta))
            if dist < self.standoff_tol:
                self.done = True
                bot.drive(0.0, 0.0)
                return
            self._cruise(bot, t, delta, dist)
            return

        delta = self.goal - here
        yaw_err = _wrap(self.goal_yaw - bot.yaw)

        # Stage 1: drive to an entry point behind the goal, avoiding obstacles.
        # Heading control alone cannot fix a sideways offset -- a differential
        # drive has no sideways -- so stopping when the FORWARD error is small
        # parks the robot happily half a metre off to one side. Lining up from
        # behind makes the last leg a straight run.
        if self.stage == "cruise":
            entry = self._entry_point()
            d = entry - here
            n = float(np.linalg.norm(d))
            if n < self.entry_tol:
                self.stage = "turn"
            else:
                self._cruise(bot, t, d, n)
                return

        # Stage 2: a pose controller for the last half metre.
        #
        # Decoupled "turn to the heading, then drive along it" does not
        # converge here: a differential drive cannot correct a sideways offset
        # while holding its heading, and a balancing base lags far enough
        # behind a proportional velocity command to limit-cycle around the
        # goal. The polar (rho, alpha, beta) controller drives position AND
        # orientation to zero together.
        rho = float(np.linalg.norm(delta))
        if rho < self.standoff_tol and abs(yaw_err) < self.yaw_tol \
                and bot.ground_speed < 0.06:
            self.done = True
            bot.drive(0.0, 0.0)
            return
        # Only fall back to obstacle-aware cruising if we have been pushed well
        # away from the goal. A lateral-error test here looks sensible and
        # deadlocks: the entry point is already reached, so cruise immediately
        # hands back to this stage, which immediately aborts again, and the
        # robot sits still forever. The polar controller converges from any
        # pose, which is the whole reason to use it.
        if rho > self.recruise_radius:
            self.stage = "cruise"
            bot.drive(0.0, 0.0)
            return

        if rho < self.standoff_tol:
            # on the spot: only the heading is left to fix
            bot.drive(0.0, float(np.clip(1.5 * yaw_err, -0.4, 0.4)))
            return

        alpha = _wrap(float(np.arctan2(delta[1], delta[0])) - bot.yaw)
        reverse = abs(alpha) > np.pi / 2
        if reverse:
            alpha = _wrap(alpha - np.pi)
        beta = _wrap(yaw_err - alpha)
        v = self.k_rho * rho * float(np.cos(alpha))
        w = self.k_alpha * alpha + self.k_beta * beta
        if reverse:
            v = -v
        bot.drive(float(np.clip(v, -0.18, 0.18)),
                  float(np.clip(w, -0.6, 0.6)))

    def _cruise(self, bot, t, delta, dist):
        goal_bearing = _wrap(float(np.arctan2(delta[1], delta[0])) - bot.yaw)
        if t >= self._next:
            self._next = t + self.period
            self._plan(bot, goal_bearing, t)

        w = float(np.clip(self.heading_gain * self._steer, -self.w_max, self.w_max))
        v = self.v_max * max(0.0, np.cos(self._steer)) * min(1.0, dist / 0.5)
        if self._ahead < self.stop_range:
            v = 0.0            # nose to nose with something: turn on the spot
        elif self._ahead < self.safe_range:
            v *= float(np.clip((self._ahead - self.stop_range)
                               / (self.safe_range - self.stop_range), 0.15, 1.0))

        # Recovery. "Stuck" is commanding motion and not getting it, which is
        # what being wedged against something looks like. Testing the COMMAND
        # for non-zero, or the balance loop's odometry (which reads non-zero
        # from the wheels slipping), never fires while the robot pushes
        # uselessly into a pillar at full throttle.
        if abs(v) <= 0.05 or bot.ground_speed > 0.05:
            self._stuck_since = None
        elif self._stuck_since is None:
            self._stuck_since = t
        if self._stuck_since is not None and t - self._stuck_since > 1.2:
            v = -0.22
            w = self.w_max * (1.0 if self._steer >= 0 else -1.0)
            if t - self._stuck_since > 3.5:
                self._stuck_since = None
        bot.drive(float(v), w)
