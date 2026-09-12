"""Closed-loop visual navigation: RGB-D -> detections + map -> A* -> drive.

    bot.step(60.0, controller=VisualNavigator())

The robot spins once to build a map, picks out the red column with the colour
detector, plans a path around whatever the depth camera has put in the grid,
and drives it -- replanning as new obstacles come into view.

Only the camera decides *where to go*. Pose still comes from `bot.position`
and `bot.yaw`, i.e. ground truth: mapping and planning are solved here,
localisation is not. Swapping those two reads for an estimator is the honest
next step and would not change anything else in this file.
"""
from __future__ import annotations

import numpy as np

from . import occupancy, perception, planning


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class VisualNavigator:
    """Search -> detect -> map -> plan -> follow, driven by the RGB-D camera."""

    SCAN, NAVIGATE, ARRIVED, STUCK = "scan", "navigate", "arrived", "stuck"

    def __init__(self, goal_label: str = perception.GOAL_LABEL,
                 scan_rate: float = 0.7, scan_turns: float = 1.0,
                 sense_period: float = 0.2, plan_period: float = 1.0,
                 stop_distance: float = 1.0, v_max: float = 0.42,
                 w_max: float = 1.0, waypoint_tol: float = 0.22,
                 heading_gain: float = 1.5, width: int = 320, height: int = 240,
                 max_range: float = 12.0, robot_radius: float = 0.30,
                 re_engage_margin: float = 0.75,
                 grid: occupancy.OccupancyGrid | None = None,
                 detector=None, verbose: bool = False):
        # `detector(obs, robot_yaw=...) -> list[Detection]`. Defaults to the
        # colour detector; a YoloDetector satisfies the same contract, so
        # nothing below this line changes when you swap them.
        self.detector = detector if detector is not None else perception.detect
        self.goal_label = goal_label
        self.scan_rate, self.scan_turns = scan_rate, scan_turns
        self.sense_period, self.plan_period = sense_period, plan_period
        self.stop_distance = stop_distance
        self.v_max, self.w_max = v_max, w_max
        self.waypoint_tol, self.heading_gain = waypoint_tol, heading_gain
        self.width, self.height, self.max_range = width, height, max_range
        self.robot_radius = robot_radius
        self.re_engage_margin = re_engage_margin
        self.grid = grid if grid is not None else occupancy.OccupancyGrid()
        self.verbose = verbose

        self.state = self.SCAN
        self.detections: list[perception.Detection] = []
        self.goal_xy: np.ndarray | None = None
        self.path: list[np.ndarray] = []
        self.obs: perception.Observation | None = None
        self.blocked: np.ndarray | None = None
        self.log: list[str] = []
        # Richest frame of the run, kept for diagnostics: the last frame is
        # usually nose-up against the goal and shows almost nothing.
        self.best_obs: perception.Observation | None = None
        self.best_detections: list[perception.Detection] = []
        self.best_time = 0.0

        self._scan_start = None
        self._scan_yaw = 0.0
        self._last_yaw = None
        self._next_sense = 0.0
        self._next_plan = 0.0
        self._last_t = None
        self._wp = 0
        self._cmd = (0.0, 0.0)
        self._plan_failures = 0

    @property
    def done(self):
        return self.state in (self.ARRIVED, self.STUCK)

    # ------------------------------------------------------------- sensing
    def sense(self, bot, t: float = 0.0):
        self.obs = perception.observe(bot, width=self.width, height=self.height,
                                      max_range=self.max_range)
        # Pass sim time through so time-gated detectors (e.g. the local LLM)
        # pace themselves on the sim clock, not the wall clock.
        self.detections = self.detector(self.obs, robot_yaw=bot.yaw, t=t)
        if len(self.detections) >= len(self.best_detections):
            self.best_obs, self.best_detections, self.best_time = (
                self.obs, list(self.detections), t)

        hits = perception.obstacle_points(self.obs)
        free = perception.floor_points(self.obs)
        self.grid.integrate(self.obs.cam_pos[:2], hits, free)

        goals = [d for d in self.detections if d.label == self.goal_label]
        if goals:
            seen = max(goals, key=lambda d: d.pixels)
            # The detector sees the near face; the column centre is behind it.
            # Nudging along the view ray keeps the goal off the surface, which
            # otherwise plans a path into the obstacle it is standing on.
            ray = seen.position[:2] - self.obs.cam_pos[:2]
            ray /= max(np.linalg.norm(ray), 1e-6)
            estimate = seen.position[:2] + ray * 0.2
            self.goal_xy = (estimate if self.goal_xy is None
                            else 0.7 * self.goal_xy + 0.3 * estimate)

    # ------------------------------------------------------------ planning
    def replan(self, bot):
        if self.goal_xy is None:
            return False
        blocked, soft = self.grid.costmap(robot_radius=self.robot_radius)
        # Stand off from the goal instead of driving into it.
        here = bot.position[:2]
        delta = self.goal_xy - here
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return False
        stand_off = self.goal_xy - delta / dist * self.stop_distance

        start = self.grid.to_cell(here)[0]
        goal = self.grid.to_cell(stand_off)[0]
        if blocked[tuple(start)]:
            # The robot is inside its own inflation (hugging a wall). Free the
            # cells it physically occupies, or no plan can ever start.
            blocked = blocked.copy()
            sx, sy = start
            blocked[max(sx - 2, 0):sx + 3, max(sy - 2, 0):sy + 3] = False

        cells = planning.astar(blocked, start, goal, soft=soft)
        if cells is None:
            self._plan_failures += 1
            if self._plan_failures >= 8:
                self.state = self.STUCK
            return False
        self._plan_failures = 0
        cells = planning.shortcut(blocked, cells)
        self.path = [self.grid.to_world(c)[0] for c in cells[1:]] or [stand_off]
        self.blocked = blocked
        self._wp = 0
        return True

    # ------------------------------------------------------------- recovery
    def _displaced(self, bot):
        """Has something moved the robot away from where it settled?

        A viewer reset, a shove, a wheel slip and a localisation jump all look
        identical from here: the goal is suddenly much further off than the
        arrival threshold. Without this check ARRIVED is terminal and the
        robot holds station for ever -- the kidnapped-robot failure. The
        margin is hysteresis, so sitting near the threshold cannot oscillate.
        """
        if self.goal_xy is None:
            return False
        gap = float(np.linalg.norm(self.goal_xy - bot.position[:2]))
        return gap > self.stop_distance + 0.15 + self.re_engage_margin

    # ------------------------------------------------------------ following
    def _follow(self, bot):
        here = bot.position[:2]
        while self._wp < len(self.path) - 1 and \
                np.linalg.norm(self.path[self._wp] - here) < self.waypoint_tol:
            self._wp += 1
        target = self.path[self._wp]
        delta = target - here
        dist = float(np.linalg.norm(delta))
        err = _wrap(float(np.arctan2(delta[1], delta[0])) - bot.yaw)
        w = float(np.clip(self.heading_gain * err, -self.w_max, self.w_max))
        # Turn before translating: a balancing two-wheeler that tries to do
        # both at once on a big heading error carves a wide arc into whatever
        # it was trying to avoid.
        v = self.v_max * max(0.0, np.cos(err)) * min(1.0, dist / 0.5)
        return v, w

    # ----------------------------------------------------------------- tick
    def __call__(self, bot, t):
        # A sim reset (the viewer's R key, or BracketBot.reset) zeros the
        # clock, so t jumps backwards below the scheduled times and this
        # navigator would never sense or replan again -- the same trap the
        # ObstacleAvoider hit. Re-seed both clocks when time runs backwards.
        #
        # "Runs backwards" means earlier than the PREVIOUS tick. Testing
        # `t < self._next_*` is true whenever the next event is in the future,
        # which is almost every step: it sensed on every physics step and
        # replanned A* far more often than plan_period, making runs ~100x
        # slower than real time.
        if self._last_t is not None and t < self._last_t:
            self._next_sense = t
            self._next_plan = t
        self._last_t = t

        if t >= self._next_sense:
            self._next_sense = t + self.sense_period
            self.sense(bot, t)

        if self.done and self._displaced(bot):
            self._note(f"t={t:.1f}s displaced to "
                       f"({bot.position[0]:.2f}, {bot.position[1]:.2f}), re-engaging")
            self.state = self.NAVIGATE
            self._plan_failures = 0
            self._next_plan = 0.0
            self._wp = 0

        if self.state == self.SCAN:
            if self._scan_start is None:
                self._scan_start, self._last_yaw = t, bot.yaw
            self._scan_yaw += abs(_wrap(bot.yaw - self._last_yaw))
            self._last_yaw = bot.yaw
            turned = self._scan_yaw >= 2 * np.pi * self.scan_turns
            if turned and self.goal_xy is not None:
                if self.replan(bot):
                    self._note(f"t={t:.1f}s scan done, goal at "
                               f"({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}), "
                               f"{len(self.path)} waypoints")
                    self.state = self.NAVIGATE
            elif turned:
                self._scan_yaw = 0.0      # nothing found, go round again
            bot.drive(0.0, self.scan_rate)
            return

        if self.state == self.NAVIGATE:
            if self.goal_xy is not None and \
                    np.linalg.norm(self.goal_xy - bot.position[:2]) < self.stop_distance + 0.15:
                self.state = self.ARRIVED
                self._note(f"t={t:.1f}s arrived, "
                           f"{np.linalg.norm(self.goal_xy - bot.position[:2]):.2f}m from goal")
                bot.drive(0.0, 0.0)
                return
            if t >= self._next_plan:
                self._next_plan = t + self.plan_period
                self.replan(bot)
            if self.path:
                self._cmd = self._follow(bot)
            bot.drive(*self._cmd)
            return

        bot.drive(0.0, 0.0)

    def _note(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)
