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
from .scene import SceneInfo


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class VisualNavigator:
    """Search -> detect -> map -> plan -> follow, driven by the RGB-D camera."""

    SURVEY, SCAN, NAVIGATE, ARRIVED, STUCK = ("survey", "scan", "navigate",
                                             "arrived", "stuck")

    @classmethod
    def for_bot(cls, bot, goal=None, camera: str | None = None, **kw):
        """Build a navigator configured from the scene the robot is in.

        Reads the world extent, floor height, footprint radius and camera off
        the model, so the same call works in a scene this code has never seen.
        `goal` may be an (x, y) world coordinate -- which needs no detector
        and no colour palette -- or a class label to look for.
        """
        info = SceneInfo.from_model(bot.model, bot.data, camera=camera)
        return cls(scene=info, goal=goal, **kw)

    def __init__(self, goal=None, scene: SceneInfo | None = None,
                 goal_label: str = perception.GOAL_LABEL,
                 scan_rate: float = 0.7, scan_turns: float = 1.0,
                 sense_period: float = 0.2, plan_period: float = 1.0,
                 stop_distance: float = 1.0, v_max: float = 0.42,
                 w_max: float = 1.0, waypoint_tol: float = 0.22,
                 heading_gain: float = 1.5, width: int = 320, height: int = 240,
                 max_range: float = 12.0, robot_radius: float | None = None,
                 re_engage_margin: float = 0.75,
                 grid: occupancy.OccupancyGrid | None = None,
                 resolution: float = 0.10,
                 survey=None, survey_turn_tol: float = 0.08,
                 survey_settle: float = 0.3, survey_turn_timeout: float = 6.0,
                 provisional_distance: float = 3.0,
                 detector=None, verbose: bool = False):
        # `detector(obs, robot_yaw=...) -> list[Detection]`. Defaults to the
        # colour detector; a YoloDetector satisfies the same contract, so
        # nothing below this line changes when you swap them.
        self.scene = scene
        self.detector = detector if detector is not None else perception.detect
        self.goal_label = goal_label
        self.scan_rate, self.scan_turns = scan_rate, scan_turns
        self.sense_period, self.plan_period = sense_period, plan_period
        self.stop_distance = stop_distance
        self.v_max, self.w_max = v_max, w_max
        self.waypoint_tol, self.heading_gain = waypoint_tol, heading_gain
        self.width, self.height, self.max_range = width, height, max_range
        self.re_engage_margin = re_engage_margin
        self.verbose = verbose

        # Everything below is measured from the scene when one was supplied,
        # and falls back to the values tuned for the original course when it
        # was not -- so existing callers keep their behaviour exactly.
        self.camera = scene.camera if scene else None
        self.floor_z = scene.floor_z if scene else 0.0
        # Anything above the robot's head is not an obstacle it can hit. A
        # doorway lintel or a low ceiling otherwise projects straight down
        # into the 2-D grid and seals a gap the robot can drive through.
        self.obstacle_ceiling = (scene.robot_top + 0.10) if scene else 2.0
        self.robot_radius = (robot_radius if robot_radius is not None
                             else (scene.robot_radius + 0.05 if scene else 0.30))
        self.self_radius = (scene.robot_radius + 0.25) if scene else 0.55
        if scene:
            self.max_range = min(max_range, scene.map_range * 1.5)
        self.grid = grid if grid is not None else (
            occupancy.OccupancyGrid.covering(scene.bounds, resolution) if scene
            else occupancy.OccupancyGrid())

        # A coordinate goal needs no detector and no colour palette; a label
        # goal has to be found first, which is what the scan is for.
        self.fixed_goal = None
        if goal is not None and not isinstance(goal, str):
            self.fixed_goal = np.asarray(goal, float)[:2]
        elif isinstance(goal, str):
            self.goal_label = goal

        # An LLM survey (vision_sim.llm_survey.LlmSurvey) replaces the spin
        # scan: photograph every direction first, then ask the model once. A
        # coordinate goal needs no finding, so it skips the survey entirely.
        self.survey = survey
        self.survey_turn_tol, self.survey_settle = survey_turn_tol, survey_settle
        self.survey_turn_timeout = survey_turn_timeout
        self.provisional_distance = provisional_distance
        self.survey_shots: list = []
        self.survey_result = None
        self._survey_plan: list[float] | None = None
        self._survey_idx = 0
        self._settle_since: float | None = None
        self._turn_since: float | None = None
        self._goal_provisional = False

        self.state = (self.SURVEY if (survey is not None and self.fixed_goal is None)
                      else self.SCAN)
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
        self.obs = perception.observe(bot, camera=self.camera, width=self.width,
                                      height=self.height, max_range=self.max_range)
        # Pass sim time through so time-gated detectors (e.g. the local LLM)
        # pace themselves on the sim clock, not the wall clock. During a survey
        # the model looks at every photo at once at the end, so querying the
        # per-frame detector while turning would only duplicate that, ~10 s a go.
        self.detections = ([] if self.state == self.SURVEY
                           else self.detector(self.obs, robot_yaw=bot.yaw, t=t))
        if len(self.detections) >= len(self.best_detections):
            self.best_obs, self.best_detections, self.best_time = (
                self.obs, list(self.detections), t)

        self._integrate(self.obs)

        if self.fixed_goal is not None:
            self.goal_xy = self.fixed_goal
            return

        goals = [d for d in self.detections if d.label == self.goal_label]
        if goals:
            seen = max(goals, key=lambda d: d.pixels)
            # The detector sees the near face; the column centre is behind it.
            # Nudging along the view ray keeps the goal off the surface, which
            # otherwise plans a path into the obstacle it is standing on.
            ray = seen.position[:2] - self.obs.cam_pos[:2]
            ray /= max(np.linalg.norm(ray), 1e-6)
            estimate = seen.position[:2] + ray * 0.2
            if self.goal_xy is None or self._goal_provisional:
                # A heading-only survey answer is a guess at the distance; the
                # first real sighting replaces it outright rather than being
                # averaged with it.
                if self._goal_provisional:
                    self._note(f"t={t:.1f}s goal sighted at ({estimate[0]:.2f}, "
                               f"{estimate[1]:.2f}), replacing the survey's "
                               "provisional goal")
                self.goal_xy = estimate
                self._goal_provisional = False
            else:
                self.goal_xy = 0.7 * self.goal_xy + 0.3 * estimate

    def _integrate(self, obs):
        """Fold one RGB-D frame into the occupancy grid."""
        hits = perception.obstacle_points(obs, self_radius=self.self_radius,
                                          floor_z=self.floor_z,
                                          max_height=self.obstacle_ceiling)
        free = perception.floor_points(obs, floor_z=self.floor_z)
        self.grid.integrate(obs.cam_pos[:2], hits, free)

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

    # -------------------------------------------------------------- survey
    def _survey_tick(self, bot, t):
        """Turn to each planned heading, hold still, photograph; then ask once."""
        from .llm_survey import SurveyShot

        if self._survey_plan is None:
            self._survey_plan = self.survey.headings(bot.yaw)
            self._turn_since = t
            self._note(f"t={t:.1f}s survey: {len(self._survey_plan)} photos from "
                       f"({bot.position[0]:.2f}, {bot.position[1]:.2f})")

        if self._survey_idx < len(self._survey_plan):
            target = self._survey_plan[self._survey_idx]
            err = _wrap(target - bot.yaw)
            spinning = abs(float(bot.state[5])) > 0.15
            timed_out = t - self._turn_since > self.survey_turn_timeout
            if (abs(err) > self.survey_turn_tol or spinning) and not timed_out:
                self._settle_since = None
                bot.drive(0.0, float(np.clip(self.heading_gain * err,
                                             -self.w_max, self.w_max)))
                return
            # On target: hold still long enough for the balancing base to stop
            # rocking, or the photo's recorded heading is not the one it shows.
            bot.drive(0.0, 0.0)
            if self._settle_since is None:
                self._settle_since = t
            if t - self._settle_since < self.survey_settle and not timed_out:
                return
            if timed_out:
                self._note(f"t={t:.1f}s survey: photo {self._survey_idx} taken "
                           f"{abs(np.degrees(err)):.0f} deg off target after "
                           f"{self.survey_turn_timeout:.0f}s")
            obs = perception.observe(bot, camera=self.camera, width=self.width,
                                     height=self.height, max_range=self.max_range)
            self._integrate(obs)
            self.obs = obs
            # Record the heading the robot ACTUALLY faced, not the planned one.
            self.survey_shots.append(SurveyShot(self._survey_idx, float(bot.yaw),
                                                bot.position[:2].copy(), obs))
            self._survey_idx += 1
            self._settle_since = None
            self._turn_since = t
            return

        bot.drive(0.0, 0.0)
        self._note(f"t={t:.1f}s survey: asking the model about "
                   f"{len(self.survey_shots)} photos")
        self.survey_result = self.survey.query(self.survey_shots)
        self._finish_survey(bot, t, self.survey_result)

    def _finish_survey(self, bot, t, res):
        here = bot.position[:2]
        if res.found and res.detection is not None:
            cam = self.survey_shots[res.photo].obs.cam_pos[:2]
            pos = res.detection.position[:2]
            ray = pos - cam
            ray = ray / max(float(np.linalg.norm(ray)), 1e-6)
            # Same nudge as sense(): the detector sees the near face.
            self.goal_xy = pos + ray * 0.2
            self._goal_provisional = False
            self._note(f"t={t:.1f}s survey: goal at ({self.goal_xy[0]:.2f}, "
                       f"{self.goal_xy[1]:.2f}) from photo {res.photo}, heading "
                       f"{np.degrees(res.heading):.0f} deg")
        elif res.found and res.heading is not None:
            reach = min(self.provisional_distance, self.max_range)
            guess = here + reach * np.array([np.cos(res.heading), np.sin(res.heading)])
            self.goal_xy = self._clip_to_grid(guess)
            self._goal_provisional = True
            self._note(f"t={t:.1f}s survey: goal heading {np.degrees(res.heading):.0f} deg "
                       f"(via {res.heading_source}) but no range; heading for "
                       f"({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}) until it is sighted")
        else:
            why = res.error or "goal not found"
            self._note(f"t={t:.1f}s survey: {why}; falling back to a spin scan")
            self.state = self.SCAN
            return
        self.replan(bot)
        self._next_plan = t + self.plan_period
        self.state = self.NAVIGATE

    def _clip_to_grid(self, xy):
        """Keep a guessed goal inside the map, or planning cannot address it."""
        g = self.grid
        try:
            (x0, y0), (nx, ny), r = g.origin, g.size, g.resolution
        except (AttributeError, TypeError, ValueError):
            return np.asarray(xy, float)
        m = 2 * r
        return np.array([np.clip(xy[0], x0 + m, x0 + nx * r - m),
                         np.clip(xy[1], y0 + m, y0 + ny * r - m)])

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
            self._settle_since = None
            self._turn_since = t
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

        if self.state == self.SURVEY:
            self._survey_tick(bot, t)
            return

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
                if self._goal_provisional:
                    # Reached the survey's guess without ever seeing the goal:
                    # that is not arriving. Look around from here instead.
                    self._note(f"t={t:.1f}s reached the provisional goal without "
                               "sighting the target; scanning")
                    self._goal_provisional = False
                    self.goal_xy = None
                    self.path = []
                    self._scan_start, self._scan_yaw = None, 0.0
                    self.state = self.SCAN
                    bot.drive(0.0, 0.0)
                    return
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
