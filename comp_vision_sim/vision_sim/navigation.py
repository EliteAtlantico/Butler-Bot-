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


def _wrap_array(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class VisualNavigator:
    """Search -> detect -> map -> plan -> follow, driven by the RGB-D camera."""

    SURVEY, SCAN, EXPLORE, NAVIGATE, ARRIVED, STUCK = (
        "survey", "scan", "explore", "navigate", "arrived", "stuck")

    @classmethod
    def for_bot(cls, bot, goal=None, camera: str | None = None, **kw):
        """Build a navigator configured from the scene the robot is in.

        Reads the world extent, floor height, footprint radius and camera off
        the model, so the same call works in a scene this code has never seen.
        `goal` may be an (x, y) world coordinate -- which needs no detector
        and no colour palette -- or a class label to look for.
        """
        info = SceneInfo.from_model(bot.model, bot.data, camera=camera)
        if goal is not None and not isinstance(goal, str):
            # The grid is sized to the scenery, which says nothing about where
            # you want to go. A goal beyond it can never be planned to.
            info = info.including(goal)
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
                 safety_range: float = 0.55,
                 grid: occupancy.OccupancyGrid | None = None,
                 # 0.10 m could not represent a standard interior door: the
                 # free corridor through a 0.76 m opening is ~0.12 m, around
                 # one cell, and the planner found nothing through it.
                 # Measured at 0.05 m it passes down to 0.64 m, for 4x the
                 # cells but only ~25% more wall time -- rendering dominates,
                 # not A*.
                 resolution: float = 0.05,
                 survey=None, survey_turn_tol: float = 0.08,
                 survey_settle: float = 0.3, survey_turn_timeout: float = 6.0,
                 provisional_distance: float = 3.0,
                 track: str = "detector", lost_after: float = 4.0,
                 relocate_distance: float = 1.0, max_researches: int = 3,
                 research_cooldown: float = 3.0, sighting_range: float = 7.0,
                 presence_radius: float = 0.45, presence_min_points: int = 20,
                 goal_probe_height: float = 0.6, research_scan_turns: int = 2,
                 explorer=None, explore_waypoint_tol: float = 0.45,
                 max_explore_steps: int = 8, on_give_up=None,
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
        # Reactive stop: forward motion is cut when a return inside this cone
        # is closer than this range. Sized off the footprint so it scales with
        # the robot rather than being a tuned constant.
        self.safety_cone = np.deg2rad(40.0)
        self.safety_range = safety_range

        # Everything below is measured from the scene when one was supplied,
        # and falls back to the values tuned for the original course when it
        # was not -- so existing callers keep their behaviour exactly.
        self.camera = scene.camera if scene else None
        self.floor_z = scene.floor_z if scene else 0.0
        # Anything above the robot's head is not an obstacle it can hit. A
        # doorway lintel or a low ceiling otherwise projects straight down
        # into the 2-D grid and seals a gap the robot can drive through.
        self.obstacle_ceiling = (scene.robot_top + 0.10) if scene else 2.0
        # Inflation has to cover the grid's own quantisation, not just the
        # footprint. "Occupied" means an obstacle somewhere inside that cell,
        # so true clearance can fall short of the inflation by half a cell
        # diagonal. At radius+0.05 on a 0.10 m grid that deficit exceeded the
        # margin outright, and the robot threaded gaps barely wider than
        # itself and caught a shoulder. Deriving the margin from the
        # resolution keeps the guarantee if either is retuned, and stays
        # tight enough to clear a standard interior door.
        quantisation = 0.75 * resolution
        self.robot_radius = (robot_radius if robot_radius is not None
                             else (scene.robot_radius + quantisation if scene
                                   else 0.25 + quantisation))
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

        # Losing the goal, and finding it again.
        #   track="detector": the goal is confirmed by the per-frame detector.
        #   track="depth":    once a goal is known, confirm it from the depth
        #                     image alone (is something still standing there?)
        #                     and stop asking the detector -- with the LLM
        #                     detector that removes every per-frame query.
        if track not in ("detector", "depth"):
            raise ValueError(f"track must be 'detector' or 'depth', not {track!r}")
        self.track = track
        self.lost_after, self.relocate_distance = lost_after, relocate_distance
        self.max_researches, self.research_cooldown = max_researches, research_cooldown
        self.sighting_range = sighting_range
        self.presence_radius, self.presence_min_points = presence_radius, presence_min_points
        self.goal_probe_height = goal_probe_height
        self.research_scan_turns = research_scan_turns
        self._empty_turns = 0
        self.researches = 0
        self._last_sighting: float | None = None
        self._miss_since: float | None = None
        self._search_done_at: float | None = None

        # LLM-guided exploration (vision_sim.llm_explore.LlmExplorer): survey,
        # let the fast detector look at every photo, and when it sees nothing
        # ask the model where to go next; drive there and survey again.
        self.explorer = explorer
        self.explore_waypoint_tol = explore_waypoint_tol
        self.max_explore_steps = max_explore_steps
        self.explored: list[np.ndarray] = []
        self.explore_target: np.ndarray | None = None
        self.explore_result = None
        self.explore_steps = 0
        self._explore_plan_failures = 0
        # Called once, as on_give_up(navigator, reason), when the search ends
        # without the goal -- where a remote-control hand-off plugs in.
        self.on_give_up = on_give_up
        self.give_up_reason: str | None = None
        if explorer is not None:
            explorer.floor_z = self.floor_z
            explorer.ceiling = getattr(self, "obstacle_ceiling", explorer.ceiling)

        searching = survey is not None or explorer is not None
        self.state = (self.SURVEY if (searching and self.fixed_goal is None)
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
        self._off_grid_warned = False
        self._settled_at = None
        self.range_ahead = np.inf

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
        goal_known = self.goal_xy is not None and not self._goal_provisional
        use_detector = (self.state != self.SURVEY and
                        not (self.track == "depth" and goal_known))
        self.detections = (self.detector(self.obs, robot_yaw=bot.yaw, t=t)
                           if use_detector else [])
        if len(self.detections) >= len(self.best_detections):
            self.best_obs, self.best_detections, self.best_time = (
                self.obs, list(self.detections), t)

        self._integrate(self.obs)

        # Reactive clearance, computed straight off this frame's returns and
        # deliberately independent of the map and the planner: those run at
        # 1 Hz and can be wrong, and neither is allowed to be the only thing
        # standing between the robot and a wall.
        #
        # It must NOT reuse `hits`. That set drops everything within
        # self_radius of the chassis to reject the robot's own arms, which
        # means an obstacle closer than ~0.5 m disappears from it entirely --
        # measured: at a 0.35 m gap the filtered set reported clear. Safety
        # reads the near field with only a token exclusion for the camera
        # housing, which is the one place those returns matter.
        near = perception.obstacle_points(
            self.obs, self_radius=0.12, min_range=0.15,
            floor_z=self.floor_z, max_height=self.obstacle_ceiling)
        self.range_ahead = self._forward_clearance(bot, near)

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
            elif np.linalg.norm(estimate - self.goal_xy) > self.relocate_distance:
                # Seen somewhere else entirely: the goal moved (or the old
                # estimate was wrong). Averaging would crawl there over many
                # sightings while driving toward the midpoint; jump instead.
                self._note(f"t={t:.1f}s goal seen at ({estimate[0]:.2f}, {estimate[1]:.2f}), "
                           f"{np.linalg.norm(estimate - self.goal_xy):.2f} m from where it was; "
                           "moving the goal there")
                self.goal_xy = estimate
                self.path = []
                self._next_plan = t
            else:
                self.goal_xy = 0.7 * self.goal_xy + 0.3 * estimate
        self._update_tracking(t, bool(goals))

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
        ok = self._plan_to(bot, self.goal_xy, self.stop_distance)
        if ok is None:
            return False
        if not ok:
            self._plan_failures += 1
            if self._plan_failures >= 8:
                if self.fixed_goal is None:
                    # No route to where we think the goal is: the estimate may
                    # be wrong, so look again before giving up.
                    self._research(bot, self._last_t or 0.0, "no route to the goal")
                else:
                    # _settle, not a bare state write: displacement is measured
                    # from where the robot actually stopped, so the kidnapped-
                    # robot check cannot misfire against a stale reference.
                    self._settle(bot, self.STUCK)
            return False
        self._plan_failures = 0
        return True

    def _plan_to(self, bot, target_xy, stop_distance):
        """A* from the robot to `stop_distance` short of a target.

        True on success (sets self.path), False when there is no route, None
        when the robot is already on the target.
        """
        target_xy = np.asarray(target_xy, float)
        blocked, soft = self.grid.costmap(robot_radius=self.robot_radius)
        # Stand off from the target instead of driving into it.
        here = bot.position[:2]
        delta = target_xy - here
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return None
        stand_off = target_xy - delta / dist * stop_distance

        start = self.grid.to_cell(here)[0]
        goal = self.grid.to_cell(stand_off)[0]
        # Off-grid start or goal fails inside A* with no explanation and looks
        # exactly like "no route exists". Say which it is.
        for label, cell in (("goal", goal), ("robot", start)):
            if not self.grid.inside(cell)[0]:
                if not self._off_grid_warned:
                    self._off_grid_warned = True
                    x0, y0 = self.grid.origin
                    x1 = x0 + self.grid.size[0] * self.grid.resolution
                    y1 = y0 + self.grid.size[1] * self.grid.resolution
                    self._note(f"{label} is outside the map "
                               f"(x {x0:.1f}..{x1:.1f}, y {y0:.1f}..{y1:.1f}) "
                               f"-- no plan is possible; widen the grid")
                return False

        if blocked[tuple(start)]:
            # The robot is inside its own inflation (hugging a wall). Free the
            # cells it physically occupies, or no plan can ever start.
            blocked = blocked.copy()
            sx, sy = start
            blocked[max(sx - 2, 0):sx + 3, max(sy - 2, 0):sy + 3] = False

        # Weight the clearance cost heavily: given a choice the route should
        # run down the middle of open space rather than shave past furniture,
        # and only hug a wall where that is genuinely the only way through.
        cells = planning.astar(blocked, start, goal, soft=soft, soft_weight=5.0)
        if cells is None:
            # Drop the old path. Keeping it meant a robot whose route had just
            # been invalidated carried on following the stale plan straight
            # into whatever had blocked it, and leaned there until it fell.
            #
            # Counting the failure and deciding when to give up belong to the
            # caller: `replan` for a goal, `_explore_tick` for a waypoint (it
            # keeps its own tally). Counting here too would advance both
            # tallies on a single failure and halve the effective patience.
            self.path = []
            self._cmd = (0.0, 0.0)
            return False
        cells = planning.shortcut(blocked, cells)
        self.path = [self.grid.to_world(c)[0] for c in cells[1:]] or [stand_off]
        self.blocked = blocked
        self._wp = 0
        return True

    def _forward_clearance(self, bot, hits) -> float:
        """Distance to the nearest obstacle return inside a forward cone."""
        if hits is None or len(hits) == 0:
            return np.inf
        d = hits[:, :2] - bot.position[:2]
        rng = np.linalg.norm(d, axis=1)
        bearing = _wrap_array(np.arctan2(d[:, 1], d[:, 0]) - bot.yaw)
        cone = np.abs(bearing) < self.safety_cone
        return float(rng[cone].min()) if cone.any() else np.inf

    # ------------------------------------------------------------- recovery
    def _settle(self, bot, state):
        """Stop, and remember where -- displacement is measured from here."""
        self.state = state
        self._settled_at = bot.position[:2].copy()
        self.path = []
        self._cmd = (0.0, 0.0)

    def _displaced(self, bot):
        """Has something moved the robot away from where it settled?

        Measured against the settle position, NOT against distance to the
        goal. Using the goal meant a robot that stopped because no route
        existed was always "displaced" -- it re-engaged immediately, drove
        back into whatever had blocked it, gave up, and repeated, leaning on
        the obstacle for the whole run. A viewer reset, a shove or a wheel
        slip all still register, because those genuinely move the robot.
        """
        if self._settled_at is None:
            return False
        moved = float(np.linalg.norm(bot.position[:2] - self._settled_at))
        return moved > self.re_engage_margin

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
            self._survey_plan = (self.explorer or self.survey).headings(bot.yaw)
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
        if self.explorer is not None:
            self._finish_explore_survey(bot, t)
            return
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
            self._search_done_at = t
            self.state = self.SCAN
            return
        self._search_done_at = t
        self._miss_since = None
        self.replan(bot)
        self._next_plan = t + self.plan_period
        self.state = self.NAVIGATE

    # ------------------------------------------------------------- explore
    def _start_survey(self, t):
        self.survey_shots = []
        self.survey_result = None
        self._survey_plan = None
        self._survey_idx = 0
        self._settle_since = None
        self._turn_since = t
        self.state = self.SURVEY

    def _cheap_detector(self) -> bool:
        """Fast enough to run on every survey photo (YOLO, colour, geometric).
        An LLM detector is not: that would be one model query per photo."""
        return not hasattr(self.detector, "client")

    def _detect_in_shots(self, t):
        """Run the detector on every survey photo; the goal estimate, or None."""
        best = None
        for shot in self.survey_shots:
            dets = self.detector(shot.obs, robot_yaw=shot.heading, t=t)
            if len(dets) >= len(self.best_detections):
                self.best_obs, self.best_detections, self.best_time = shot.obs, list(dets), t
            for d in dets:
                if d.label == self.goal_label and (best is None or d.pixels > best[1].pixels):
                    best = (shot, d)
        if best is None:
            return None
        shot, d = best
        ray = d.position[:2] - shot.obs.cam_pos[:2]
        ray = ray / max(float(np.linalg.norm(ray)), 1e-6)
        # Same nudge as sense(): the detector sees the near face.
        return d.position[:2] + ray * 0.2, shot.index

    def _finish_explore_survey(self, bot, t):
        here = bot.position[:2].copy()
        # 1. The fast detector looks at every photo first. If it already sees
        #    the object, the model has nothing to add and is not asked.
        if self._cheap_detector():
            hit = self._detect_in_shots(t)
            if hit is not None:
                self.goal_xy, photo = hit
                self._goal_provisional = False
                self._note(f"t={t:.1f}s survey: detector found the goal in photo {photo} at "
                           f"({self.goal_xy[0]:.2f}, {self.goal_xy[1]:.2f}); no model query needed")
                self._go_to_goal(bot, t)
                return
        if self.explore_steps >= self.max_explore_steps:
            self._give_up(bot, t, f"explored {self.explore_steps} place(s) without finding the goal")
            return
        # 2. Nothing seen: ask the model where the goal is, or where to look next.
        self._note(f"t={t:.1f}s survey: nothing found; asking the model where to look next")
        res = self.explorer.query(self.survey_shots, visited=self.explored + [here])
        self.explore_result = res
        self.explore_steps += 1
        self.explored.append(here)
        if res.found and res.goal.detection is not None:
            self._finish_survey(bot, t, res.goal)
            return
        if res.waypoint is None:
            self._give_up(bot, t, f"explore: nowhere left to go ({'; '.join(res.notes) or res.error})")
            return
        self.explore_target = self._clip_to_grid(res.waypoint)
        self._explore_plan_failures = 0
        self._note(f"t={t:.1f}s explore {self.explore_steps}: heading for "
                   f"({self.explore_target[0]:.2f}, {self.explore_target[1]:.2f}) via "
                   f"{res.waypoint_source}: {res.reason}")
        self.path = []
        self._plan_to(bot, self.explore_target, 0.0)
        self._next_plan = t + self.plan_period
        self.state = self.EXPLORE

    def _go_to_goal(self, bot, t):
        self.explore_target = None
        self._search_done_at = t
        self._miss_since = None
        self.replan(bot)
        self._next_plan = t + self.plan_period
        self.state = self.NAVIGATE

    def _explore_tick(self, bot, t):
        # The detector runs every sense tick while exploring; the moment it
        # puts a goal on the map, stop exploring and go to it.
        if self.goal_xy is not None and not self._goal_provisional:
            self._note(f"t={t:.1f}s goal spotted at ({self.goal_xy[0]:.2f}, "
                       f"{self.goal_xy[1]:.2f}) while exploring; heading for it")
            self._go_to_goal(bot, t)
            bot.drive(0.0, 0.0)
            return
        here = bot.position[:2]
        if np.linalg.norm(self.explore_target - here) < self.explore_waypoint_tol:
            self._note(f"t={t:.1f}s reached waypoint ({self.explore_target[0]:.2f}, "
                       f"{self.explore_target[1]:.2f}); surveying again")
            self.explore_target = None
            self._start_survey(t)
            bot.drive(0.0, 0.0)
            return
        if t >= self._next_plan:
            self._next_plan = t + self.plan_period
            ok = self._plan_to(bot, self.explore_target, 0.0)
            if ok is False:
                self._explore_plan_failures += 1
                if self._explore_plan_failures >= 5:
                    self._note(f"t={t:.1f}s no route to waypoint ({self.explore_target[0]:.2f}, "
                               f"{self.explore_target[1]:.2f}); surveying again from here")
                    self.explored.append(np.asarray(self.explore_target, float))
                    self.explore_target = None
                    self._start_survey(t)
                    bot.drive(0.0, 0.0)
                    return
            elif ok:
                self._explore_plan_failures = 0
        if self.path:
            self._cmd = self._follow(bot)
        bot.drive(*self._cmd)

    # ------------------------------------------------------- losing the goal
    def _update_tracking(self, t, sighted):
        """Record a sighting, or start/continue a miss when one was expected."""
        tracking = (self.fixed_goal is None and self.goal_xy is not None and
                    not self._goal_provisional and
                    self.state in (self.NAVIGATE, self.ARRIVED))
        if not tracking or self.obs is None:
            self._miss_since = None
            return
        if not sighted and self.track == "depth":
            sighted = self._present(self.obs)
        if sighted:
            self._last_sighting = t
            self._miss_since = None
            return
        if not self._expect_visible(self.obs):
            # Out of frame, out of range or behind something: not seeing it
            # says nothing about whether it is still there.
            self._miss_since = None
            return
        if self._miss_since is None:
            self._miss_since = t

    def _goal_probe(self):
        return np.array([self.goal_xy[0], self.goal_xy[1],
                         self.floor_z + self.goal_probe_height])

    def _expect_visible(self, obs) -> bool:
        """Should the camera be able to see the goal in this frame?"""
        target = self._goal_probe()
        if np.linalg.norm(target[:2] - obs.cam_pos[:2]) > self.sighting_range:
            return False
        p = obs.cam_mat.T @ (target - obs.cam_pos)
        if p[2] >= -1e-6:
            return False                              # behind the camera
        intr = obs.intrinsics
        u = intr.fx * p[0] / -p[2] + intr.cx
        v = intr.fy * -p[1] / -p[2] + intr.cy
        margin = 0.1 * intr.width
        if not (margin <= u <= intr.width - margin and 0 <= v < intr.height):
            return False
        ui, vi = int(u), int(v)
        patch = obs.depth[max(vi - 3, 0):vi + 4, max(ui - 3, 0):ui + 4]
        blocked = np.isfinite(patch) & (patch < -p[2] - 0.6)
        return float(blocked.mean()) < 0.5 if patch.size else False

    def _present(self, obs) -> bool:
        """Does the depth image show something standing at the goal?"""
        pts = obs.points[obs.valid]
        if not len(pts):
            return False
        z = pts[:, 2] - self.floor_z
        near = np.linalg.norm(pts[:, :2] - self.goal_xy, axis=1) < self.presence_radius
        standing = (z > 0.25) & (z < getattr(self, "obstacle_ceiling", 2.0))
        return int((near & standing).sum()) >= self.presence_min_points

    def _goal_lost(self, t) -> bool:
        if self._miss_since is None or t - self._miss_since < self.lost_after:
            return False
        cooled = (self._search_done_at is None or
                  t - self._search_done_at >= self.research_cooldown)
        return cooled

    def _research(self, bot, t, why):
        """Forget the goal and run the search again from where the robot is."""
        self._miss_since = None
        if self.researches >= self.max_researches:
            self._give_up(bot, t, f"{why}; already re-searched {self.researches} time(s)")
            return
        self.researches += 1
        self._note(f"t={t:.1f}s {why}; re-running the search "
                   f"({self.researches}/{self.max_researches}) from "
                   f"({bot.position[0]:.2f}, {bot.position[1]:.2f})")
        self.goal_xy = None
        self._goal_provisional = False
        self.path = []
        self._wp = 0
        self._plan_failures = 0
        self._empty_turns = 0
        forget = getattr(self.detector, "forget", None)
        if callable(forget):
            forget()
        self.explore_target = None
        if self.survey is not None or self.explorer is not None:
            self._start_survey(t)
        else:
            self._scan_start, self._scan_yaw = None, 0.0
            self.state = self.SCAN
        bot.drive(0.0, 0.0)

    def _give_up(self, bot, t, reason):
        """End the search without the goal, and tell whoever is listening."""
        self._note(f"t={t:.1f}s {reason}; giving up")
        self.state = self.STUCK
        bot.drive(0.0, 0.0)
        if self.give_up_reason is None:
            self.give_up_reason = reason
            if callable(self.on_give_up):
                self.on_give_up(self, reason)

    @property
    def outcome(self) -> str | None:
        """'found' once arrived, 'gave_up' once the search ended without the goal."""
        return {self.ARRIVED: "found", self.STUCK: "gave_up"}.get(self.state)

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
            self._miss_since = None
            self._search_done_at = None
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

        if self._goal_lost(t):
            self._research(bot, t, f"goal not seen for {t - self._miss_since:.1f}s "
                                   "where it should be visible")

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
                    self._search_done_at = t
                    self._miss_since = None
                    self.state = self.NAVIGATE
            elif turned:
                self._scan_yaw = 0.0      # nothing found, go round again
                self._empty_turns += 1
                # The very first search spins until it finds something. A
                # RE-search that keeps turning up nothing is a failed search:
                # without this, a goal that vanished leaves the robot spinning
                # for ever and never reaching STUCK.
                if self.researches > 0 and self._empty_turns >= self.research_scan_turns:
                    self._research(bot, t, f"goal not found after {self._empty_turns} "
                                           "full turn(s)")
                    return
            bot.drive(0.0, self.scan_rate)
            return

        if self.state == self.EXPLORE:
            self._explore_tick(bot, t)
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
                self._note(f"t={t:.1f}s arrived, "
                           f"{np.linalg.norm(self.goal_xy - bot.position[:2]):.2f}m from goal")
                self._settle(bot, self.ARRIVED)
                bot.drive(0.0, 0.0)
                return
            if t >= self._next_plan:
                self._next_plan = t + self.plan_period
                self.replan(bot)
            self._cmd = self._follow(bot) if self.path else (0.0, 0.0)
            v, w = self._cmd
            # Reactive stop. The planner works off a 1 Hz map and can be wrong
            # about what is in front of the robot right now; this reads the
            # current frame and refuses forward motion regardless. Turning
            # stays allowed, so the robot can still look for a way out.
            if self.range_ahead < self.safety_range:
                v = 0.0
            bot.drive(v, w)
            return

        bot.drive(0.0, 0.0)

    def _note(self, msg):
        self.log.append(msg)
        if self.verbose:
            print(msg)
