"""LLM-guided exploration: where should the robot look next?

The survey (`llm_survey.py`) answers "where is the goal?" -- which only helps
when the goal is already in one of the photos. When it is not (behind a wall,
in another room), something has to decide where to go to find it. This module
asks the local LLM exactly that: given the robot's 8 labelled photos and the
places it has already explored, either report the goal, or put a box around the
most promising patch of open floor to drive to next (a doorway, a gap, the
entrance to an unexplored area).

    explorer = LlmExplorer()
    result = explorer.query(shots, visited=[(0.0, 0.0)])
    result.goal          # a SurveyResult if the model saw the goal, else None
    result.waypoint      # world (x, y) to drive to next, or None

The navigator drives to the waypoint, surveys again there, and repeats; a fast
per-frame detector (YOLO) runs the whole time and takes over the moment it
sees the object.

Turning the model's explore box into a waypoint uses geometry, not the model's
sense of distance: the box's column gives an exact bearing (`pixel_heading`),
the depth image along that column says how far is actually open, and the
waypoint is placed short of the nearest obstacle. If the answer is unusable, or
points somewhere already explored, `fallback_waypoint` picks the most open,
least explored direction from the depth images alone -- no model needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .llm_reasoner import DEFAULT_BASE_URL, DEFAULT_MODEL, _extract_json, _goal_pixel, _to_bool, _to_float, _to_int
from .llm_survey import LlmSurvey, SurveyResult, SurveyShot, _wrap, pixel_heading

EXPLORE_PROMPT = """You are the navigator of a balancing two-wheeled robot searching an indoor space for {target}.
The robot stood still and turned in place, taking {n} photos. Each photo below is preceded by its own label: the robot's
WORLD position (x, y in metres), the WORLD heading that photo faces, its angle from photo 0, and the world headings seen
along its left and right edges. Headings are in degrees, 0 = +x axis, counter-clockwise positive. Every photo is
{width} px wide and {height} px tall.
{visited}
First, look for {target} in every photo. If it is visible, report it.
If it is not visible anywhere, choose WHERE TO GO NEXT to find it: the most promising patch of open floor to drive toward
-- a doorway, a gap between obstacles, the entrance to an unexplored area, or far open space. Prefer places the robot has
not explored. Do not choose walls, dead ends, or floor right next to the robot.

Answer with ONLY a single-line compact JSON object -- no prose, no markdown fences. Keys, in this order:
{{"goal_found": true/false, "goal_photo": <photo index or null>, "goal_bbox_2d": [x1, y1, x2, y2] or null, "explore_photo": <photo index or null>, "explore_bbox_2d": [x1, y1, x2, y2] or null, "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}
Boxes are in THAT photo, with coordinates normalised to 0-1000 across the photo's width (x) and height (y).
explore_bbox_2d covers the opening or patch of floor to drive toward. Null the goal fields when goal_found is false,
and the explore fields when it is true."""


@dataclass
class ExploreResult:
    goal: SurveyResult | None = None          # set when the model saw the goal
    waypoint: np.ndarray | None = None        # where to drive next
    waypoint_source: str | None = None        # "model" | "fallback" | None
    explore_photo: int | None = None
    explore_pixel: tuple[int, int] | None = None
    heading: float | None = None              # world bearing of the waypoint (rad)
    confidence: float | None = None
    reason: str = ""
    raw: dict | None = None
    latency: float | None = None
    error: str | None = None
    completion_tokens: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.goal is not None and self.goal.found


# ----------------------------------------------------------------- geometry
def free_distance(obs, u: float, robot_xy, floor_z: float = 0.0,
                  ceiling: float = 1.75, band: int = 4, min_range: float = 0.45) -> float:
    """How far the floor is open along image column u: horizontal distance to
    the nearest thing standing between knee and head height in that column.

    Returns inf when nothing in range stands in the way.
    """
    w = obs.intrinsics.width
    c0, c1 = max(int(round(u)) - band, 0), min(int(round(u)) + band + 1, w)
    pts = obs.points[:, c0:c1][obs.valid[:, c0:c1]]
    if not len(pts):
        return float("inf")
    z = pts[:, 2] - floor_z
    d = np.linalg.norm(pts[:, :2] - np.asarray(robot_xy, float), axis=1)
    standing = (z > 0.15) & (z < ceiling) & (d > min_range)
    return float(d[standing].min()) if standing.any() else float("inf")


def waypoint_from_pixel(shot: SurveyShot, u: float, v: float, max_step: float = 3.5,
                        min_step: float = 0.8, margin: float = 0.7, floor_z: float = 0.0,
                        ceiling: float = 1.75):
    """A reachable waypoint toward pixel (u, v) of a shot, or None if blocked.

    The bearing is exact (the pixel's column). The distance is the smallest of:
    the step cap, how far the pixel's own depth return is, and the open floor
    along that column minus a safety margin.
    """
    obs = shot.obs
    heading = pixel_heading(shot, u)
    step = max_step
    h, w = obs.depth.shape
    ui, vi = int(np.clip(round(u), 0, w - 1)), int(np.clip(round(v), 0, h - 1))
    if obs.valid[vi, ui]:
        point = obs.points[vi, ui]
        step = min(step, float(np.linalg.norm(point[:2] - shot.robot_xy)))
    free = free_distance(obs, u, shot.robot_xy, floor_z, ceiling)
    step = min(step, free - margin)
    if step < min_step:
        return None, heading
    return np.asarray(shot.robot_xy, float) + step * np.array([np.cos(heading), np.sin(heading)]), heading


def _near_visited(xy, visited, radius: float) -> bool:
    return any(np.linalg.norm(np.asarray(xy) - np.asarray(p)) < radius for p in visited)


def fallback_waypoint(shots: list[SurveyShot], visited, max_step: float = 3.5,
                      min_step: float = 0.8, margin: float = 0.7, visit_radius: float = 1.0,
                      floor_z: float = 0.0, ceiling: float = 1.75):
    """The most open, least explored direction, from the depth images alone.

    Scores three columns per photo by how far the floor is open along them,
    plus how far the resulting waypoint is from every place already explored.
    Returns (waypoint, heading) or (None, None) if everything is blocked.
    """
    best = (None, None, -np.inf)
    for s in shots:
        intr = s.obs.intrinsics
        for off in (-0.25, 0.0, 0.25):
            u = intr.cx + off * intr.width
            free = free_distance(s.obs, u, s.robot_xy, floor_z, ceiling)
            step = min(max_step, free - margin)
            if step < min_step:
                continue
            heading = pixel_heading(s, u)
            wp = np.asarray(s.robot_xy, float) + step * np.array([np.cos(heading), np.sin(heading)])
            novelty = min((np.linalg.norm(wp - np.asarray(p)) for p in visited), default=3.0)
            score = step + min(novelty, 3.0) - (3.0 if _near_visited(wp, visited, visit_radius) else 0.0)
            if score > best[2]:
                best = (wp, heading, score)
    return best[0], best[1]


# ----------------------------------------------------------------- explorer
class LlmExplorer:
    """Asks the model where the goal is, or else where to look next."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 n_shots: int = 8, goal_label: str = "target", min_confidence: float = 0.30,
                 max_tokens: int = 1024, temperature: float = 0.1, timeout: float = 300.0,
                 verbose: bool = False, thinking: bool = False, max_step: float = 3.5,
                 min_step: float = 0.8, margin: float = 0.7, visit_radius: float = 1.0,
                 floor_z: float = 0.0, ceiling: float = 1.75,
                 target: str = "a tall RED cylinder"):
        # The survey supplies the plan, the per-photo labels, the client and the
        # goal ranging, so a goal the explorer reports is handled identically.
        self.survey = LlmSurvey(base_url, model, n_shots=n_shots, goal_label=goal_label,
                                min_confidence=min_confidence, max_tokens=max_tokens,
                                temperature=temperature, timeout=timeout, verbose=verbose,
                                thinking=thinking)
        self.client = self.survey.client
        self.n_shots = n_shots
        self.target = target            # what to look for, as the prompt names it
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.max_step, self.min_step, self.margin = max_step, min_step, margin
        self.visit_radius = visit_radius
        self.floor_z, self.ceiling = floor_z, ceiling
        self.queries = self.errors = 0
        self.last_result: ExploreResult | None = None

    def headings(self, start_yaw: float) -> list[float]:
        return self.survey.headings(start_yaw)

    def prompt(self, shots: list[SurveyShot], visited) -> str:
        intr = shots[0].obs.intrinsics
        if visited:
            where = ", ".join(f"({p[0]:.1f}, {p[1]:.1f})" for p in visited)
            visited_txt = f"\nPlaces already explored (the target was not found there): {where}.\n"
        else:
            visited_txt = "\nNothing has been explored yet.\n"
        return EXPLORE_PROMPT.format(n=len(shots), width=intr.width, height=intr.height,
                                     visited=visited_txt, target=self.target)

    def query(self, shots: list[SurveyShot], visited=()) -> ExploreResult:
        if not shots:
            raise ValueError("no shots to explore from")
        visited = [np.asarray(p, float)[:2] for p in visited]
        try:
            content, dt = self.client.ask_images([s.obs.rgb for s in shots],
                                                 [s.label(shots[0].heading) for s in shots],
                                                 self.prompt(shots, visited))
        except Exception as e:  # server down, timeout, HTTP error
            self.errors += 1
            res = self._fallback(shots, visited, ExploreResult(error=f"{type(e).__name__}: {e}"),
                                 "model unavailable")
            return self._finish(res)
        usage = getattr(self.client, "last_usage", {}) or {}
        try:
            raw = _extract_json(content)
        except ValueError as e:
            self.errors += 1
            res = ExploreResult(error=f"unparseable answer ({e})", latency=dt,
                                completion_tokens=usage.get("completion_tokens"))
            return self._finish(self._fallback(shots, visited, res, "unparseable answer"))
        self.queries += 1
        res = self.interpret(shots, raw, visited)
        res.latency, res.completion_tokens = dt, usage.get("completion_tokens")
        return self._finish(res)

    def interpret(self, shots: list[SurveyShot], raw: dict, visited=()) -> ExploreResult:
        visited = [np.asarray(p, float)[:2] for p in visited]
        res = ExploreResult(confidence=_to_float(raw.get("confidence")),
                            reason=str(raw.get("reason", "")), raw=raw)

        if _to_bool(raw.get("goal_found")):
            goal = self.survey.interpret(shots, {
                "goal_found": True, "photo": raw.get("goal_photo"),
                "bbox_2d": raw.get("goal_bbox_2d"), "confidence": raw.get("confidence")})
            if goal.found:
                res.goal = goal
                return res
            res.notes.append("model reported the goal but it could not be located; exploring")

        photo = _to_int(raw.get("explore_photo"))
        if photo is None or not (0 <= photo < len(shots)):
            return self._fallback(shots, visited, res, "no usable explore photo")
        intr = shots[photo].obs.intrinsics
        pixel = _goal_pixel({"bbox_2d": raw.get("explore_bbox_2d")}, intr.width, intr.height)
        if pixel is None:
            return self._fallback(shots, visited, res, "no usable explore box")
        res.explore_photo, res.explore_pixel = photo, pixel
        wp, heading = waypoint_from_pixel(shots[photo], pixel[0], pixel[1], self.max_step,
                                          self.min_step, self.margin, self.floor_z, self.ceiling)
        res.heading = heading
        if wp is None:
            return self._fallback(shots, visited, res, "explore target is blocked or too close")
        if _near_visited(wp, visited, self.visit_radius):
            return self._fallback(shots, visited, res, "explore target was already explored")
        res.waypoint, res.waypoint_source = wp, "model"
        return res

    def _fallback(self, shots, visited, res: ExploreResult, why: str) -> ExploreResult:
        wp, heading = fallback_waypoint(shots, visited, self.max_step, self.min_step,
                                        self.margin, self.visit_radius, self.floor_z, self.ceiling)
        res.notes.append(f"{why}; using the most open unexplored direction")
        if wp is not None:
            res.waypoint, res.waypoint_source, res.heading = wp, "fallback", heading
        else:
            res.notes.append("every direction is blocked")
        return res

    def _finish(self, res: ExploreResult) -> ExploreResult:
        self.last_result = res
        if self.verbose:
            if res.found:
                d = res.goal.detection
                where = f"at ({d.position[0]:.2f}, {d.position[1]:.2f})" if d is not None else "unranged"
                print(f"    [explore] goal in photo {res.goal.photo} {where}: {res.reason}")
            elif res.waypoint is not None:
                print(f"    [explore] next waypoint ({res.waypoint[0]:.2f}, {res.waypoint[1]:.2f}) "
                      f"via {res.waypoint_source}"
                      f"{f' (photo {res.explore_photo} px {res.explore_pixel})' if res.explore_photo is not None else ''}"
                      f": {res.reason} {'; '.join(res.notes)}")
            else:
                print(f"    [explore] nowhere to go: {res.error or ''} {'; '.join(res.notes)}")
        return res

    def __repr__(self):
        return (f"<LlmExplorer {self.client.model} shots={self.n_shots} "
                f"queries={self.queries} errors={self.errors}>")
