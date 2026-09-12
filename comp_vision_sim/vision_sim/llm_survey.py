"""A panoramic survey: 8 photos around the robot, one LLM query, one direction.

The single-frame detector asks the model about whatever happens to be in front
of the camera, one ~10 s query at a time, while the robot spins. A survey does
the looking first and the thinking once: the robot turns in place to N evenly
spaced headings, takes a registered RGB-D shot at each, and sends ALL of them to
the model in a single request, every photo labelled with the robot's world
position and the camera's world heading. The model compares the views and
answers which photo holds the goal, where in that photo, and which world heading
leads to it.

    survey = LlmSurvey(n_shots=8)
    shots = [...]                        # SurveyShot per heading (VisualNavigator takes them)
    result = survey.query(shots)
    result.detection                     # ranged goal, or None
    result.heading                       # world heading to the goal (rad), or None

What comes back is used in order of how much it can be trusted:

  1. photo + pixel -> ranged against THAT shot's depth image, exactly like the
     single-frame detector, giving a full world position;
  2. photo + pixel but no usable range -> the pixel's bearing in that photo;
  3. heading only -> the model's stated world heading;
  4. nothing -> not found; the caller falls back to a plain spin scan.

The model is asked to answer directly, but on multi-photo prompts this Qwen
build reasons anyway (into `reasoning_content`) and a 1024-token cap was used
up before it wrote a single character of the answer. The survey therefore
defaults to a much larger `max_tokens`; one query with 8 photos measured
~20-50 s and ~2.8k completion tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .llm_reasoner import (DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient,
                           LlmGoalDetector, _extract_json, _goal_pixel, _to_bool,
                           _to_float, _to_int)
from .perception import Detection, Observation


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


SURVEY_PROMPT = """You are the visual cortex of a balancing two-wheeled robot on an indoor obstacle course.
The robot stood still and turned in place, taking {n} photos, one per heading. Each photo below is
preceded by its own label: the robot's WORLD position (x, y in metres), the WORLD heading that photo
faces, its angle from photo 0, and the world headings seen along its left and right edges. Headings are
in degrees, 0 = +x axis, counter-clockwise positive, so 90 = +y. Every photo is {width} px wide and
{height} px tall (horizontal_px: 0 = left edge; vertical_px: 0 = top edge). Within a photo, things to
the LEFT of centre are at a LARGER world heading than the camera's, things to the RIGHT at a smaller one.

GOAL: a tall RED cylinder standing on the floor.
OTHERS: orange barriers (low walls), blue pillars, a grey checkerboard floor, a hazy sky. Ignore the
robot's own white arms if they enter a frame.

Compare all the photos. If the goal appears in more than one, choose the photo where it is closest to
the horizontal centre. Answer with ONLY a single-line compact JSON object -- no prose, no markdown
fences, nothing after the closing brace. Keys, in this order:
{{"goal_found": true/false, "photo": <photo index or null>, "bbox_2d": [x1, y1, x2, y2] or null, "heading_deg": <world heading from the robot toward the goal, or null>, "confidence": <0.0-1.0>, "reason": "<one short sentence>"}}
bbox_2d is the goal's bounding box in THAT photo, with coordinates normalised to 0-1000 across the photo's
width (x) and height (y). If you are not confident the red goal is in any photo, set goal_found=false and
null photo, bbox_2d and heading_deg."""


@dataclass
class SurveyShot:
    """One photo of the survey and exactly where it was taken from."""
    index: int
    heading: float              # world yaw of the camera when the photo was taken (rad)
    robot_xy: np.ndarray        # (2,) world position of the robot
    obs: Observation            # the registered RGB-D frame

    def label(self, reference_heading: float | None = None) -> str:
        """The text sent immediately before this photo: where it was taken from
        and exactly which directions it shows.

        Each photo carries its OWN direction -- its world heading, its angle
        from photo 0, and the world headings at its left and right edges -- so
        the model can place anything it sees in a photo without relying on the
        photos' order or on the other labels.
        """
        heading = np.degrees(self.heading) % 360
        parts = [f"Photo {self.index}: robot at ({self.robot_xy[0]:.2f}, {self.robot_xy[1]:.2f})",
                 f"this photo faces world heading {heading:.0f} deg"]
        if reference_heading is not None:
            rel = np.degrees(_wrap(self.heading - reference_heading)) % 360
            parts.append("it is the reference direction for the other photos"
                         if self.index == 0 or rel < 0.5 else
                         f"that is {rel:.0f} deg counter-clockwise from photo 0")
        span = self.view_span()
        if span is not None:
            left, right = (np.degrees(a) % 360 for a in span)
            parts.append(f"its left edge looks along {left:.0f} deg and its right edge "
                         f"along {right:.0f} deg")
        return "; ".join(parts)

    def view_span(self) -> tuple[float, float] | None:
        """World headings (rad) seen at this photo's left and right edges."""
        if self.obs is None:
            return None
        intr = self.obs.intrinsics
        half = float(np.arctan2(intr.cx, intr.fx))
        return float(_wrap(self.heading + half)), float(_wrap(self.heading - half))


@dataclass
class SurveyResult:
    found: bool
    photo: int | None = None
    pixel: tuple[int, int] | None = None
    heading: float | None = None          # world heading toward the goal (rad)
    heading_source: str | None = None     # "range" | "pixel" | "model" | None
    confidence: float | None = None
    reason: str = ""
    detection: Detection | None = None    # ranged goal position, when available
    raw: dict | None = None
    latency: float | None = None
    error: str | None = None
    finish_reason: str | None = None
    completion_tokens: int | None = None
    notes: list[str] = field(default_factory=list)


class LlmSurvey:
    """Builds the survey plan, asks the model once, and interprets the answer."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 n_shots: int = 8, goal_label: str = "target",
                 min_confidence: float = 0.40, max_tokens: int = 6144,
                 temperature: float = 0.1, timeout: float = 300.0,
                 verbose: bool = False, thinking: bool = False):
        if n_shots < 1:
            raise ValueError("a survey needs at least one shot")
        self.n_shots = int(n_shots)
        self.goal_label = goal_label
        self.min_confidence = min_confidence
        self.verbose = verbose
        self.client = LLMClient(base_url, model, max_tokens, temperature, timeout, verbose,
                                thinking=thinking)
        # Reuse the single-frame detector's pixel -> world ranging verbatim, so
        # a survey hit and a live-frame hit are converted identically.
        self._ranger = LlmGoalDetector(base_url, model, goal_label=goal_label,
                                       min_confidence=0.0)
        self.queries = 0
        self.errors = 0
        self.truncated = 0
        self.last_result: SurveyResult | None = None

    # ---------------------------------------------------------------- plan
    def headings(self, start_yaw: float) -> list[float]:
        """World yaws of the N shots, evenly spaced, starting where the robot faces."""
        step = 2 * np.pi / self.n_shots
        return [float(_wrap(start_yaw + k * step)) for k in range(self.n_shots)]

    def prompt(self, shots: list[SurveyShot]) -> str:
        intr = shots[0].obs.intrinsics
        return SURVEY_PROMPT.format(n=len(shots), width=intr.width, height=intr.height)

    # --------------------------------------------------------------- query
    def query(self, shots: list[SurveyShot]) -> SurveyResult:
        if not shots:
            raise ValueError("no shots to survey")
        try:
            content, dt = self.client.ask_images([s.obs.rgb for s in shots],
                                                 [s.label(shots[0].heading) for s in shots],
                                                 self.prompt(shots))
        except Exception as e:  # server down, timeout, HTTP error
            self.errors += 1
            res = SurveyResult(found=False, error=f"{type(e).__name__}: {e}")
            return self._finish(res)

        finish = getattr(self.client, "last_finish_reason", None)
        usage = getattr(self.client, "last_usage", {}) or {}
        if finish == "length":
            self.truncated += 1
        try:
            raw = _extract_json(content)
        except ValueError as e:
            self.errors += 1
            res = SurveyResult(found=False, latency=dt, finish_reason=finish,
                               completion_tokens=usage.get("completion_tokens"),
                               error=f"unparseable answer ({e})")
            return self._finish(res)

        self.queries += 1
        res = self.interpret(shots, raw)
        res.latency, res.finish_reason = dt, finish
        res.completion_tokens = usage.get("completion_tokens")
        return self._finish(res)

    def _finish(self, res: SurveyResult) -> SurveyResult:
        self.last_result = res
        if self.verbose:
            if res.error:
                print(f"    [survey] failed: {res.error}")
            elif not res.found:
                print(f"    [survey] goal not found ({res.reason!r})")
            else:
                where = (f"at ({res.detection.position[0]:.2f}, {res.detection.position[1]:.2f})"
                         if res.detection is not None else "unranged")
                print(f"    [survey] photo {res.photo} px {res.pixel} heading "
                      f"{np.degrees(res.heading):.0f} deg via {res.heading_source}, {where}, "
                      f"conf={res.confidence}: {res.reason}")
        return res

    # ------------------------------------------------------------ interpret
    def interpret(self, shots: list[SurveyShot], raw: dict) -> SurveyResult:
        """Turn the model's JSON into the most trustworthy goal estimate."""
        conf = _to_float(raw.get("confidence"))
        reason = str(raw.get("reason", ""))
        res = SurveyResult(found=False, confidence=conf, reason=reason, raw=raw)

        photo = _to_int(raw.get("photo"))
        if photo is not None and not (0 <= photo < len(shots)):
            res.notes.append(f"photo {photo} out of range")
            photo = None
        # The box is normalised within the CHOSEN photo, so it can only be
        # turned into pixels once we know which photo that is.
        pixel = None
        if photo is not None and shots[photo].obs is not None:
            intr = shots[photo].obs.intrinsics
            pixel = _goal_pixel({"bbox_2d": raw.get("bbox_2d")}, intr.width, intr.height)
        if pixel is None:
            pixel = _pixel(raw.get("goal_px"))           # legacy pixel answers
        model_heading = _to_float(raw.get("heading_deg"))

        found_flag = raw.get("goal_found")
        found = (_to_bool(found_flag) if found_flag is not None
                 else (photo is not None or model_heading is not None))
        if not found:
            return res
        if conf is not None and conf < self.min_confidence:
            res.notes.append(f"confidence {conf} below {self.min_confidence}")
            return res

        res.found = True
        res.photo, res.pixel = photo, pixel

        if photo is not None and pixel is not None:
            shot = shots[photo]
            det = self._ranger._to_detection(
                shot.obs, {"goal_found": True, "goal_horizontal_px": pixel[0],
                           "goal_vertical_px": pixel[1], "goal_confidence": 1.0},
                robot_yaw=shot.heading)
            if det is not None and det.pixels > 0:
                res.detection = det
                d = det.position[:2] - shot.robot_xy
                res.heading, res.heading_source = float(np.arctan2(d[1], d[0])), "range"
                return res
            # No depth under the pixel: the column of the pixel still gives an
            # exact bearing inside that photo, which beats the model's guess.
            res.heading = pixel_heading(shot, pixel[0])
            res.heading_source = "pixel"
            return res

        if model_heading is not None:
            res.heading = float(_wrap(np.radians(model_heading)))
            res.heading_source = "model"
            return res
        if photo is not None:
            res.heading, res.heading_source = shots[photo].heading, "photo"
            return res

        res.found = False
        res.notes.append("goal_found without any photo, pixel or heading")
        return res

    def __repr__(self):
        return (f"<LlmSurvey {self.client.model} shots={self.n_shots} "
                f"queries={self.queries} errors={self.errors} truncated={self.truncated}>")


def _pixel(v) -> tuple[int, int] | None:
    if isinstance(v, dict):
        v = [v.get("horizontal_px", v.get("u", v.get("x"))),
             v.get("vertical_px", v.get("v", v.get("y")))]
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    u, w = _to_int(v[0]), _to_int(v[1])
    return None if u is None or w is None else (u, w)


def pixel_heading(shot: SurveyShot, u: float) -> float:
    """World heading of the ray through image column u of a shot.

    Image right is the robot's right, i.e. a SMALLER world heading, so the
    offset is subtracted.
    """
    intr = shot.obs.intrinsics
    return float(_wrap(shot.heading - np.arctan2(u - intr.cx, intr.fx)))
