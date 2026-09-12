"""The local VLM as a scene-reasoning goal detector.

This closes the loop the user asked for: the robot *looks* at its head-camera
frame, asks the local LLM (llama-server, OpenAI-compatible) to **reason about
what it sees and where the goal is**, the model answers with a natural-language
reasoning plus the goal's pixel position, and we convert that pixel into a world
position by fusing it with the registered depth image -- exactly the way
`YoloDetector` ranges a detection box. The result is a `Detection` on the same
contract as the colour detector, so it drops into `VisualNavigator` unchanged:
the depth camera still builds the obstacle map, A* still plans the route, and
the robot drives to *where the LLM thinks the goal is*.

    det = LlmGoalDetector(base_url="http://localhost:8080/v1")
    dets = det(obs, robot_yaw=bot.yaw, t=bot.time)   # -> [Detection]

The model here is a *slow, deep* look, not a per-frame classifier: one query
takes on the order of ten seconds of wall time (it runs its full internal
reasoning chain before emitting JSON). Two consequences drive the design:

  * queries are gated on **sim time** (`t`), not wall clock. Headless MuJoCo
    races far ahead of realtime, and gating on the wall clock would fire
    hundreds of ten-second calls per simulated second. Callers that have no
    sim time (offline tests) fall back to a wall-clock gate.
  * between queries the last good detection is cached and re-returned, so the
    fast servo/planning loop never blocks on the model.
"""
from __future__ import annotations

import base64
import io
import json
import re
import time

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .perception import Detection, Observation  # noqa: F401  (type hints)


DEFAULT_BASE_URL = "http://localhost:8080/v1"
DEFAULT_MODEL = "Qwen/Qwen3.8-27B"


# --------------------------------------------------------------------------- prompt
PROMPT_TEMPLATE = """You are the visual cortex of a balancing two-wheeled robot on an indoor obstacle course.
You see ONE frame from its head-mounted camera. The image is {width} px wide and {height} px tall
(horizontal_px: 0 = left edge; vertical_px: 0 = top edge).

GOAL: a tall RED cylinder standing on the floor. That is the one object you must locate precisely.
OTHERS: orange barriers (low walls), blue pillars, a grey checkerboard floor, a hazy sky. Ignore the
robot's own white arms if they enter the frame.

Reason about what you see, then answer with ONLY a single-line compact JSON object -- no prose,
no markdown fences, no newlines, nothing after the closing brace. Keys, in this order:
{{
  "scene": "<one short sentence: what the scene is overall>",
  "goal_found": true/false,
  "goal_horizontal_px": <int or null>,
  "goal_vertical_px": <int or null>,
  "goal_confidence": <0.0-1.0>,
  "goal_depth_cue": "<one phrase: near/mid/far and left/centre/right, e.g. 'mid distance, slightly left of centre, rising above the barrier'>",
  "other_objects": [ {{ "label": "barrier|pillar|other", "horizontal_px": <int or null>, "vertical_px": <int or null>, "note": "<short>" }} ],
  "movement": "<one sentence: what the robot should do to reach the goal, e.g. 'rotate ~10 deg left, then advance while keeping the column in view'>"
}}
Put the GOAL's vertical axis at goal_horizontal_px/goal_vertical_px (its visible centre, not its base).
If you are not confident the red goal is present, set goal_found=false and null the two pixel fields."""


def _extract_json(text: str):
    """First JSON object in the text, tolerating fences and surrounding chatter.

    Uses the real JSON decoder from each candidate `{` rather than counting
    braces: a brace counter ends early on a `}` inside a string value (e.g. a
    scene description that mentions one), and then the whole query is lost.
    """
    if text is None:
        raise ValueError("empty response")
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass                     # fall through to the scan below
    decoder = json.JSONDecoder()
    start = t.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in response: {t[:120]!r}")
    while start >= 0:
        try:
            obj, _ = decoder.raw_decode(t, start)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        start = t.find("{", start + 1)
    raise ValueError(f"no decodable JSON object in response: {t[:120]!r}")


def _to_int(x):
    try:
        return int(round(float(x)))
    except (TypeError, ValueError):
        return None


def _to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _png_data_url(rgb: np.ndarray) -> str:
    """An RGB array as a base64 PNG data URL, the form the server accepts."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


_TRUE = {"true", "yes", "y", "1"}
_FALSE = {"false", "no", "n", "0", "null", "none", ""}


def _to_bool(x) -> bool:
    """JSON-ish truth. A model that writes "false" as a string must not be
    read as True, which is what plain truthiness does to any non-empty str."""
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    if isinstance(x, (int, float)):
        return x != 0
    v = str(x).strip().lower()
    if v in _TRUE:
        return True
    if v in _FALSE:
        return False
    return False


# --------------------------------------------------------------------------- client
class LLMClient:
    """Minimal OpenAI-compatible chat client for a local llama-server."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 max_tokens: int = 2048, temperature: float = 0.1,
                 timeout: float = 120.0, verbose: bool = False):
        if requests is None:
            raise RuntimeError("the 'requests' package is required for LLMGoalDetector")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self.verbose = verbose
        self.last_finish_reason: str | None = None
        self.last_usage: dict = {}
        self.last_reasoning_chars = 0

    def ask_image(self, rgb: np.ndarray, prompt: str) -> tuple[str, float]:
        """Send one image + prompt; return (content_text, latency_seconds)."""
        return self._chat([
            {"type": "text", "text": prompt + "\n\n/no_think"},
            {"type": "image_url", "image_url": {"url": _png_data_url(rgb)}},
        ])

    def ask_images(self, images, labels, prompt: str) -> tuple[str, float]:
        """Several images in ONE request, each preceded by its own text label.

        The label is what ties a photo to where it was taken ("Photo 3: robot
        at (1.0, 2.0), camera heading 135 deg"), so the model can reason across
        views instead of looking at each in isolation.
        """
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        content = [{"type": "text", "text": prompt + "\n\n/no_think"}]
        for rgb, label in zip(images, labels):
            content.append({"type": "text", "text": str(label)})
            content.append({"type": "image_url",
                            "image_url": {"url": _png_data_url(rgb)}})
        return self._chat(content)

    def _chat(self, content: list) -> tuple[str, float]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            # The Qwen build otherwise spends the whole budget on internal
            # reasoning and returns empty content; force straight-to-answer.
            # It still reasons on hard multi-image prompts despite this --
            # see `last_reasoning_chars` -- so give those a large max_tokens.
            "enable_thinking": False,
        }
        t0 = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", json=payload,
                          timeout=self.timeout)
        dt = time.time() - t0
        if r.status_code != 200:
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        choice = data["choices"][0]
        # "length" means the token cap cut the answer off -- the JSON is
        # incomplete and will fail to parse. Recorded so the caller can count it.
        self.last_finish_reason = choice.get("finish_reason")
        self.last_usage = data.get("usage", {}) or {}
        message = choice.get("message", {}) or {}
        self.last_reasoning_chars = len(message.get("reasoning_content") or "")
        content = message.get("content") or ""
        if self.verbose:
            print(f"    [llm] {dt:.1f}s  usage={self.last_usage.get('completion_tokens', '?')} tok"
                  f"  finish={self.last_finish_reason}"
                  f"{f'  reasoning={self.last_reasoning_chars} chars' if self.last_reasoning_chars else ''}")
        return content, dt


# --------------------------------------------------------------------------- reasoner
class LlmGoalDetector:
    """Callable detector: Observation -> list[Detection], goal located by the VLM.

    Satisfies the same contract as `perception.detect` and `YoloDetector`, so
    `VisualNavigator(detector=LlmGoalDetector(...))` works with no other change.
    `t` (sim seconds) gates how often the model is actually queried; it is
    optional for callers that don't have a clock.
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 goal_label: str = "target", query_period: float = 3.0,
                 min_confidence: float = 0.40, patch_radius: int = 8,
                 min_height: float = 0.08, max_height: float = 2.5,
                 min_range: float = 0.4, max_range: float = 12.0,
                 self_radius: float = 0.55, ground_z: float = 0.0,
                 max_tokens: int = 2048, temperature: float = 0.1,
                 timeout: float = 120.0, verbose: bool = False):
        self.client = LLMClient(base_url, model, max_tokens, temperature, timeout, verbose)
        self.goal_label = goal_label
        self.query_period = max(0.5, float(query_period))
        self.min_confidence = min_confidence
        self.patch_radius = patch_radius
        self.min_height, self.max_height = min_height, max_height
        self.min_range, self.max_range = min_range, max_range
        self.self_radius, self.ground_z = self_radius, ground_z
        self.verbose = verbose

        self._last_query: float | None = None   # sim (or wall) time of last query
        self._clock = "sim"                     # or "wall" when t is never given
        self._cache: list[Detection] = []
        self.last_reasoning: dict | None = None
        self.last_latency: float | None = None
        self.queries = 0
        self.errors = 0
        self.truncated = 0

    # --------------------------------------------------------------- cadence
    def _due(self, t: float | None) -> bool:
        clock = "wall" if t is None else "sim"
        now = time.monotonic() if t is None else float(t)
        if self._last_query is None or clock != self._clock:
            self._clock = clock
            return True
        if now < self._last_query:
            # The clock ran backwards: a sim reset (viewer R, BracketBot.reset)
            # zeroes t. Without this the detector stays silent until t climbs
            # back past the pre-reset query time -- tens of seconds blind.
            return True
        return (now - self._last_query) >= self.query_period

    # ------------------------------------------------------------------ call
    def __call__(self, obs: Observation, robot_yaw: float = 0.0, t: float | None = None,
                 **_) -> list[Detection]:
        if not self._due(t):
            return self._cache

        # Stamp the ATTEMPT, not the success. Stamping only on success means a
        # down server is retried on every sense tick, and each retry blocks the
        # whole sim loop for up to `timeout` seconds.
        self._last_query = t if t is not None else time.monotonic()

        prompt = PROMPT_TEMPLATE.format(width=obs.intrinsics.width,
                                        height=obs.intrinsics.height)
        det = None
        try:
            content, dt = self.client.ask_image(obs.rgb, prompt)
            if getattr(self.client, "last_finish_reason", None) == "length":
                self.truncated += 1
            reason = _extract_json(content)
            det = self._to_detection(obs, reason, robot_yaw)
            self.queries += 1
            self.last_latency = dt
        except Exception as e:  # model down, timeout, or bad JSON: keep last good
            self.errors += 1
            if self.verbose:
                print(f"    [llm] query failed ({type(e).__name__}: {e}); reusing cache")
            return self._cache

        self.last_reasoning = reason
        self._cache = [det] if det is not None else []
        if self.verbose:
            scene = str(reason.get("scene", ""))
            mv = str(reason.get("movement", ""))
            if det is not None:
                print(f"    [llm] t={self._clock}={self._last_query:.1f} FOUND "
                      f"{self.goal_label} @ px({_to_int(reason.get('goal_horizontal_px'))},"
                      f"{_to_int(reason.get('goal_vertical_px'))}) "
                      f"conf={_to_float(reason.get('goal_confidence'))}")
            print(f"          scene : {scene}")
            if mv:
                print(f"          move  : {mv}")
        return self._cache

    # -------------------------------------------------------------- pixel->world
    def _to_detection(self, obs, reason, robot_yaw):
        if not _to_bool(reason.get("goal_found")):
            if self.verbose:
                print(f"    [llm] goal not found: {reason.get('scene', '')!r}")
            return None
        u = _to_int(reason.get("goal_horizontal_px"))
        v = _to_int(reason.get("goal_vertical_px"))
        conf = _to_float(reason.get("goal_confidence"))
        if conf is None:
            conf = 1.0
        if u is None or v is None or conf < self.min_confidence:
            if self.verbose:
                print(f"    [llm] low-confidence/invalid goal (conf={conf}, px={u},{v})")
            return None

        h, w = obs.depth.shape
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        r = self.patch_radius
        u0, u1 = max(u - r, 0), min(u + r, w - 1)
        v0, v1 = max(v - r, 0), min(v + r, h - 1)

        patch_pts = obs.points[v0:v1 + 1, u0:u1 + 1]
        patch_d = obs.depth[v0:v1 + 1, u0:u1 + 1]
        z = patch_pts[..., 2]
        good = (np.isfinite(patch_d) & (patch_d > self.min_range) &
                (patch_d < self.max_range) &
                (z > self.min_height) & (z < self.max_height))
        good &= (np.linalg.norm(patch_pts[..., :2] - obs.robot_xy, axis=-1)
                 > self.self_radius)

        if int(good.sum()) >= 8:
            # Depth fusion: the median of the goal's visible surface, same as
            # YoloDetector -- robust to floor/sky seen past the column.
            centre = np.median(patch_pts[good], axis=0)
            pixels = int(good.sum())
        else:
            # No usable depth under the pixel (e.g. goal seen over a barrier):
            # fire the ray through the pixel and drop it to the ground plane.
            centre, pixels = self._ray_to_ground(obs, u, v)

        delta = centre[:2] - obs.robot_xy
        dist = float(np.linalg.norm(centre - obs.cam_pos))
        return Detection(
            label=self.goal_label,
            position=np.asarray(centre, float),
            distance=dist,
            bearing=float(np.arctan2(delta[1], delta[0]) - robot_yaw),
            extent=np.zeros(3),
            pixels=pixels,
            bbox=(int(u0), int(v0), int(u1), int(v1)))

    def _ray_to_ground(self, obs, u, v):
        """World point where the camera ray through pixel (u, v) hits z = ground_z."""
        intr = obs.intrinsics
        x = (u - intr.cx) / intr.fx
        y = -(v - intr.cy) / intr.fy
        ray_cam = np.array([x, y, -1.0])          # optical axis is -z of the cam
        ray = obs.cam_mat @ ray_cam
        ray /= np.linalg.norm(ray)
        if ray[2] >= -1e-6:                        # not pointing down to the floor
            t = 2.0
            centre = obs.cam_pos + ray * t
            return centre, 0
        t = (self.ground_z - obs.cam_pos[2]) / ray[2]
        t = float(np.clip(t, 0.5, self.max_range))
        return obs.cam_pos + ray * t, 0

    def __repr__(self):
        return (f"<LlmGoalDetector {self.client.model} every={self.query_period}s "
                f"queries={self.queries} errors={self.errors} truncated={self.truncated}>")
