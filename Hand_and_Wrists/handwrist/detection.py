"""Find any object by what it is with the head RGB-D camera, and measure it for a grasp.

This is the estimator `Pick` uses by default: it sees "the ball" or "the remote"
because they look like a ball and a remote. `vision.CameraEstimator` -- which finds
the seven living-room items by calibrated colour windows -- is kept for
comparison. This keeps its geometry -- the top-face outline, the silhouette
width, the handle -- and replaces the colour step with an open-vocabulary
detector, so the grasp planner gets the same ObjectEstimate for anything that
can be named:

  1. A box around the object: YOLO-World (pretrained, any noun phrase) when it
     is confident, otherwise the vision LLM. Each catalogue object is asked
     for by the names a detector knows it by (QUERIES: a mug is also a "cup").
     A colour in the request ("the red mug") only ranks the candidates: the
     right colour beats a more confident box of the wrong one, but an object
     is never missed for being described in the wrong colour.
  2. The depth pixels inside the box, lifted to world xyz by `observe()`.
  3. What the object stands on, from a ray cast down through the middle of the
     box (as `GraspPlanner.support` does), and only the points above that
     surface and near the box's centre: the table top and the wall behind the
     object are inside the box too.
  4. The 3-D cluster nearest the centre, then CameraEstimator's geometry.

    see = DetectionEstimator()
    est = see(bot, CATALOGUE["mug"])                              # ObjectEstimate or None
    see.aliases["mug"] = "red mug"                                # prefer the red one
    see.last_box                                                  # what the detector saw
"""
from __future__ import annotations

import re
from dataclasses import replace

import mujoco
import numpy as np

from vision_sim.perception import rgb_to_hsv

from .objects import truth_estimate
from .places import _geom_top
from .vision import CAMERA, CameraEstimator, Sighting, clusters

# A query competes only with big things. Near-synonyms in the vocabulary take
# the box from the name asked for: with "cup" and "bottle" also listed, a mug
# asked for as "mug" came back labelled "cup" and the can as "bottle".
DISTRACTORS = ("person", "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf",
               "door", "potted plant", "lamp", "television", "refrigerator", "sink", "wall",
               "basket")
YOLO_CACHE = 3          # detectors kept, one per vocabulary
# A box a distractor claims more confidently, overlapping this much, is the
# distractor's: the laundry basket scores 0.30 as a "cardboard box" but higher
# as a basket, and handing the basket to the grasp planner as the box fails.
CLAIMED_IOU = 0.5
# The zoom pass: overlapping square crops half the image wide (320 px in the
# 640 x 480 head camera), each looked at ZOOM times bigger. Keys on the floor at
# pick range are ~20 x 10 px there, too small for YOLO-World in the whole
# frame; twice as big, they are found.
TILE, ZOOM = 320, 2
# every geom group but 2, the household looks (and robot visuals): a support
# is what an object rests on, not the picture of it
_NOT_LOOKS = np.array([1, 1, 0, 1, 1, 1], np.uint8)

# What to ask the detector for, per catalogue object: its own name and what an
# off-the-shelf detector is likelier to call it. A box under any of them is
# the object. Measured on the head camera's renders of the scene models.
QUERIES = {
    "mug": ("mug", "cup"),
    "can": ("soda can", "can"),
    "bottle": ("bottle", "water bottle"),
    "remote": ("remote control", "tv remote"),
    "ball": ("tennis ball", "ball"),
}


def _hue(centre, width, sat=0.30, val=0.18):
    def match(hsv):
        d = np.abs((hsv[..., 0] - centre + 180.0) % 360.0 - 180.0)
        return (d <= width) & (hsv[..., 1] >= sat) & (hsv[..., 2] >= val)
    return match


# A colour word -> which pixels are that colour, broadly, as a person means it.
# Deliberately loose: it ranks candidates the detector found, it never finds one.
COLOURS = {
    "red": _hue(0, 22), "orange": _hue(28, 12), "yellow": _hue(56, 16), "gold": _hue(42, 12, 0.35, 0.40),
    "green": _hue(115, 45), "teal": _hue(178, 18), "cyan": _hue(188, 18), "blue": _hue(220, 30),
    "purple": _hue(275, 25), "violet": _hue(275, 25), "pink": _hue(325, 25),
    "brown": lambda hsv: (_hue(28, 20, 0.30, 0.10)(hsv) & (hsv[..., 2] <= 0.60)),
    "white": lambda hsv: (hsv[..., 1] <= 0.20) & (hsv[..., 2] >= 0.70),
    "black": lambda hsv: hsv[..., 2] <= 0.22,
    "grey": lambda hsv: (hsv[..., 1] <= 0.20) & (hsv[..., 2] > 0.22) & (hsv[..., 2] < 0.75),
    "silver": lambda hsv: (hsv[..., 1] <= 0.20) & (hsv[..., 2] >= 0.45),
}
COLOURS["gray"] = COLOURS["grey"]
COLOURS["golden"] = COLOURS["brass"] = COLOURS["gold"]
_FILLER = {"coloured", "colored", "colour", "color", "dark", "light", "bright"}


def split_colour(text: str) -> tuple[str | None, str]:
    """'the Red mug' -> ('red', 'the mug'): the first colour word, and the rest."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    colour = next((w for w in words if w in COLOURS), None)
    rest = [w for w in words if w != colour and w not in _FILLER]
    return colour, " ".join(rest)


def colour_fraction(rgb, box, colour: str) -> float:
    """Share of the middle of `box` (inner 60 %) that is `colour`."""
    h, w = rgb.shape[:2]
    u0, v0, u1, v1 = box
    du, dv = 0.2 * (u1 - u0), 0.2 * (v1 - v0)
    u0, u1 = int(max(u0 + du, 0)), int(min(u1 - du, w - 1)) + 1
    v0, v1 = int(max(v0 + dv, 0)), int(min(v1 - dv, h - 1)) + 1
    if u1 <= u0 or v1 <= v0:
        return 0.0
    return float(COLOURS[colour](rgb_to_hsv(rgb[v0:v1, u0:u1])).mean())


def _iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area = lambda r: max(r[2] - r[0], 0) * max(r[3] - r[1], 0)  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1e-9)


def _nms(hits, iou=0.5) -> list[dict]:
    """Most confident first, dropping any box that repeats a kept one."""
    kept = []
    for h in sorted(hits, key=lambda h: -h["confidence"]):
        if all(_iou(h["box"], k["box"]) < iou for k in kept):
            kept.append(h)
    return kept


def _tiles(w, h, tile=TILE):
    """Top-left corners of overlapping tiles (half a tile apart) covering w x h."""
    def starts(n):
        if n <= tile:
            return [0]
        k = int(np.ceil((n - tile) / (tile / 2)))
        return [int(round(i * (n - tile) / k)) for i in range(k + 1)]
    return [(x, y) for y in starts(h) for x in starts(w)]


def rank_by_colour(rgb, hits, colour: str) -> list[dict]:
    """Candidates best first, with colour as the tie-breaker it should be: a
    box whose middle is a quarter or more the named colour keeps its full
    confidence, one with none of it keeps 35 %."""
    for h in hits:
        h["colour_match"] = round(colour_fraction(rgb, h["box"], colour), 3)
    return sorted(hits, key=lambda h: -h["confidence"] * (0.35 + 0.65 * min(1.0, h["colour_match"] / 0.25)))

LLM_BOX_PROMPT = (
    "Find {name} in this photo from a home robot's head camera. The image is {width} px wide and {height}"
    " px tall. Ignore the robot's own white arms.\n"
    'Answer with ONLY a single-line compact JSON object -- no prose, no markdown fences:\n'
    '{{"found": true/false, "bbox_2d": [x1, y1, x2, y2] or null, "confidence": <0.0-1.0>}}\n'
    'bbox_2d is a tight box around {name}, with coordinates normalised to 0-1000 across the image width '
    '(x) and height (y). If it is not clearly visible, set found=false.'
)


class DetectionEstimator(CameraEstimator):
    """Estimator for `Pick` that finds `spec.name` with an open-vocabulary detector."""

    BACKENDS = ("auto", "yolo", "llm")

    def __init__(self, backend: str = "auto", weights=None, conf: float = 0.05,
                 imgsz: int = 640, yolo_accept: float = 0.30,
                 llm_url: str = "http://localhost:8080/v1",
                 llm_model: str = "Qwen/Qwen3.8-27B", llm_min_confidence: float = 0.4,
                 camera=CAMERA, width=640, height=480, max_range=4.0, min_points=15,
                 self_radius=0.35):
        if backend not in self.BACKENDS:
            raise ValueError(f"backend must be one of {self.BACKENDS}, not {backend!r}")
        # colour windows are not used; passing any makes the parent skip loading them
        super().__init__(camera=camera, width=width, height=height, max_range=max_range,
                         min_points=min_points, self_radius=self_radius, colours={"-": None})
        self.backend, self.weights = backend, weights
        self.conf, self.imgsz, self.yolo_accept = conf, imgsz, yolo_accept
        self.llm_url, self.llm_model = llm_url, llm_model
        self.llm_min_confidence = llm_min_confidence
        self.last_box: dict | None = None
        # object name -> what to ask the detector for instead: sim props often do
        # not look like their names but can be
        # described ("small object on the floor")
        self.aliases: dict[str, str] = {}
        self.yolo_error: str | None = None
        # set when the vision LLM could not be reached; it is not asked again,
        # so a robot with no LLM server does not wait out a timeout every look
        self.llm_error: str | None = None
        self._yolo: dict[tuple, object] = {}      # vocabulary -> detector, oldest first
        self._bot = None

    def __call__(self, bot, spec):
        self._bot = bot                      # for the support ray cast
        return super().__call__(bot, spec)

    # ---------------------------------------------------------------- boxes
    def yolo(self, names):
        """A YOLO-World detector for `names`, with only big things as distractors.

        Setting a vocabulary embeds every name with CLIP (a second or two), so
        detectors are cached per vocabulary.
        """
        from vision_sim.yolo_detector import DEFAULT_WEIGHTS, YoloDetector
        names = list(dict.fromkeys(names))
        key = tuple(names)
        if key not in self._yolo:
            if len(self._yolo) >= YOLO_CACHE:
                self._yolo.pop(next(iter(self._yolo)))
            extra = [c for c in DISTRACTORS if c not in names]
            self._yolo[key] = YoloDetector(self.weights or DEFAULT_WEIGHTS, target=names[0],
                                           classes=names[1:] + extra, conf=self.conf,
                                           imgsz=self.imgsz)
        return self._yolo[key]

    def yolo_boxes(self, rgb, names, zoom: bool = False) -> list[dict]:
        """YOLO boxes for `names`, most confident first.

        A box a distractor claims more confidently is left out (CLAIMED_IOU).
        zoom: also look at overlapping tiles blown up ZOOM times, for things
        only a few pixels across; their boxes come back in full-frame pixels
        with source "yolo-zoom"."""
        from PIL import Image
        det = self.yolo(names)
        h, w = rgb.shape[:2]
        tile = w * TILE // 640                      # the same six tiles at any resolution
        views = [(rgb, 0, 0, 1)]
        if zoom:
            for x, y in _tiles(w, h, tile):
                crop = Image.fromarray(np.ascontiguousarray(rgb[y:y + tile, x:x + tile]))
                big = crop.resize((crop.width * ZOOM, crop.height * ZOOM), Image.BICUBIC)
                views.append((np.asarray(big), x, y, ZOOM))
        wanted, out, claimed = set(names), [], []
        for img, ox, oy, s in views:
            # Ultralytics reads a raw array as BGR; MuJoCo renders RGB.
            result = det._predict(np.ascontiguousarray(img[..., ::-1]))
            for box in result.boxes:
                name = det.names[int(box.cls)]
                u0, v0, u1, v1 = (int(round(c / s + o)) for c, o in
                                  zip(box.xyxy[0].tolist(), (ox, oy, ox, oy)))
                hit = {"name": name, "confidence": float(box.conf), "box": (u0, v0, u1, v1),
                       "source": "yolo" if s == 1 else "yolo-zoom"}
                (out if name in wanted else claimed).append(hit)
        out = [o for o in out if not any(c["confidence"] > o["confidence"]
                                         and _iou(c["box"], o["box"]) >= CLAIMED_IOU
                                         for c in claimed)]
        return _nms(out)

    def llm_box(self, rgb, name) -> dict | None:
        from vision_sim.llm_reasoner import LLMClient, _extract_json, _to_bool
        h, w = rgb.shape[:2]
        client = LLMClient(self.llm_url, self.llm_model, max_tokens=200, timeout=120)
        text, _ = client.ask_image(rgb, LLM_BOX_PROMPT.format(name=name, width=w, height=h))
        try:
            raw = _extract_json(text)
        except ValueError:
            return None
        box = raw.get("bbox_2d")
        try:
            conf = float(raw.get("confidence") or 0.0)
            x0, y0, x1, y1 = (float(v) for v in box)
        except (TypeError, ValueError):
            return None
        if not _to_bool(raw.get("found")) or conf < self.llm_min_confidence:
            return None
        u0, u1 = sorted((int(x0 * w / 1000), int(x1 * w / 1000)))
        v0, v1 = sorted((int(y0 * h / 1000), int(y1 * h / 1000)))
        return {"name": name, "confidence": conf, "box": (u0, v0, u1, v1), "source": "llm"}

    def find(self, rgb, name) -> dict | None:
        """The best box for `name`, found by what the object is.

        `name` may carry a colour ("red mug"): the object is looked for by its
        noun, under every name in QUERIES, and the colour only ranks what is
        found. With nothing confident in the whole frame, it looks again with
        the zoom pass. auto: a confident YOLO box, else the vision LLM's, else
        YOLO's weak one."""
        colour, noun = split_colour(name)
        noun = noun or name
        weak = None
        if self.backend in ("auto", "yolo"):
            hits = []
            for zoom in (False, True):
                try:
                    hits = self.yolo_boxes(rgb, list(QUERIES.get(noun, (noun,))), zoom=zoom)
                except ImportError as e:
                    if self.backend == "yolo":
                        raise
                    self.yolo_error, hits = str(e), []
                    break
                if colour is not None:
                    hits = rank_by_colour(rgb, hits, colour)
                strong = [h for h in hits if h["confidence"] >= self.yolo_accept]
                if strong:
                    return strong[0]
            if hits and self.backend == "yolo":
                return hits[0]
            weak = hits[0] if hits else None
        if self.backend in ("auto", "llm") and self.llm_error is None:
            try:
                return self.llm_box(rgb, name) or weak
            except (OSError, RuntimeError, ValueError) as e:     # no server, a timeout, no requests
                if self.backend == "llm":
                    raise
                self.llm_error = f"{type(e).__name__}: {e}"
        return weak

    # ---------------------------------------------------------------- depth
    def support_z(self, xy, z_from) -> float:
        """Height of the fixed surface under `xy`, looking through loose items."""
        m, d = self._bot.model, self._bot.data
        start = np.array([xy[0], xy[1], z_from], float)
        down = np.array([0.0, 0.0, -1.0])
        hit = np.zeros(1, np.int32)
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, start, down, _NOT_LOOKS, 1, -1, hit)
            g = int(hit[0])
            if dist < 0 or g < 0:
                return 0.0
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                return float(start[2] - dist)
            if m.body_weldid[m.geom_bodyid[g]] != 0:     # the object, another item, the robot
                start = start + down * (dist + 0.002)
                continue
            return _geom_top(m, d, g)
        return 0.0

    def object_points(self, obs, box):
        """(points, pixels, support height) of the object in `box`, or None."""
        h, w = obs.depth.shape
        u0, v0, u1, v1 = box
        u0, u1, v0, v1 = max(u0, 0), min(u1, w - 1), max(v0, 0), min(v1, h - 1)
        if u1 <= u0 or v1 <= v0:
            return None
        patch = obs.points[v0:v1 + 1, u0:u1 + 1]
        ok = obs.valid[v0:v1 + 1, u0:u1 + 1] & (obs.depth[v0:v1 + 1, u0:u1 + 1] < self.max_range)
        ok &= np.linalg.norm(patch[..., :2] - obs.robot_xy, axis=-1) > self.self_radius
        if ok.sum() < self.min_points:
            return None
        rows, cols = np.nonzero(ok)
        pts = patch[ok]
        pix = np.stack([rows + v0, cols + u0], axis=1)
        bh, bw = v1 - v0 + 1, u1 - u0 + 1
        core = ((rows >= 0.25 * bh) & (rows <= 0.75 * bh)
                & (cols >= 0.25 * bw) & (cols <= 0.75 * bw))
        middle = pts[core] if core.sum() >= 3 else pts
        c0 = np.median(middle[:, :2], axis=0)
        top = float(np.percentile(middle[:, 2], 95))
        support = (self.support_z(c0, top + 0.05) if self._bot is not None
                   else float(np.percentile(pts[:, 2], 2)))
        dist = float(np.linalg.norm(np.array([c0[0], c0[1], top]) - obs.cam_pos))
        radius = float(np.clip(0.6 * max(bw, bh) * dist / obs.intrinsics.fy, 0.06, 0.35))
        keep = ((pts[:, 2] > support + 0.004)
                & (np.linalg.norm(pts[:, :2] - c0, axis=1) < radius))
        if keep.sum() < self.min_points:
            return None
        pts, pix = pts[keep], pix[keep]
        labels = clusters(pts)
        best, best_d = None, np.inf
        for k in np.unique(labels):
            sel = labels == k
            if sel.sum() < self.min_points:
                continue
            gap = float(np.linalg.norm(np.median(pts[sel, :2], axis=0) - c0))
            if gap < best_d:
                best, best_d = sel, gap
        if best is None:
            return None
        return pts[best], pix[best], support

    def sight(self, obs, spec):
        found = self.find(obs.rgb, self.aliases.get(spec.name, spec.name))
        self.last_box = found
        if found is None:
            return None
        got = self.object_points(obs, found["box"])
        if got is None:
            return None
        pts, pix, _ = got
        return Sighting(spec.name, pts, pix, float(np.linalg.norm(pts.mean(0) - obs.cam_pos)))

    def locate(self, obs, found) -> dict | None:
        """World centre, size and support of a detected box (for listing what is in view)."""
        got = self.object_points(obs, found["box"])
        if got is None:
            return None
        pts, _, support = got
        lo, hi = pts.min(0), pts.max(0)
        return {"center": (lo + hi) / 2, "size": hi - lo, "bottom_z": float(lo[2]),
                "support_z": float(support)}


class TruthByName:
    """Ground truth for a loose body named like the object: debugging and tests."""

    def __call__(self, bot, spec):
        m = bot.model
        key = spec.name.strip().lower().replace(" ", "_")
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, key)
        if b < 0:
            return None
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                 for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        names = [n for n in names if n]
        grasp = f"{key}_body" if f"{key}_body" in names else (names[0] if names else None)
        if grasp is None:
            return None
        handle = (f"{key}_handle" if spec.handle_geom is not None and f"{key}_handle" in names
                  else None)
        try:
            est = truth_estimate(m, bot.data, replace(spec, grasp_geom=grasp, handle_geom=handle))
        except (KeyError, ValueError):
            return None
        est.name = spec.name
        return est
