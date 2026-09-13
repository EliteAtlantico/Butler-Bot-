"""Find any object by name with the head RGB-D camera, and measure it for a grasp.

`vision.CameraEstimator` finds the seven living-room items by calibrated colour.
This keeps its geometry -- the top-face outline, the silhouette width, the
handle -- and replaces the colour step with an open-vocabulary detector, so the
grasp planner gets the same ObjectEstimate for anything that can be named:

  1. A box around the object: YOLO-World (pretrained, any noun phrase) when it
     is confident, otherwise the vision LLM. On these renders YOLO finds a mug
     as "cup" at 0.56 but as "mug" hardly at all, and misses small flat things;
     the vision LLM boxes both to within a few millimetres of the truth.
  2. The depth pixels inside the box, lifted to world xyz by `observe()`.
  3. What the object stands on, from a ray cast down through the middle of the
     box (as `GraspPlanner.support` does), and only the points above that
     surface and near the box's centre: the table top and the wall behind the
     object are inside the box too.
  4. The 3-D cluster nearest the centre, then CameraEstimator's geometry.

    see = DetectionEstimator()
    est = see(bot, ObjectSpec("mug", "", handle_geom="handle"))   # ObjectEstimate or None
    see.last_box                                                  # what the detector saw
"""
from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np

from .objects import truth_estimate
from .places import _geom_top
from .vision import CAMERA, CameraEstimator, Sighting, clusters

# A query competes only with big things. Near-synonyms in the vocabulary take
# the box from the name asked for: with "cup" and "bottle" also listed, a mug
# asked for as "mug" came back labelled "cup" and the can as "bottle".
DISTRACTORS = ("person", "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf",
               "door", "potted plant", "lamp", "television", "refrigerator", "sink", "wall")
YOLO_CACHE = 3          # detectors kept, one per vocabulary

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
        # not look like their names (the "keys" are a small block) but can be
        # described ("small object on the floor")
        self.aliases: dict[str, str] = {}
        self.yolo_error: str | None = None
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

    def yolo_boxes(self, rgb, names) -> list[dict]:
        """YOLO boxes for `names`, most confident first."""
        det = self.yolo(names)
        # Ultralytics reads a raw array as BGR; MuJoCo renders RGB.
        result = det._predict(np.ascontiguousarray(rgb[..., ::-1]))
        wanted, out = set(names), []
        for box in result.boxes:
            name = det.names[int(box.cls)]
            if name in wanted:
                u0, v0, u1, v1 = (int(round(v)) for v in box.xyxy[0].tolist())
                out.append({"name": name, "confidence": float(box.conf),
                            "box": (u0, v0, u1, v1), "source": "yolo"})
        return sorted(out, key=lambda b: -b["confidence"])

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
        """The best box for `name`. auto: a confident YOLO box, else the vision
        LLM's, else YOLO's weak one."""
        weak = None
        if self.backend in ("auto", "yolo"):
            try:
                hits = self.yolo_boxes(rgb, [name])
            except ImportError as e:
                if self.backend == "yolo":
                    raise
                self.yolo_error, hits = str(e), []
            if hits and (self.backend == "yolo" or hits[0]["confidence"] >= self.yolo_accept):
                return hits[0]
            weak = hits[0] if hits else None
        if self.backend in ("auto", "llm"):
            return self.llm_box(rgb, name) or weak
        return weak

    # ---------------------------------------------------------------- depth
    def support_z(self, xy, z_from) -> float:
        """Height of the fixed surface under `xy`, looking through loose items."""
        m, d = self._bot.model, self._bot.data
        start = np.array([xy[0], xy[1], z_from], float)
        down = np.array([0.0, 0.0, -1.0])
        hit = np.zeros(1, np.int32)
        for _ in range(8):
            dist = mujoco.mj_ray(m, d, start, down, None, 1, -1, hit)
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
