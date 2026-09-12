"""YOLO detection fused with depth, behind the same interface as the colour
detector in `perception`.

By default this is a PRETRAINED open-vocabulary model (YOLO-World): it finds
whatever it is told to look for, by name, with no training on this simulator.
The object of interest is `target` -- any short noun phrase ("mug", "red
cylinder", "potted plant") -- and its detections come back labelled
`perception.GOAL_LABEL`, which is what the navigator drives to. A generic
household vocabulary is detected alongside it; those keep their own names.

A YOLO box says *what* and *where in the image*. It says nothing about range,
which is what a navigator needs -- so every box is paired with the depth
pixels inside it to recover a world position. The median of the nearer half
of those pixels is used rather than the mean: a box around a cylinder also
contains the floor and sky behind it, and averaging those in drags the
estimate metres past the object.

    detector = YoloDetector(target="mug")                  # pretrained, open vocabulary
    detector = YoloDetector("runs/bracketbot_yolo/weights/best.pt")   # a fixed-class net
    dets = detector(obs)          # same Detection objects perception returns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .perception import GOAL_LABEL, Detection, Observation

# Downloaded on first use into the repo's gitignored weights/ directory.
DEFAULT_WEIGHTS = "yolov8l-worldv2.pt"
WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights"
DEFAULT_TARGET = "red cylinder"

# Detected alongside the target, so the rest of the room is reported under its
# own names (useful context, never the goal). Callers can replace it with
# `classes=`.
HOUSEHOLD_CLASSES = (
    "person", "chair", "sofa", "table", "desk", "bed", "cabinet", "shelf", "bookshelf",
    "door", "potted plant", "lamp", "trash bin", "cardboard box", "box", "mug", "cup",
    "bottle", "can", "bowl", "remote control", "keys", "phone", "laptop", "book",
    "backpack", "ball", "shoe", "pillow", "television", "refrigerator", "sink", "wall",
    "column")


def resolve_weights(weights) -> str:
    """A path that exists, or a pretrained Ultralytics model name (e.g.
    'yolov8s-worldv2.pt', 'yolo11s.pt') fetched into WEIGHTS_DIR."""
    path = Path(weights)
    if path.exists():
        return str(path)
    if path.parent == Path("."):
        from ultralytics.utils.downloads import GITHUB_ASSETS_NAMES
        cached = WEIGHTS_DIR / path.name
        if cached.exists() or path.name in GITHUB_ASSETS_NAMES:
            WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
            return str(cached)          # Ultralytics downloads it on load
    raise FileNotFoundError(
        f"no YOLO weights at {weights}, and it is not a pretrained Ultralytics "
        f"model name such as {DEFAULT_WEIGHTS!r}")


def _is_open_vocab(model) -> bool:
    return type(model.model).__name__ in ("WorldModel", "YOLOEModel", "YOLOESegModel")


class YoloDetector:
    """Callable detector: Observation -> list[Detection], ranged from depth."""

    def __init__(self, weights=DEFAULT_WEIGHTS, target: str | None = None,
                 classes=None, conf: float = 0.25, iou: float = 0.5,
                 imgsz: int = 320, min_height: float = 0.08,
                 max_height: float = 2.5, min_range: float = 0.4,
                 self_radius: float = 0.55, device: str | None = None,
                 goal_label: str = GOAL_LABEL, verbose: bool = False):
        from ultralytics import YOLO

        self.weights = resolve_weights(weights)
        self.model = YOLO(self.weights)
        self.open_vocab = _is_open_vocab(self.model)
        if self.open_vocab:
            self.target = (target or DEFAULT_TARGET).strip()
            names = [self.target] + [c for c in (classes or HOUSEHOLD_CLASSES)
                                     if c.strip() and c.strip() != self.target]
            self.set_classes(names)
        else:
            # A fixed-class net only knows its own names. The nets trained here
            # already call the goal "target"; a COCO model has to be told which
            # of its 80 classes is the object of interest.
            known = set(self.model.names.values())
            if target is not None and target not in known:
                raise ValueError(f"{Path(self.weights).name} has no class {target!r}; it knows "
                                 f"{sorted(known)}. Use an open-vocabulary model such as "
                                 f"{DEFAULT_WEIGHTS!r} to look for anything by name.")
            self.target = target
        self.names = self.model.names
        self.goal_label = goal_label
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.min_height, self.max_height = min_height, max_height
        self.min_range, self.self_radius = min_range, self_radius
        self.device, self.verbose = device, verbose

    def set_classes(self, names):
        """Replace the open vocabulary; names[0] is the target."""
        names = list(names)
        if type(self.model.model).__name__ == "WorldModel":
            self.model.set_classes(names)
            # The CLIP text encoder is only needed to embed the names. Ultralytics
            # keeps it as a submodule, so it would follow the detector onto the
            # GPU -- ~600 MB next to an LLM that already fills most of the card.
            self.model.model.clip_model = None
        else:
            self.model.set_classes(names, self.model.get_text_pe(names))
        self.target = names[0]
        self.names = self.model.names

    def names_index(self, name: str) -> int:
        """The class index the model uses for `name`."""
        return next(i for i, n in self.names.items() if n == name)

    def label_for(self, cls: int) -> str:
        name = self.names[int(cls)]
        return self.goal_label if (self.target is not None and name == self.target) else name

    def __call__(self, obs: Observation, robot_yaw: float = 0.0,
                 **_) -> list[Detection]:
        # Ultralytics treats a raw ndarray as BGR (the OpenCV convention) but
        # MuJoCo renders RGB. Handing it RGB silently swaps red and blue, so
        # the red goal column classifies as a blue pillar and vice versa --
        # with the boxes still pixel-perfect, which makes it look like a
        # labelling bug rather than a channel bug. Training read its images
        # from disk through cv2, so only inference was ever affected.
        bgr = np.ascontiguousarray(obs.rgb[..., ::-1])
        result = self._predict(bgr)
        out: list[Detection] = []
        for box in result.boxes:
            u0, v0, u1, v1 = (int(round(v)) for v in box.xyxy[0].tolist())
            det = self._range_box(obs, self.label_for(box.cls), u0, v0, u1, v1,
                                  robot_yaw, float(box.conf))
            if det is not None:
                out.append(det)
        out.sort(key=lambda d: d.distance)
        return out

    def _predict(self, bgr):
        try:
            return self.model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                      device=self.device, verbose=False)[0]
        except RuntimeError as e:          # torch.OutOfMemoryError is a RuntimeError
            if self.device == "cpu" or "out of memory" not in str(e).lower():
                raise
        # The local LLM shares the GPU and usually holds most of it. A detector
        # that cannot fit is still useful on the CPU, just slower.
        import torch
        print(f"[yolo] no GPU memory left for {Path(self.weights).name}; running it on the CPU")
        torch.cuda.empty_cache()
        self.device, self.model.predictor = "cpu", None
        return self.model.predict(bgr, conf=self.conf, iou=self.iou, imgsz=self.imgsz,
                                  device=self.device, verbose=False)[0]

    def _range_box(self, obs, label, u0, v0, u1, v1, robot_yaw, conf):
        h, w = obs.depth.shape
        u0, u1 = max(u0, 0), min(u1, w - 1)
        v0, v1 = max(v0, 0), min(v1, h - 1)
        if u1 <= u0 or v1 <= v0:
            return None

        patch_pts = obs.points[v0:v1 + 1, u0:u1 + 1]
        patch_d = obs.depth[v0:v1 + 1, u0:u1 + 1]
        z = patch_pts[..., 2]
        good = (np.isfinite(patch_d) & (patch_d > self.min_range) &
                (z > self.min_height) & (z < self.max_height))
        good &= np.linalg.norm(patch_pts[..., :2] - obs.robot_xy,
                               axis=-1) > self.self_radius
        if good.sum() < 8:
            return None

        d = patch_d[good]
        # Keep the nearer half: the far tail of a box is background seen past
        # the object's silhouette, not the object.
        near = d <= np.median(d)
        pts = patch_pts[good][near]
        if len(pts) < 4:
            pts = patch_pts[good]
        centre = np.median(pts, axis=0)

        delta = centre[:2] - obs.robot_xy
        return Detection(
            label=label,
            position=centre,
            distance=float(np.linalg.norm(centre - obs.cam_pos)),
            bearing=float(np.arctan2(delta[1], delta[0]) - robot_yaw),
            extent=pts.max(0) - pts.min(0),
            pixels=int(good.sum()),
            bbox=(u0, v0, u1, v1))

    def __repr__(self):
        vocab = f"target={self.target!r} " if self.target else ""
        return f"<YoloDetector {Path(self.weights).name} {vocab}{len(self.names)} classes conf={self.conf}>"
