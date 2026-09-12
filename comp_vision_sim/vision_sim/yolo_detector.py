"""YOLO detection fused with depth, behind the same interface as the colour
detector in `perception`.

A YOLO box says *what* and *where in the image*. It says nothing about range,
which is what a navigator needs -- so every box is paired with the depth
pixels inside it to recover a world position. The median of the nearer half
of those pixels is used rather than the mean: a box around a cylinder also
contains the floor and sky behind it, and averaging those in drags the
estimate metres past the object.

    detector = YoloDetector("runs/bracketbot_yolo/weights/best.pt")
    dets = detector(obs)          # same Detection objects perception returns
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .perception import Detection, Observation


class YoloDetector:
    """Callable detector: Observation -> list[Detection], ranged from depth."""

    def __init__(self, weights, conf: float = 0.35, iou: float = 0.5,
                 imgsz: int = 320, min_height: float = 0.08,
                 max_height: float = 2.5, min_range: float = 0.4,
                 self_radius: float = 0.55, device: str = "cpu",
                 verbose: bool = False):
        from ultralytics import YOLO

        weights = Path(weights)
        if not weights.exists():
            raise FileNotFoundError(
                f"no YOLO weights at {weights}. Train them first:\n"
                f"    python yolo_train.py")
        self.model = YOLO(str(weights))
        self.names = self.model.names
        self.conf, self.iou, self.imgsz = conf, iou, imgsz
        self.min_height, self.max_height = min_height, max_height
        self.min_range, self.self_radius = min_range, self_radius
        self.device, self.verbose = device, verbose

    def __call__(self, obs: Observation, robot_yaw: float = 0.0,
                 **_) -> list[Detection]:
        # Ultralytics treats a raw ndarray as BGR (the OpenCV convention) but
        # MuJoCo renders RGB. Handing it RGB silently swaps red and blue, so
        # the red goal column classifies as a blue pillar and vice versa --
        # with the boxes still pixel-perfect, which makes it look like a
        # labelling bug rather than a channel bug. Training read its images
        # from disk through cv2, so only inference was ever affected.
        bgr = np.ascontiguousarray(obs.rgb[..., ::-1])
        result = self.model.predict(bgr, conf=self.conf, iou=self.iou,
                                    imgsz=self.imgsz, device=self.device,
                                    verbose=False)[0]
        out: list[Detection] = []
        for box in result.boxes:
            u0, v0, u1, v1 = (int(round(v)) for v in box.xyxy[0].tolist())
            label = self.names[int(box.cls)]
            det = self._range_box(obs, label, u0, v0, u1, v1, robot_yaw,
                                  float(box.conf))
            if det is not None:
                out.append(det)
        out.sort(key=lambda d: d.distance)
        return out

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
        return f"<YoloDetector {len(self.names)} classes conf={self.conf}>"
