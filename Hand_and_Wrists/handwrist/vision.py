"""Find household items with the head RGB-D camera.

    see = CameraEstimator()
    est = see(bot, CATALOGUE["remote"])     # ObjectEstimate, or None if not in view

The team's `vision_sim.perception.observe()` renders the head camera's colour
and depth images from one pose -- registered by construction -- and lifts
every pixel into world xyz. This module turns that into what the grasp
planner needs:

  1. Colour-segment the item. Hue/saturation windows are calibrated offline
     by tools/calibrate_colours.py from segmentation renders; at run time
     the detector sees nothing but colour and depth.
  2. Keep the biggest 3-D cluster, so a stray edge pixel or a second thing
     of the same colour does not drag the estimate.
  3. Read the geometry off the points. From a camera 1.5 m up, the TOP FACE
     of an upright item is fully visible, so its outline gives the centre,
     width, length and axis directly -- the side surface would only give the
     half facing the camera, biasing every centre toward the robot.

`vision_sim.detect()` is tuned for navigation-scale obstacles: it drops
anything under 12 cm tall and clusters on a 30 cm grid, which removes every
floor item and merges the three on the coffee table. So it is not used here.

The head camera is tilted 22 deg down with a 58 deg vertical field of view:
it cannot see the floor nearer than ~1.25 m, or a coffee-table top nearer
than ~0.9 m. Items are therefore observed from a distance and the grasp is
made on that estimate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from vision_sim.perception import observe, rgb_to_hsv

from . import PACKAGE_DIR
from .objects import ObjectEstimate, ObjectSpec

COLOURS_FILE = PACKAGE_DIR / "colours.json"
CAMERA = "head_depth"


@dataclass(frozen=True)
class ColourWindow:
    hue: tuple[float, float]      # degrees, (lo, hi); wraps through 0 if lo > hi
    sat_min: float
    val_min: float

    def mask(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue
        inside = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return inside & (s >= self.sat_min) & (v >= self.val_min)

    @property
    def center(self):
        """Middle of the hue window, in degrees (handles the 0/360 wrap)."""
        lo, hi = self.hue
        span = (hi - lo) % 360.0
        return (lo + span / 2) % 360.0


def _hue_dist(h, c):
    d = np.abs(h - c) % 360.0
    return np.minimum(d, 360.0 - d)


def load_colours(path=COLOURS_FILE) -> dict[str, ColourWindow]:
    if not path.exists():
        raise FileNotFoundError(
            f"no colour calibration at {path}; run tools/calibrate_colours.py")
    raw = json.loads(path.read_text())
    return {k: ColourWindow(tuple(v["hue"]), v["sat_min"], v["val_min"])
            for k, v in raw.items()}


def clusters(pts, res=0.02):
    """Connected components of 3-D points on a `res` voxel grid."""
    labels = np.full(len(pts), -1, int)
    if not len(pts):
        return labels
    cells = np.floor(pts / res).astype(np.int64)
    buckets: dict[tuple, list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        buckets.setdefault(c, []).append(i)
    nbrs = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)]
    nxt = 0
    for seed in buckets:
        if labels[buckets[seed][0]] != -1:
            continue
        stack, seen, members = [seed], {seed}, []
        while stack:
            cx, cy, cz = stack.pop()
            members.extend(buckets[(cx, cy, cz)])
            for dx, dy, dz in nbrs:
                n = (cx + dx, cy + dy, cz + dz)
                if n in buckets and n not in seen:
                    seen.add(n)
                    stack.append(n)
        labels[members] = nxt
        nxt += 1
    return labels


@dataclass
class Sighting:
    """The pixels and points vision attributed to one item (for debugging)."""
    name: str
    points: np.ndarray            # (N, 3) world xyz
    pixels: np.ndarray            # (N, 2) image (v, u)
    distance: float


class CameraEstimator:
    """Estimator for `Pick`: looks with the head camera, returns an
    ObjectEstimate, or None when the item is not in view."""

    def __init__(self, camera=CAMERA, width=640, height=480, max_range=4.0,
                 min_points=15, self_radius=0.35, colours=None):
        self.camera = camera
        self.width, self.height, self.max_range = width, height, max_range
        self.min_points, self.self_radius = min_points, self_radius
        self.colours = colours or load_colours()
        self.last_obs = None
        self.last_sighting: Sighting | None = None

    def __call__(self, bot, spec: ObjectSpec) -> ObjectEstimate | None:
        self.last_obs = observe(bot, self.camera, self.width, self.height,
                                self.max_range)
        return self.estimate(self.last_obs, spec)

    # -------------------------------------------------------------- segment
    def sight(self, obs, spec: ObjectSpec) -> Sighting | None:
        win = self.colours[spec.name]
        hsv = rgb_to_hsv(obs.rgb)
        sel = win.mask(hsv) & obs.valid
        # Calibrated windows overlap at the edges -- the ball's pink runs into
        # the mug's red and the remote's purple -- and a tidy-up "found" a
        # ball on the coffee table that was really the mug. A pixel two
        # windows claim goes to the item whose window centre is nearer.
        own = _hue_dist(hsv[..., 0], win.center)
        for other, w in self.colours.items():
            if other != spec.name:
                both = sel & w.mask(hsv)
                if both.any():
                    sel &= ~(both & (_hue_dist(hsv[..., 0], w.center) < own))
        # the robot's own arms are close to the camera; items never are
        sel &= np.linalg.norm(obs.points[..., :2] - obs.robot_xy, axis=-1) > self.self_radius
        if sel.sum() < self.min_points:
            return None
        pts, pix = obs.points[sel], np.argwhere(sel)
        lab = clusters(pts)
        best = np.bincount(lab).argmax()
        keep = lab == best
        if keep.sum() < self.min_points:
            return None
        pts, pix = pts[keep], pix[keep]
        return Sighting(spec.name, pts, pix,
                        float(np.linalg.norm(pts.mean(0) - obs.cam_pos)))

    # ------------------------------------------------------------- geometry
    def estimate(self, obs, spec: ObjectSpec) -> ObjectEstimate | None:
        s = self.sight(obs, spec)
        self.last_sighting = s
        if s is None:
            return None
        pts = s.points
        px = s.distance / obs.intrinsics.fy      # one pixel's footprint, m
        z = pts[:, 2]
        top, bottom = np.percentile(z, 99), np.percentile(z, 1)

        # the top face: a thin slice under the highest points (for a sphere,
        # the upper cap -- there is no flat top)
        band = max(0.015, (top - bottom) / 3) if spec.shape == "sphere" else 0.012
        face = pts[z > top - band][:, :2]
        if len(face) < 5:
            face = pts[:, :2]
        c = face.mean(0)
        if len(face) >= 3:
            _, vecs = np.linalg.eigh(np.cov((face - c).T))
            major, minor = vecs[:, 1], vecs[:, 0]
        else:
            major, minor = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        a, b = (face - c) @ major, (face - c) @ minor
        a_lo, a_hi = np.percentile(a, [1, 99])
        b_lo, b_hi = np.percentile(b, [1, 99])
        # centre on the middle of the outline, not the point centroid, which
        # sits wherever the pixels happen to be densest
        c = c + major * (a_lo + a_hi) / 2 + minor * (b_lo + b_hi) / 2
        length, width = a_hi - a_lo + px, b_hi - b_lo + px

        if spec.shape == "sphere":
            # the cap under-reads the diameter; the silhouette across the line
            # of sight does not
            view = c - obs.cam_pos[:2]
            across = np.array([-view[1], view[0]]) / max(np.linalg.norm(view), 1e-9)
            w = pts[:, :2] @ across
            width = length = float(np.percentile(w, 99) - np.percentile(w, 1) + px)
        elif spec.shape == "round":
            width = length = float((length + width) / 2)
            # A bottle's top face is its neck: 32 mm across a 70 mm body. So
            # also measure the silhouette across the line of sight, slice by
            # slice, and trust the body when it is much wider than the top.
            # (A mug's handle only widens some slices, and only when it points
            # sideways; the 10-90 % span and the 1.5x test keep it out.)
            body = self._slice_width(pts, c, obs, bottom, top)
            if body is not None and body > 1.5 * width:
                width = length = float(body + px)

        axis_yaw = None
        if spec.aligned:
            axis_yaw = float(np.arctan2(major[1], major[0]))
        elif spec.handle_geom is not None:
            # handle = points standing clear of the round body
            r = np.linalg.norm(pts[:, :2] - c, axis=1)
            out = pts[r > width / 2 + 0.006]
            if len(out) >= 4:
                d = out[:, :2].mean(0) - c
                axis_yaw = float(np.arctan2(d[1], d[0]))

        center = np.array([c[0], c[1], (top + bottom) / 2])
        return ObjectEstimate(name=spec.name, center=center, width=float(width),
                              height=float(top - bottom), length=float(length),
                              axis_yaw=axis_yaw, body_id=-1, source="camera")

    @staticmethod
    def _slice_width(pts, c, obs, bottom, top, dz=0.01, min_pts=8):
        """Median silhouette width across the line of sight over 1 cm slices."""
        view = c - obs.cam_pos[:2]
        across = np.array([-view[1], view[0]]) / max(np.linalg.norm(view), 1e-9)
        w = pts[:, :2] @ across
        spans = []
        for z0 in np.arange(bottom, top, dz):
            s = w[(pts[:, 2] >= z0) & (pts[:, 2] < z0 + dz)]
            if len(s) >= min_pts:
                spans.append(np.percentile(s, 90) - np.percentile(s, 10))
        # 10-90 % of a uniform spread is 0.8 of the full width
        return float(np.median(spans) / 0.8) if spans else None
