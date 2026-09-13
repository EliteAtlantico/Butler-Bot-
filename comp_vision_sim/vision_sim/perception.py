"""RGB-D perception: registered colour+depth -> world points -> object detections.

The head camera is declared once in the model as `head_depth`; rendering it
with a colour renderer and with a depth renderer gives two images from the
*same* pose, so they are registered by construction -- no extrinsic calibration
and no reprojection, which is the sim equivalent of a RealSense's aligned
streams.

    obs = observe(bot)
    for det in detect(obs):
        print(det.label, det.position, det.distance)

Detection fuses colour with geometry on purpose. Neither works alone here: the
checkerboard floor renders the same hue as the blue pillars, and height alone
cannot tell a goal from an obstacle.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

CAMERA = "head_depth"


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_model(cls, model, cam_id: int, width: int, height: int) -> "Intrinsics":
        # MuJoCo's cam_fovy is the VERTICAL field of view, in degrees.
        f = 0.5 * height / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
        return cls(f, f, width / 2, height / 2, width, height)


@dataclass
class Observation:
    """One RGB-D frame, already lifted into the world frame."""
    rgb: np.ndarray           # (H, W, 3) uint8
    depth: np.ndarray         # (H, W) float32 metres, inf past max_range
    points: np.ndarray        # (H, W, 3) world xyz, nan where no return
    valid: np.ndarray         # (H, W) bool
    cam_pos: np.ndarray       # (3,) camera origin in world
    cam_mat: np.ndarray       # (3, 3) camera->world rotation
    intrinsics: Intrinsics
    robot_xy: np.ndarray      # (2,) chassis position, for self-return rejection


@dataclass
class Detection:
    label: str
    position: np.ndarray      # world xyz centroid of the visible surface
    distance: float           # metres from camera
    bearing: float            # radians in the robot frame, + is left
    extent: np.ndarray        # world xyz bounding-box size
    pixels: int
    bbox: tuple[int, int, int, int] = (0, 0, 0, 0)   # u0, v0, u1, v1

    def __repr__(self):
        x, y, z = self.position
        return (f"<{self.label} at ({x:.2f}, {y:.2f}, {z:.2f}) "
                f"{self.distance:.2f}m {np.rad2deg(self.bearing):+.0f}deg "
                f"{self.pixels}px>")


@dataclass(frozen=True)
class ObjectClass:
    """A colour prior. `hue` is a (lo, hi) window in degrees and may wrap 360."""
    label: str
    hue: tuple[float, float]
    sat_min: float = 0.45
    val_min: float = 0.15
    min_pixels: int = 25

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        lo, hi = self.hue
        inside = (h >= lo) & (h <= hi) if lo <= hi else (h >= lo) | (h <= hi)
        return inside & (s >= self.sat_min) & (v >= self.val_min)


# Hues measured off the rendered scene, not copied from the material rgba --
# lighting shifts both saturation and value away from the declared colour.
DEFAULT_CLASSES = (
    ObjectClass("target", hue=(345.0, 20.0), sat_min=0.50),
    ObjectClass("barrier", hue=(22.0, 55.0), sat_min=0.50),
    ObjectClass("pillar", hue=(195.0, 240.0), sat_min=0.45),
)

GOAL_LABEL = "target"


def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Vectorised HSV. Hue in degrees [0, 360), sat and val in [0, 1]."""
    a = rgb.astype(np.float32) / 255.0
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mx, mn = a.max(-1), a.min(-1)
    diff = mx - mn
    safe = np.where(diff == 0, 1.0, diff)
    hue = np.select(
        [diff == 0, mx == r, mx == g],
        [0.0, 60.0 * (((g - b) / safe) % 6), 60.0 * ((b - r) / safe + 2)],
        default=60.0 * ((r - g) / safe + 4))
    sat = np.where(mx == 0, 0.0, diff / np.where(mx == 0, 1.0, mx))
    return np.stack([hue, sat, mx], -1)


def deproject(depth: np.ndarray, intr: Intrinsics) -> np.ndarray:
    """Depth image -> camera-frame xyz (x right, y up, -z forward), nan if none.

    MuJoCo's depth is the distance along the optical axis, not the slant range,
    so it multiplies straight into the pinhole model.
    """
    v, u = np.mgrid[0:intr.height, 0:intr.width]
    z = np.where(np.isfinite(depth), depth, np.nan)
    return np.stack([(u - intr.cx) * z / intr.fx,
                     -(v - intr.cy) * z / intr.fy,
                     -z], -1)


def observe(bot, camera: str | None = None, width: int = 320, height: int = 240,
            max_range: float = 12.0) -> Observation:
    """Render one registered RGB-D frame and lift it into world coordinates.

    `camera` defaults to whatever forward-looking fixed camera the model
    happens to declare, so an unfamiliar scene does not need to have named
    its camera `head_depth`.
    """
    if camera is None:
        from .scene import pick_camera
        camera = pick_camera(bot.model)
    cam_id = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam_id < 0:
        raise ValueError(f"no camera named {camera!r}")
    intr = Intrinsics.from_model(bot.model, cam_id, width, height)

    rgb = bot.camera(camera, width, height)
    depth = bot.depth(camera, width, height, max_range=max_range)

    # Read the pose AFTER rendering: update_scene runs mj_forward-ish work, and
    # a stale pose silently skews every world point by a frame of motion.
    cam_pos = bot.data.cam_xpos[cam_id].copy()
    cam_mat = bot.data.cam_xmat[cam_id].reshape(3, 3).copy()

    pts_cam = deproject(depth, intr)
    points = pts_cam @ cam_mat.T + cam_pos
    return Observation(rgb=rgb, depth=depth, points=points,
                       valid=np.isfinite(depth), cam_pos=cam_pos,
                       cam_mat=cam_mat, intrinsics=intr,
                       robot_xy=bot.position[:2].copy())


def _cluster_xy(xy: np.ndarray, res: float) -> np.ndarray:
    """Grid-cell connected components in the ground plane.

    Clustering in world XY rather than in image space means one object split
    across the frame by an occluder still comes back as a single detection.
    """
    if len(xy) == 0:
        return np.zeros(0, int)
    cells = np.floor(xy / res).astype(np.int64)
    buckets: dict[tuple[int, int], list[int]] = {}
    for i, c in enumerate(map(tuple, cells)):
        buckets.setdefault(c, []).append(i)

    labels = np.full(len(xy), -1, int)
    nxt = 0
    for seed in buckets:
        if labels[buckets[seed][0]] != -1:
            continue
        stack, members = [seed], []
        seen = {seed}
        while stack:
            cx, cy = stack.pop()
            members.extend(buckets[(cx, cy)])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (cx + dx, cy + dy)
                    if n in buckets and n not in seen:
                        seen.add(n)
                        stack.append(n)
        labels[members] = nxt
        nxt += 1
    return labels


def detect(obs: Observation, classes=DEFAULT_CLASSES, min_height: float = 0.12,
           max_height: float = 2.5, min_range: float = 0.4,
           self_radius: float = 0.55, cluster_res: float = 0.3,
           robot_yaw: float = 0.0, floor_z: float = 0.0,
           **_) -> list[Detection]:
    """Colour-classify pixels, then cluster the survivors in 3D.

    `min_height` is what removes the floor, which renders the same hue as the
    blue pillars and would otherwise swamp every blue detection.
    """
    hsv = rgb_to_hsv(obs.rgb)
    z = obs.points[..., 2] - floor_z
    ground = obs.valid & (z > min_height) & (z < max_height)
    ground &= obs.depth > min_range
    # Anything standing within self_radius of the chassis is the robot's own
    # arms, not scenery.
    ground &= np.linalg.norm(obs.points[..., :2] - obs.robot_xy, axis=-1) > self_radius

    out: list[Detection] = []
    for cls in classes:
        sel = cls.mask(hsv) & ground
        if sel.sum() < cls.min_pixels:
            continue
        pts = obs.points[sel]
        pix = np.argwhere(sel)          # (N, 2) as (v, u), aligned with pts
        labels = _cluster_xy(pts[:, :2], cluster_res)
        for lab in range(labels.max() + 1):
            member = labels == lab
            blob = pts[member]
            if len(blob) < cls.min_pixels:
                continue
            vu = pix[member]
            centre = blob.mean(0)
            delta = centre[:2] - obs.robot_xy
            out.append(Detection(
                label=cls.label,
                position=centre,
                distance=float(np.linalg.norm(centre - obs.cam_pos)),
                bearing=float(np.arctan2(delta[1], delta[0]) - robot_yaw),
                extent=blob.max(0) - blob.min(0),
                pixels=int(len(blob)),
                bbox=(int(vu[:, 1].min()), int(vu[:, 0].min()),
                      int(vu[:, 1].max()), int(vu[:, 0].max()))))
    out.sort(key=lambda d: d.distance)
    return out


class GeometricDetector:
    """Colour-free detection: cluster whatever stands above the floor.

    The colour detector needs a palette that matches the scene. This one
    needs nothing -- it reports every clustered obstacle as one unlabelled
    object, which is all a planner requires. Use it when driving a scene
    whose contents are unknown, paired with a coordinate goal.
    """

    def __init__(self, label: str = "obstacle", cluster_res: float = 0.35,
                 min_points: int = 30, min_height: float = 0.12,
                 max_height: float = 2.5, min_range: float = 0.4,
                 max_range: float = 10.0, self_radius: float = 0.55,
                 floor_z: float = 0.0):
        self.label, self.cluster_res, self.min_points = label, cluster_res, min_points
        self.min_height, self.max_height = min_height, max_height
        self.min_range, self.max_range = min_range, max_range
        self.self_radius, self.floor_z = self_radius, floor_z

    def __call__(self, obs: Observation, robot_yaw: float = 0.0,
                 **_) -> list[Detection]:
        z = obs.points[..., 2] - self.floor_z
        sel = (obs.valid & (z > self.min_height) & (z < self.max_height) &
               (obs.depth > self.min_range) & (obs.depth < self.max_range))
        sel &= np.linalg.norm(obs.points[..., :2] - obs.robot_xy,
                              axis=-1) > self.self_radius
        if not sel.any():
            return []
        pts = obs.points[sel]
        pix = np.argwhere(sel)
        labels = _cluster_xy(pts[:, :2], self.cluster_res)
        out: list[Detection] = []
        for lab in range(labels.max() + 1):
            member = labels == lab
            blob = pts[member]
            if len(blob) < self.min_points:
                continue
            vu = pix[member]
            centre = blob.mean(0)
            delta = centre[:2] - obs.robot_xy
            out.append(Detection(
                label=self.label, position=centre,
                distance=float(np.linalg.norm(centre - obs.cam_pos)),
                bearing=float(np.arctan2(delta[1], delta[0]) - robot_yaw),
                extent=blob.max(0) - blob.min(0), pixels=int(len(blob)),
                bbox=(int(vu[:, 1].min()), int(vu[:, 0].min()),
                      int(vu[:, 1].max()), int(vu[:, 0].max()))))
        out.sort(key=lambda d: d.distance)
        return out


# Past this the stereo baseline of a real RGB-D head stops resolving disparity,
# so mapping anything further would be trusting numbers hardware cannot give.
MAP_RANGE = 8.0


def obstacle_points(obs: Observation, min_height: float = 0.10,
                    max_height: float = 2.0, min_range: float = 0.4,
                    max_range: float = MAP_RANGE,
                    self_radius: float = 0.55,
                    floor_z: float = 0.0) -> np.ndarray:
    """Every return that is neither floor nor sky, as world xyz.

    Heights are relative to `floor_z`, not to the world origin, so a scene
    whose ground plane is not at z=0 still separates floor from obstacle.

    Deliberately colour-blind: a depth camera does not know what it is looking
    at, and the map should not depend on the detector recognising an object.
    """
    z = obs.points[..., 2] - floor_z
    sel = (obs.valid & (z > min_height) & (z < max_height) &
           (obs.depth > min_range) & (obs.depth < max_range))
    sel &= np.linalg.norm(obs.points[..., :2] - obs.robot_xy, axis=-1) > self_radius
    return obs.points[sel]


def floor_points(obs: Observation, max_height: float = 0.10,
                 min_range: float = 0.4,
                 max_range: float = MAP_RANGE,
                 floor_z: float = 0.0) -> np.ndarray:
    """Returns that landed on the ground plane -- observed free space."""
    z = obs.points[..., 2] - floor_z
    sel = (obs.valid & (z <= max_height) & (obs.depth > min_range) &
           (obs.depth < max_range))
    return obs.points[sel]
