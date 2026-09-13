"""Read the scene's own geometry, so the stack has nothing hard-coded in it.

Everything the navigator used to assume -- a floor at z=0, a world inside
+-6 m, a camera called `head_depth`, a 0.30 m robot -- is a property of one
particular XML. Measuring them from the model instead is what lets the same
code drive a different scene, or a different robot, without edits.

    info = SceneInfo.from_model(bot.model, bot.data)
    grid = OccupancyGrid.covering(info.bounds, resolution=0.10)

The robot/scenery split is the one non-obvious bit, and the obvious test is
wrong: `body_rootid` does NOT separate them, because a static prop parented
to the world roots at itself exactly like a free-floating robot does. The
test that works is `body_weldid`, which is 0 for anything rigidly welded to
the world -- so scenery welds to 0 and articulated bodies do not. No name
matching required, which is what lets an unfamiliar scene work.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class SceneInfo:
    bounds: tuple[float, float, float, float]   # xmin, ymin, xmax, ymax
    floor_z: float
    robot_radius: float
    robot_top: float
    camera: str
    map_range: float

    @property
    def span(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0

    def including(self, xy, margin: float = 2.0) -> "SceneInfo":
        """Widen the bounds to contain a point.

        A goal outside the grid is unreachable in a way that looks like a
        planner failure: to_cell() returns an out-of-range index, every A*
        call returns None, and the robot just sits reporting STUCK. Scene
        geometry alone does not know where you intend to send it.
        """
        from dataclasses import replace
        x, y = float(xy[0]), float(xy[1])
        x0, y0, x1, y1 = self.bounds
        return replace(self, bounds=(min(x0, x - margin), min(y0, y - margin),
                                     max(x1, x + margin), max(y1, y + margin)))

    def describe(self) -> str:
        x0, y0, x1, y1 = self.bounds
        return (f"scene x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]  "
                f"floor z={self.floor_z:.2f}  robot r={self.robot_radius:.2f} "
                f"top={self.robot_top:.2f}  camera={self.camera!r}  "
                f"map_range={self.map_range:.1f}m")

    # ------------------------------------------------------------ discovery
    @classmethod
    def from_model(cls, model, data, camera: str | None = None,
                   margin: float = 2.0, max_map_range: float = 8.0):
        root = _robot_root(model)
        robot = _robot_geoms(model, root)
        static = ~robot

        floor_z = _floor_height(model, data, static)
        radius, top = _robot_extent(model, data, robot, floor_z, root)
        bounds = _scenery_bounds(model, data, static, floor_z)

        # The grid has to hold the robot as well as the scenery, or the very
        # first to_cell() falls outside it.
        rx, ry = float(data.xpos[root][0]), float(data.xpos[root][1])
        x0 = min(bounds[0], rx) - margin
        y0 = min(bounds[1], ry) - margin
        x1 = max(bounds[2], rx) + margin
        y1 = max(bounds[3], ry) + margin

        diag = float(np.hypot(x1 - x0, y1 - y0))
        return cls(bounds=(x0, y0, x1, y1), floor_z=floor_z,
                   robot_radius=radius, robot_top=top,
                   camera=camera or pick_camera(model),
                   map_range=min(max_map_range, diag))


def pick_camera(model, prefer=("depth", "rgbd", "head", "front")) -> str:
    """Choose a forward-looking fixed camera without being told its name."""
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
             for i in range(model.ncam)]
    names = [n for n in names if n]
    if not names:
        raise ValueError("model declares no cameras; the vision stack needs one")
    fixed = [n for n in names
             if model.cam_mode[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)]
             == mujoco.mjtCamLight.mjCAMLIGHT_FIXED]
    pool = fixed or names
    for want in prefer:
        for n in pool:
            if want in n.lower():
                return n
    return pool[0]


def _robot_root(model) -> int:
    """The body at the top of the articulated tree, or the world body.

    Chosen by subtree size rather than document order. A scene containing
    pickable objects has many free bodies -- each is its own root of a
    one-body tree -- and the robot is simply the largest of them. Taking the
    first moving body instead makes the answer depend on whether an item
    happens to be declared before the robot.
    """
    moving = np.where(model.body_weldid != 0)[0]
    if not len(moving):
        return 0
    roots, counts = np.unique(model.body_rootid[moving], return_counts=True)
    return int(roots[int(np.argmax(counts))])


def _robot_geoms(model, root: int | None = None) -> np.ndarray:
    """Boolean mask over geoms: True where the geom belongs to the robot.

    Membership is "in the robot's kinematic tree", not merely "not welded to
    the world". A free-floating mug is not welded to the world either, and
    counting it as a robot part measured the footprint radius of the furnished
    demo scene at 8.36 m instead of 0.25 -- which blocks every cell in the
    costmap and leaves the planner nowhere legal to go.
    """
    if root is None:
        root = _robot_root(model)
    return model.body_rootid[model.geom_bodyid] == root


def _floor_height(model, data, static: np.ndarray) -> float:
    planes = np.where(static & (model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE))[0]
    if len(planes):
        return float(data.geom_xpos[planes][:, 2].min())
    ids = np.where(static)[0]
    if not len(ids):
        return 0.0
    return float((data.geom_xpos[ids][:, 2] - model.geom_rbound[ids]).min())


def _robot_extent(model, data, robot: np.ndarray, floor_z: float,
                  root_body: int | None = None):
    """Planar footprint radius and height of the robot, from its own geoms."""
    ids = np.where(robot)[0]
    if not len(ids):
        return 0.3, 1.0
    if root_body is None:
        root_body = _robot_root(model)
    root = data.xpos[root_body][:2]
    half = np.array([_geom_halfextent(model, data, g) for g in ids])
    d = np.linalg.norm(data.geom_xpos[ids][:, :2] - root, axis=1) + half[:, :2].max(1)
    top = float((data.geom_xpos[ids][:, 2] + half[:, 2]).max() - floor_z)
    # A 95th percentile rather than the max: one stray arm geom sticking out
    # should not inflate the footprint the planner has to keep clear.
    return float(np.percentile(d, 95)), top


def _geom_halfextent(model, data, g) -> np.ndarray:
    """World-axis half-extent of one geom.

    Deliberately not geom_rbound: that is a bounding SPHERE, so a 12 m wall
    0.2 m thick reports a 6 m radius in every direction and inflates the
    scene bounds into empty space on all sides. Rotating the local box
    half-sizes by |R| gives the real axis-aligned extent instead.
    """
    h = model.geom_aabb[g, 3:6]
    if not np.any(h):
        h = np.full(3, float(model.geom_rbound[g]))
    R = data.geom_xmat[g].reshape(3, 3)
    return np.abs(R) @ h


def _scenery_bounds(model, data, static: np.ndarray, floor_z: float):
    """XY extent of static geometry that stands above the floor."""
    ids = [g for g in np.where(static)[0]
           if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]
    lo, hi = [], []
    for g in ids:
        h = _geom_halfextent(model, data, g)
        c = data.geom_xpos[g]
        if c[2] + h[2] <= floor_z + 0.05:       # flat on the floor, not an obstacle
            continue
        lo.append(c[:2] - h[:2])
        hi.append(c[:2] + h[:2])
    if not lo:
        return (-1.0, -1.0, 1.0, 1.0)
    lo, hi = np.array(lo), np.array(hi)
    return (float(lo[:, 0].min()), float(lo[:, 1].min()),
            float(hi[:, 0].max()), float(hi[:, 1].max()))
