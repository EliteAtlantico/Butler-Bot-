"""Surfaces to put things on, found in any scene's geometry.

`places.PLACES` names four spots in scene_home.xml by their geoms. This finds
their equivalent in any scene: every fixed, upward-facing box or cylinder top
the arm could set something on (tables, a sideboard, a seat, a held-out palm),
and open containers to drop things into (a floor with walls round it). Each
becomes a `PlaceSpec` that `Place` and `PlacePlanner` take exactly like a
named place.

    surfaces = {s.name: s for s in find_surfaces(bot.model, bot.data)}
    Place(bot, pick, surfaces["sideboard"].spec(obstacles=obstacle_boxes(
        bot.model, bot.data, exclude_body="sideboard")))

A top face counts when it is wide enough to hold something (legs, rails and
wall panels are not), within the arm's set-down heights, and not covered: a
torso inside a coat, a shelf inside its bookcase. Like the rest of the grasp
planner (`GraspPlanner.support`, `low_obstacles`), this reads the static
furniture from the model; loose items and the robot are never surfaces.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .places import PlaceSpec, Surface, _geom_top

MIN_HALF = 0.05              # narrower than 10 cm: a leg, a rail, a panel
MIN_TOP, MAX_TOP = 0.10, 1.0  # set-down heights the arm reaches (containers may sit lower)
WALL_MIN = 0.05              # a container's walls stand at least this far above its floor
EDGE_TOL = 0.03              # a wall stands within this of the floor's edge
CLEAR_ABOVE = 0.03           # something this close above a top's centre covers it


def _world_aabb(m, d, g):
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
    h = np.abs(R) @ m.geom_aabb[g, 3:]
    return c - h, c + h


def _fixed(m, b):
    """Static scenery: welded to the world (not the robot, not a loose item)."""
    return m.body_weldid[b] == 0 and m.body_mocapid[b] < 0


def _top_face(m, d, g):
    """(axes, half, top) of an upward-facing box or upright cylinder, else None."""
    R = d.geom_xmat[g].reshape(3, 3)
    size, kind = m.geom_size[g], m.geom_type[g]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        k = int(np.argmax(np.abs(R[2])))
        if abs(R[2, k]) < 0.99:
            return None
        other = [i for i in range(3) if i != k]
        half = np.array([size[i] for i in other])
    elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        if abs(R[2, 2]) < 0.99:
            return None
        other = [0, 1]
        half = np.array([size[0], size[0]])
    else:
        return None
    axes = np.array([R[:2, i] / max(float(np.linalg.norm(R[:2, i])), 1e-9) for i in other])
    return axes, half, _geom_top(m, d, g)


@dataclass
class FoundSurface:
    name: str
    body: str
    geom: int
    surface: Surface
    mode: str                     # "set" on top, or "drop" into a container

    @property
    def top(self) -> float:
        return float(self.surface.top)

    @property
    def size(self) -> np.ndarray:
        return 2 * self.surface.half

    def spec(self, at=None, mode=None, obstacles=None) -> PlaceSpec:
        """A PlaceSpec for `Place`: optionally at a world xy, and as set or drop."""
        mode = mode or self.mode
        if mode not in ("set", "drop"):
            raise ValueError(f"mode must be set or drop, not {mode!r}")
        return PlaceSpec(self.name, self.body, "", mode, "in" if mode == "drop" else "on",
                         surface=self.surface,
                         obstacles=None if obstacles is None else tuple(obstacles),
                         at=None if at is None else (float(at[0]), float(at[1])))


def find_surfaces(model, data, max_top=MAX_TOP) -> list[FoundSurface]:
    """Every usable surface in the scene, in model order."""
    m, d = model, data
    boxes = {g: _world_aabb(m, d, g) for g in range(m.ngeom)
             if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE
             and _fixed(m, int(m.geom_bodyid[g]))}
    found = []
    for g, (lo, hi) in boxes.items():
        b = int(m.geom_bodyid[g])
        body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if b == 0 or not body:          # loose world geoms (walls), unnamed bodies
            continue
        face = _top_face(m, d, g)
        if face is None:
            continue
        axes, half, top = face
        if half.min() < MIN_HALF:
            continue
        centre = d.geom_xpos[g].copy()
        probe = np.array([centre[0], centre[1], top + CLEAR_ABOVE])
        if any(o != g and np.all(olo <= probe) and np.all(probe <= ohi)
               for o, (olo, ohi) in boxes.items()):
            continue
        walls = 0
        rim = top
        for o, (olo, ohi) in boxes.items():
            if o == g or m.geom_bodyid[o] != b:
                continue
            local = axes @ ((olo[:2] + ohi[:2]) / 2 - centre[:2])
            if np.all(np.abs(local) <= half + 0.05):
                rim = max(rim, float(ohi[2]))
            if olo[2] > top + EDGE_TOL or ohi[2] < top + WALL_MIN:
                continue
            if (np.any(np.abs(np.abs(local) - half) < EDGE_TOL)
                    and np.all(np.abs(local) <= half + EDGE_TOL)):
                walls += 1
        mode = "drop" if walls >= 2 else "set"
        if top > max_top or (mode == "set" and top < MIN_TOP):
            continue
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or body
        found.append(FoundSurface(name, body, g, Surface(center=centre, axes=axes, half=half,
                                                         top=top, rim=rim), mode))
    # A body with one surface goes by the body's name ("sideboard", "basket").
    per_body = {}
    for s in found:
        per_body.setdefault(s.body, []).append(s)
    taken = set()
    for body, group in per_body.items():
        for i, s in enumerate(group):
            name = body if len(group) == 1 else s.name
            if name in taken or (len(group) > 1 and name == body):
                name = f"{body}_{i + 1}"
            s.name = name
            taken.add(name)
    return found


def obstacle_footprints(model, data, exclude_body=None, max_bottom=1.6):
    """(name, lo_xy, hi_xy) of the fixed scenery a drive must keep clear of: one
    per body, and one per loose world geom (walls). Things hung higher than
    `max_bottom` (a lintel) are passed under, and decoration the robot can drive
    over or through (skirting, rugs) is ignored."""
    m, d = model, data
    if isinstance(exclude_body, str):
        exclude_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, exclude_body)
    per_body, loose = {}, []
    for g in range(m.ngeom):
        b = int(m.geom_bodyid[g])
        if (m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not _fixed(m, b)
                or b == exclude_body):
            continue
        # Decoration, not scenery: a geom that can neither collide with nor be
        # collided into is there to be looked at. Skipped per geom, not per body,
        # because a body may carry both (a plant's visual leaves, its solid pot).
        # home_search.xml's skirting is one body of eleven strips round the whole
        # flat; merged into one AABB below it covered 94 m^2 -- the entire floor --
        # so every go_to in that scene was refused as "too close to the trim".
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        lo, hi = _world_aabb(m, d, g)
        if lo[2] > max_bottom:
            continue
        if b == 0:
            loose.append((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "wall",
                          lo[:2], hi[:2]))
        elif b in per_body:
            name, plo, phi = per_body[b]
            per_body[b] = (name, np.minimum(plo, lo[:2]), np.maximum(phi, hi[:2]))
        else:
            per_body[b] = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or f"body {b}",
                           lo[:2], hi[:2])
    return loose + list(per_body.values())


def obstacle_boxes(model, data, exclude_body=None, max_bottom=1.6):
    """Footprints (lo_xy, hi_xy) as `PlaceSpec.obstacles` takes them."""
    return [(lo, hi) for _, lo, hi in obstacle_footprints(model, data, exclude_body, max_bottom)]
