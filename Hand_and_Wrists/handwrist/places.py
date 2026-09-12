"""Where things go: named places to put an item, and how to reach them.

    coffee_table, side_table   set it down on the top, near the robot's edge
    basket                     lower it just inside the rim and let go
    person                     set it on the person's open hand (a hand-over)

Places are read from the scene's own geoms, so moving furniture in
scene_home.xml needs no code change. Parking is worked out the same way as
for a grasp -- every edge of the place's footprint, the arm that is holding
the item, and an IK check from the base pose each implies. The hand keeps
the orientation it picked the item up with, so the item goes down the way it
came up.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from bracketbot_sim.kinematics import rot_z


@dataclass(frozen=True)
class PlaceSpec:
    name: str
    body: str             # scene body the place is made of
    surface_geom: str     # the geom the item ends up resting on
    mode: str             # "set": lower onto it; "drop": release inside the rim
    preposition: str      # "on" / "in", for status messages


PLACES = {
    "coffee_table": PlaceSpec("coffee_table", "coffee_table", "coffee_table_top", "set", "on"),
    "side_table": PlaceSpec("side_table", "side_table", "side_table_top", "set", "on"),
    "basket": PlaceSpec("basket", "basket", "basket_floor", "drop", "in"),
    "person": PlaceSpec("person", "person", "person_palm", "set", "to"),
}


# Where successive items go in a "drop" place, in the place's own footprint
# frame, as fractions of its half-size: one after another along the long
# side, so a tidy-up lays items side by side instead of stacking them on the
# spot the first one landed.
DROP_SLOTS = ((0.0, 0.0), (-0.5, 0.0), (0.5, 0.0))   # the middle first
_drops = {}          # (id(model), place) -> items dropped there so far


def reset_drops():
    """Forget what has been dropped where -- call when a scene is (re)staged."""
    _drops.clear()


@dataclass
class Surface:
    center: np.ndarray    # (3,) world centre of the surface geom
    axes: np.ndarray      # (2, 2) rows: the footprint's x and y axes in world xy
    half: np.ndarray      # (2,) half-size along those axes
    top: float            # height an item rests at
    rim: float            # height of the tallest part of the place (basket walls)

    def contains(self, xy, margin=0.0):
        rel = self.axes @ (np.asarray(xy, float)[:2] - self.center[:2])
        return bool(np.all(np.abs(rel) <= self.half + margin))


def _geom_top(model, data, g):
    R = data.geom_xmat[g].reshape(3, 3)
    c, h = model.geom_aabb[g, :3], model.geom_aabb[g, 3:]
    return float(data.geom_xpos[g][2] + (R @ c)[2] + (np.abs(R) @ h)[2])


def footprint_of(model, data, body_name):
    """World xy bounding box (lo, hi) of every geom on a body."""
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    lo, hi = np.full(2, np.inf), np.full(2, -np.inf)
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != b:
            continue
        R = data.geom_xmat[g].reshape(3, 3)
        c = data.geom_xpos[g] + R @ model.geom_aabb[g, :3]
        h = np.abs(R) @ model.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, (c - h)[:2]), np.maximum(hi, (c + h)[:2])
    return lo, hi


def _box_dist(p, box):
    lo, hi = box
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, p - hi), 0.0)))


def route_clear(points, boxes, clearance, skip_start=0.0, skip_end=0.0, step=0.05):
    """Does the polyline through `points` keep `clearance` from every box?
    The first `skip_start` m and last `skip_end` m are not checked: the robot
    starts next to whatever it just worked at, and ends next to its target."""
    pts = [np.asarray(p, float) for p in points]
    total = sum(float(np.linalg.norm(b - a)) for a, b in zip(pts, pts[1:]))
    s0 = 0.0
    for a, b in zip(pts, pts[1:]):
        n = float(np.linalg.norm(b - a))
        for s in np.arange(0.0, n, step):
            along = s0 + s
            if along < skip_start or along > total - skip_end:
                continue
            p = a + (b - a) * (s / n)
            if any(_box_dist(p, box) < clearance for box in boxes):
                return False
        s0 += n
    return True


def surface_of(model, data, place) -> Surface:
    spec = PLACES[place] if isinstance(place, str) else place
    g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, spec.surface_geom)
    if g < 0:
        raise KeyError(f"scene has no geom {spec.surface_geom!r} for {spec.name!r}")
    R = data.geom_xmat[g].reshape(3, 3)
    axes = np.array([R[:2, k] / np.linalg.norm(R[:2, k]) for k in (0, 1)])
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
    rim = max(_geom_top(model, data, i) for i in range(model.ngeom)
              if model.geom_bodyid[i] == b)
    return Surface(center=data.geom_xpos[g].copy(), axes=axes,
                   half=model.geom_size[g][:2].copy(),
                   top=_geom_top(model, data, g), rim=rim)


@dataclass
class PlaceTarget:
    side: str
    base_xy: np.ndarray
    base_yaw: float
    above: np.ndarray        # hand hovers here first
    point: np.ndarray        # hand releases here
    mat: np.ndarray          # hand orientation, world frame
    reach: float
    cost: float

    def describe(self):
        return (f"{self.side} arm, park at ({self.base_xy[0]:.2f}, {self.base_xy[1]:.2f}) "
                f"facing {np.rad2deg(self.base_yaw):+.0f} deg, reach {self.reach:.2f} m")


class PlacePlanner:
    SET_INSET = 0.10     # set items down this far in from the near edge
    SET_CLEAR = 0.008    # ...with their bottom this far up when let go
    DROP_CLEAR = 0.03    # basket: let go with the item's bottom this far off its floor
    HOVER = 0.06         # hover this far above the release point first
    # The arm reaches high better further out: a can over the person's palm
    # needs the hand at 0.77 m, reachable 0.30 m ahead but not 0.22 m. So if
    # the closest parking spot fails its IK check, stand further back.
    EXTRA_REACH = (0.0, 0.04, 0.08)
    # The last-metre drive goes in a straight line, so a parking spot whose
    # route clips other furniture is penalised: the bottle hand-over chose to
    # come at the person from the east and drove through the side table.
    ROUTE_CLEARANCE = 0.28     # robot half-diagonal plus a margin
    BLOCKED_COST = 5.0

    def __init__(self, grasp_planner):
        self.gp = grasp_planner      # reach limits, arm offsets, IK checks

    def plan(self, place, side, held_mat_rel, grip_above_bottom, kind="top"):
        """Every feasible way to put the item down at `place`, best first.

        `held_mat_rel` is the hand orientation relative to the base when the
        item was picked (rot_z(-yaw) @ grasp_mat); `grip_above_bottom` is how
        far the grasp point sits above the item's bottom.
        """
        spec = PLACES[place]
        gp, bot = self.gp, self.gp.bot
        s = surface_of(bot.model, bot.data, spec)
        here, yaw_now = bot.position[:2], bot.yaw
        if spec.mode == "drop":
            # Hover over the rim, then lower INSIDE and let go a few cm off the
            # floor. Released above the rim, a can or mug fell ~20 cm, bounced,
            # and one ended up outside the basket.
            n_before = _drops.get((id(bot.model), place), 0)
            local = np.array(DROP_SLOTS[n_before % len(DROP_SLOTS)]) * s.half
            z = s.top + grip_above_bottom + self.DROP_CLEAR
            hover = max(s.rim + grip_above_bottom + self.DROP_CLEAR - z, 0.04)
        else:
            margin = np.minimum(self.SET_INSET, 0.6 * s.half)
            bound = s.half - margin
            local = np.clip(s.axes @ (here - s.center[:2]), -bound, bound)
            z = s.top + grip_above_bottom + self.SET_CLEAR
            hover = self.HOVER
        pxy = s.center[:2] + s.axes.T @ local
        point = np.array([pxy[0], pxy[1], z])
        above = point + np.array([0.0, 0.0, hover])
        from .skills import ApproachPose
        others = [footprint_of(bot.model, bot.data, p.body)
                  for p in PLACES.values() if p.body != spec.body]
        target_box = footprint_of(bot.model, bot.data, spec.body)
        target_body = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, spec.body)

        out = []
        for k in (0, 1):
            for sgn in (1.0, -1.0):
                n = sgn * s.axes[k]                  # outward edge normal
                inset = s.half[k] - float((pxy - s.center[:2]) @ n)
                closest = max(inset + gp.hull_front + gp.CLEARANCE, gp.MIN_REACH)
                heading = float(np.arctan2(-n[1], -n[0]))
                h = np.array([np.cos(heading), np.sin(heading)])
                l = np.array([-h[1], h[0]])
                mat = rot_z(heading) @ held_mat_rel
                for extra in self.EXTRA_REACH:
                    reach = closest + extra
                    if reach > gp.MAX_REACH[kind]:
                        break
                    base_xy = pxy - h * reach - l * gp.lateral[side]
                    err = gp._check_ik(side, base_xy, heading, (above, point), mat)
                    if err is None:
                        continue
                    dist = float(np.linalg.norm(base_xy - here))
                    bearing = float(np.arctan2(*(base_xy - here)[::-1]))
                    turn = (abs(_wrap(bearing - yaw_now)) + abs(_wrap(heading - bearing))
                            if dist > 0.15 else abs(_wrap(heading - yaw_now)))
                    # the drive: here -> entry point -> parking spot. Other
                    # furniture must stay clear all the way; the target itself
                    # only until the final creep in.
                    entry = base_xy - h * ApproachPose.ENTRY
                    clear = (route_clear([here, entry, base_xy], others,
                                         self.ROUTE_CLEARANCE, skip_start=0.3)
                             and route_clear([here, entry], [target_box],
                                             self.ROUTE_CLEARANCE, skip_start=0.3))
                    # and the base's creep while the arm works must not run it
                    # into anything low -- except the place itself, which it
                    # is meant to end up against
                    creep_ok = gp.creep_clear(base_xy, heading, exclude_body=target_body)
                    cost = (dist + 0.35 * turn + 1.5 * reach + 20.0 * err
                            + (0.0 if clear else self.BLOCKED_COST)
                            + (0.0 if creep_ok else self.BLOCKED_COST))
                    out.append(PlaceTarget(side, base_xy, heading, above, point, mat,
                                           reach, cost))
                    break
        out.sort(key=lambda p: p.cost)
        return out

    @staticmethod
    def dropped(bot, place):
        """Record that an item went into `place`, so the next takes another slot."""
        key = (id(bot.model), place)
        _drops[key] = _drops.get(key, 0) + 1

    def reachable_from_here(self, target):
        bot = self.gp.bot
        return self.gp._check_ik(target.side, bot.position[:2], bot.yaw,
                                 (target.above, target.point), target.mat) is not None


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)
