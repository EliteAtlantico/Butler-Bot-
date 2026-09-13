#!/usr/bin/env python3
"""Generate a random furnished house the robot has never seen.

    python tools/make_house.py --seed 5 --out random_house.xml

Rooms are carved by recursive splitting, every wall gets one doorway, and the
furniture is drawn from the same vocabulary and materials as home_search.xml
so the result looks like a home rather than a test fixture.

The point is that nothing about the navigation stack is told any of this. The
layout, the room count, the doorway positions and the furniture are decided
here; the robot arrives knowing only a goal coordinate and whatever its depth
camera can see.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CVS = HERE.parent

WALL_T = 0.11          # wall half-thickness
WALL_H = 1.25          # wall half-height  (2.5 m rooms)
DOOR_W = 1.05          # half-width of a doorway opening
CLEAR = 0.65           # keep furniture this far from doorways and walls


# --------------------------------------------------------------------- rooms
class Room:
    __slots__ = ("x0", "y0", "x1", "y1")

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

    @property
    def w(self): return self.x1 - self.x0

    @property
    def h(self): return self.y1 - self.y0

    @property
    def centre(self): return np.array([(self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2])

    def inset(self, d):
        return Room(self.x0 + d, self.y0 + d, self.x1 - d, self.y1 - d)


def split_rooms(rng, root: Room, depth=2, min_side=3.4):
    """Recursively cut the footprint into rooms; returns rooms and the walls
    between them, each wall carrying the doorway that pierces it."""
    rooms, walls = [root], []
    for _ in range(depth):
        nxt = []
        for r in rooms:
            vertical = r.w >= r.h
            span = r.w if vertical else r.h
            if span < 2 * min_side or rng.random() < 0.18:
                nxt.append(r)
                continue
            lo = (r.x0 if vertical else r.y0) + min_side
            hi = (r.x1 if vertical else r.y1) - min_side
            cut = rng.uniform(lo, hi)
            if vertical:
                a, b = Room(r.x0, r.y0, cut, r.y1), Room(cut, r.y0, r.x1, r.y1)
                door = rng.uniform(r.y0 + DOOR_W + 0.4, r.y1 - DOOR_W - 0.4)
                walls.append(("v", cut, r.y0, r.y1, door))
            else:
                a, b = Room(r.x0, r.y0, r.x1, cut), Room(r.x0, cut, r.x1, r.y1)
                door = rng.uniform(r.x0 + DOOR_W + 0.4, r.x1 - DOOR_W - 0.4)
                walls.append(("h", cut, r.x0, r.x1, door))
            nxt += [a, b]
        rooms = nxt
    return rooms, walls


# ----------------------------------------------------------------- furniture
def box(name, x, y, z, sx, sy, sz, mat, extra=""):
    return (f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}">'
            f'<geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" material="{mat}"{extra}/>'
            f'</body>\n')


PIECES = [
    # name,     footprint (half x, half y), builder
    ("sofa",     (1.00, 0.44)),
    ("table",    (0.78, 0.46)),
    ("shelf",    (0.20, 0.85)),
    ("console",  (0.80, 0.24)),
    ("chair",    (0.22, 0.22)),
    ("crate",    (0.30, 0.30)),
    ("bin",      (0.19, 0.19)),
    ("plant",    (0.20, 0.20)),
]


def piece_xml(kind, idx, x, y, yaw):
    e = f' euler="0 0 {yaw:.3f}"'
    t = ""
    if kind == "sofa":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.20" size="1.00 0.42 0.20" material="fabric"/>\n'
        t += '      <geom type="box" pos="0 0.33 0.46" size="1.00 0.09 0.26" material="fabric"/>\n'
        t += '      <geom type="box" pos="0.92 0 0.40" size="0.08 0.42 0.20" material="cushion"/>\n'
        t += '      <geom type="box" pos="-0.92 0 0.40" size="0.08 0.42 0.20" material="cushion"/>\n    </body>\n'
    elif kind == "table":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.73" size="0.78 0.44 0.03" material="wood"/>\n'
        for sx in (0.70, -0.70):
            for sy in (0.36, -0.36):
                t += (f'      <geom type="box" pos="{sx} {sy} 0.36" size="0.04 0.04 0.36"'
                      ' material="darkwood"/>\n')
        t += '    </body>\n'
    elif kind == "shelf":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.92" size="0.18 0.85 0.92" material="darkwood"/>\n'
        t += '      <geom type="box" pos="0.02 0 1.20" size="0.16 0.80 0.03" material="wood"/>\n'
        t += '      <geom type="box" pos="0.02 0 0.60" size="0.16 0.80 0.03" material="wood"/>\n    </body>\n'
    elif kind == "console":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.275" size="0.80 0.24 0.275" material="darkwood"/>\n'
        t += '      <geom type="box" pos="0 -0.26 0.40" size="0.28 0.02 0.12" material="wood"/>\n    </body>\n'
    elif kind == "chair":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.44" size="0.21 0.21 0.03" material="wood"/>\n'
        t += '      <geom type="box" pos="0 -0.19 0.68" size="0.21 0.03 0.22" material="wood"/>\n'
        for sx in (0.17, -0.17):
            for sy in (0.17, -0.17):
                t += (f'      <geom type="box" pos="{sx} {sy} 0.22" size="0.022 0.022 0.22"'
                      ' material="darkwood"/>\n')
        t += '    </body>\n'
    elif kind == "crate":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="box" pos="0 0 0.26" size="0.30 0.30 0.26" material="carton"/>\n'
        t += '      <geom type="box" pos="0.03 0.02 0.66" size="0.22 0.22 0.14" material="carton"/>\n    </body>\n'
    elif kind == "bin":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="cylinder" pos="0 0 0.22" size="0.19 0.22" material="metal"/>\n    </body>\n'
    elif kind == "plant":
        t += f'    <body name="f{idx}" pos="{x:.3f} {y:.3f} 0"{e}>\n'
        t += '      <geom type="cylinder" pos="0 0 0.16" size="0.17 0.16" material="terracotta" group="3"/>\n'
        t += '      <geom type="cylinder" pos="0 0 0.52" size="0.035 0.24" material="darkwood" group="3"/>\n'
        t += '      <geom type="sphere" pos="0 0 0.92" size="0.32" material="plant" group="3"/>\n'
        t += '      <geom type="mesh" mesh="m_pot" material="terracotta" contype="0" conaffinity="0"/>\n'
        t += ('      <geom type="cylinder" pos="0 0 0.281" size="0.132 0.006" material="soil"'
              ' contype="0" conaffinity="0"/>\n')
        t += ('      <geom type="cylinder" pos="0 0 0.50" size="0.024 0.22" material="darkwood"'
              ' contype="0" conaffinity="0"/>\n')
        t += ('      <geom type="mesh" mesh="m_foliage" pos="0 0 0.92" material="plant"'
              ' contype="0" conaffinity="0"/>\n    </body>\n')
    return t


def overlaps(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


# Pickable items, copied from scene_home.xml so the grasp tuning and the
# committed pick baselines still apply: same sizes, masses, contact settings
# and the geom names handwrist's CATALOGUE closes on.
ITEM_XML = {
    "mug": '''    <body name="mug" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint name="mug_free"/>
      <geom name="mug_body" class="item" type="cylinder" pos="0 0 0.05" size="0.04 0.05" material="mug" mass="0.22"/>
      <geom name="mug_handle" class="item" type="box" pos="0 0.05 0.05" size="0.008 0.012 0.03"
            material="mug" mass="0.03"/>
      <geom class="look" mesh="look_mug" material="look_mug"/>
    </body>\n''',
    "can": '''    <body name="can" pos="{x:.3f} {y:.3f} {z:.3f}">
      <freejoint name="can_free"/>
      <geom name="can_body" class="item" type="cylinder" pos="0 0 0.06" size="0.033 0.06" material="can" mass="0.15"/>
      <geom class="look" mesh="look_can" material="look_can"/>
    </body>\n''',
    "remote": '''    <body name="remote" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 0.4">
      <freejoint name="remote_free"/>
      <geom name="remote_body" class="item" type="box" pos="0 0 0.0125" size="0.09 0.024 0.0125"
            material="remote" mass="0.12"/>
      <geom class="look" mesh="look_remote" material="look_remote"/>
    </body>\n''',
    "box": '''    <body name="box" pos="{x:.3f} {y:.3f} {z:.3f}" euler="0 0 0.3">
      <freejoint name="box_free"/>
      <geom name="box_body" class="item" type="box" pos="0 0 0.025" size="0.06 0.025 0.025" material="box" mass="0.10"/>
      <geom class="look" mesh="look_box" material="look_box"/>
    </body>\n''',
}

# The place destination. handwrist's PLACES looks for a body called `basket`
# whose floor geom is `basket_floor`; the names are the contract.
# The console the items stand on has NO decorative front panel. The general
# furniture version does, and that panel spans x +/-0.28 at z 0.28-0.52 --
# directly in front of a centrally placed item, where it blocks the only
# approach the arm has. A middle item behind it reports "no reachable grasp"
# with no hint that a 2 cm strip of trim is the reason.
ITEM_CONSOLE_XML = '''    <body name="item_console" pos="{x:.3f} {y:.3f} 0">
      <geom type="box" pos="0 0 0.275" size="0.80 0.24 0.275" material="darkwood"/>
    </body>
'''

# Wicker, with a rim and hand-holds that are visual only and stay inside the
# walls' footprint and no higher than their tops -- the place planner reads
# the basket's rim height and footprint off every geom on this body.
BASKET_XML = '''    <body name="basket" pos="{x:.3f} {y:.3f} 0">
      <geom name="basket_floor" type="box" pos="0 0 0.005" size="0.22 0.17 0.005" material="look_wicker"/>
      <geom name="basket_wall_e" type="box" pos=" 0.215 0 0.10" size="0.005 0.17 0.10" material="look_wicker"/>
      <geom name="basket_wall_w" type="box" pos="-0.215 0 0.10" size="0.005 0.17 0.10" material="look_wicker"/>
      <geom name="basket_wall_n" type="box" pos="0  0.165 0.10" size="0.22 0.005 0.10" material="look_wicker"/>
      <geom name="basket_wall_s" type="box" pos="0 -0.165 0.10" size="0.22 0.005 0.10" material="look_wicker"/>
      <geom class="look" type="box" pos=" 0.212 0 0.194" size="0.008 0.170 0.006" material="look_tabletop"/>
      <geom class="look" type="box" pos="-0.212 0 0.194" size="0.008 0.170 0.006" material="look_tabletop"/>
      <geom class="look" type="box" pos="0  0.162 0.194" size="0.220 0.008 0.006" material="look_tabletop"/>
      <geom class="look" type="box" pos="0 -0.162 0.194" size="0.220 0.008 0.006" material="look_tabletop"/>
      <geom class="look" type="box" pos=" 0.2196 0 0.165" size="0.0005 0.050 0.013" rgba="0.08 0.06 0.04 1"/>
      <geom class="look" type="box" pos="-0.2196 0 0.165" size="0.0005 0.050 0.013" rgba="0.08 0.06 0.04 1"/>
    </body>\n'''


def build(seed: int):
    rng = np.random.default_rng(seed)
    W = rng.uniform(11.0, 15.0)
    H = rng.uniform(8.0, 11.0)
    # The robot always spawns at the world origin, so the house is laid out
    # around it rather than the other way round: no --start flag, and no
    # chance of the robot materialising inside a wall.
    X0 = -2.4
    root = Room(X0, -H / 2, X0 + W, H / 2)
    rooms, walls = split_rooms(rng, root, depth=2)

    # Reject a layout that puts a wall or doorway on top of the spawn.
    for kind, cut, lo, hi, door in walls:
        if kind == "v" and abs(cut) < 1.3 and lo - 0.3 <= 0.0 <= hi + 0.3:
            return None
        if kind == "h" and abs(cut) < 1.3 and lo - 0.3 <= 0.0 <= hi + 0.3:
            return None

    parts = [HEADER.format(seed=seed)]
    blocked = []          # xy rectangles nothing may be placed in

    # Shell, in the SHIFTED footprint's coordinates. Using the unshifted W/2
    # and 0 here put the west wall straight through the robot's spawn, which
    # made every generated goal unreachable.
    cx = (root.x0 + root.x1) / 2
    parts.append(box("wall_s", cx, -H / 2, WALL_H, W / 2 + WALL_T, WALL_T, WALL_H, "wall"))
    parts.append(box("wall_n", cx, H / 2, WALL_H, W / 2 + WALL_T, WALL_T, WALL_H, "wall"))
    parts.append(box("wall_w", root.x0, 0.0, WALL_H, WALL_T, H / 2, WALL_H, "wall"))
    parts.append(box("wall_e", root.x1, 0.0, WALL_H, WALL_T, H / 2, WALL_H, "wall"))
    for r in ((root.x0 - WALL_T, -H / 2 - WALL_T, root.x1 + WALL_T, -H / 2 + WALL_T),
              (root.x0 - WALL_T, H / 2 - WALL_T, root.x1 + WALL_T, H / 2 + WALL_T),
              (root.x0 - WALL_T, -H / 2, root.x0 + WALL_T, H / 2),
              (root.x1 - WALL_T, -H / 2, root.x1 + WALL_T, H / 2)):
        blocked.append(r)

    # interior walls, each with one doorway and a lintel over the whole opening
    for i, (kind, cut, lo, hi, door) in enumerate(walls):
        a_lo, a_hi = lo, door - DOOR_W
        b_lo, b_hi = door + DOOR_W, hi
        for j, (s, e) in enumerate(((a_lo, a_hi), (b_lo, b_hi))):
            if e - s < 0.12:
                continue
            if kind == "v":
                parts.append(box(f"iw{i}{j}", cut, (s + e) / 2, WALL_H,
                                 WALL_T, (e - s) / 2, WALL_H, "wall"))
                blocked.append((cut - WALL_T, s, cut + WALL_T, e))
            else:
                parts.append(box(f"iw{i}{j}", (s + e) / 2, cut, WALL_H,
                                 (e - s) / 2, WALL_T, WALL_H, "wall"))
                blocked.append((s, cut - WALL_T, e, cut + WALL_T))
        # lintel across the full opening, above the robot's head
        if kind == "v":
            parts.append(box(f"il{i}", cut, door, 2.15, WALL_T, DOOR_W, 0.35, "wall"))
            blocked.append((cut - 0.9, door - DOOR_W - 0.2, cut + 0.9, door + DOOR_W + 0.2))
        else:
            parts.append(box(f"il{i}", door, cut, 2.15, DOOR_W, WALL_T, 0.35, "wall"))
            blocked.append((door - DOOR_W - 0.2, cut - 0.9, door + DOOR_W + 0.2, cut + 0.9))

    # Keep the spawn clear before anything is placed.
    blocked.append((-0.85, -0.85, 0.85, 0.85))

    # furniture, per room
    idx = 0
    for r in rooms:
        inner = r.inset(CLEAR)
        if inner.w <= 0.6 or inner.h <= 0.6:
            continue
        for _ in range(int(rng.integers(3, 7))):
            kind, (hx, hy) = PIECES[int(rng.integers(0, len(PIECES)))]
            yaw = float(rng.choice([0, np.pi / 2, np.pi, -np.pi / 2]))
            if abs(np.sin(yaw)) > 0.5:
                hx, hy = hy, hx
            placed = False
            for _ in range(50):
                x = rng.uniform(inner.x0 + hx, inner.x1 - hx)
                y = rng.uniform(inner.y0 + hy, inner.y1 - hy)
                rect = (x - hx, y - hy, x + hx, y + hy)
                if any(overlaps(rect, b, 0.55) for b in blocked):
                    continue
                parts.append(piece_xml(kind, idx, x, y, yaw))
                blocked.append(rect)
                idx += 1
                placed = True
                break
            if not placed:
                continue

    # start and goal: different rooms when there is more than one
    def free_spot(room, rng, tries=400):
        inner = room.inset(CLEAR)
        for _ in range(tries):
            p = np.array([rng.uniform(inner.x0, inner.x1), rng.uniform(inner.y0, inner.y1)])
            rect = (p[0] - 0.55, p[1] - 0.55, p[0] + 0.55, p[1] + 0.55)
            if not any(overlaps(rect, b, 0.10) for b in blocked):
                return p
        return None

    # ---- the task: things to pick up, and somewhere to put them ----------
    # Items go on a 0.55 m console within 0.10 m of its edge, or on the floor.
    # Both numbers are measured, not chosen: the arm plans no grasp at all
    # above ~0.55 m, and none more than about 0.10 m onto a surface, because
    # the base has to stop at the furniture and reach the rest of the way.
    task = {"items": [], "basket": None, "console": None}
    order_by_size = sorted(rooms, key=lambda r: -r.w * r.h)
    spawn_room = min(rooms, key=lambda r: np.linalg.norm(r.centre))
    far_rooms = [r for r in order_by_size if r is not spawn_room] or [spawn_room]

    # console + items in the room furthest from the spawn
    item_room = max(far_rooms, key=lambda r: np.linalg.norm(r.centre))
    inner = item_room.inset(CLEAR + 0.2)
    cx, cy = inner.centre if inner.w > 0 else item_room.centre
    console_yaw = 0.0
    placed_console = False
    for _ in range(80):
        x = rng.uniform(inner.x0 + 0.85, inner.x1 - 0.85)
        y = rng.uniform(inner.y0 + 0.30, inner.y1 - 0.30)
        rect = (x - 0.82, y - 0.26, x + 0.82, y + 0.26)
        if any(overlaps(rect, b, 0.85) for b in blocked):
            continue
        parts.append(ITEM_CONSOLE_XML.format(x=x, y=y))
        blocked.append(rect)
        task["console"] = (x, y)
        placed_console = True
        break
    if not placed_console:
        return None

    cx, cy = task["console"]
    # three items along the console, each 0.10 m in from its near (-y) edge
    for k, name in enumerate(("mug", "can", "remote")):
        parts.append(ITEM_XML[name].format(x=cx - 0.40 + 0.40 * k, y=cy - 0.14, z=0.55))
        task["items"].append(name)
    # one on the floor, out in the open, so a floor grasp is exercised too
    for _ in range(120):
        x = rng.uniform(inner.x0, inner.x1)
        y = rng.uniform(inner.y0, inner.y1)
        rect = (x - 0.45, y - 0.45, x + 0.45, y + 0.45)
        if any(overlaps(rect, b, 0.30) for b in blocked):
            continue
        parts.append(ITEM_XML["box"].format(x=x, y=y, z=0.0))
        blocked.append((x - 0.12, y - 0.12, x + 0.12, y + 0.12))
        task["items"].append("box")
        break

    # the basket goes in the SPAWN room, so every put crosses a doorway
    binner = spawn_room.inset(CLEAR + 0.2)
    for _ in range(160):
        x = rng.uniform(binner.x0, binner.x1)
        y = rng.uniform(binner.y0, binner.y1)
        if np.hypot(x, y) < 1.6:                 # not on top of the robot
            continue
        rect = (x - 0.30, y - 0.25, x + 0.30, y + 0.25)
        if any(overlaps(rect, b, 0.75) for b in blocked):
            continue
        parts.append(BASKET_XML.format(x=x, y=y))
        blocked.append(rect)
        task["basket"] = (x, y)
        break
    if task["basket"] is None or len(task["items"]) < 3:
        return None

    start = np.zeros(2)                      # the robot's own spawn
    # Goal in the room furthest from the spawn, so the route has to cross the
    # house and pass through at least one doorway.
    goal = None
    for r in sorted(rooms, key=lambda r: -np.linalg.norm(r.centre - start)):
        g = free_spot(r, rng)
        if g is not None and np.linalg.norm(g - start) > 0.45 * W:
            goal = g
            break
    if goal is None:
        return None

    # lights, one per room
    for i, r in enumerate(rooms):
        c = r.centre
        parts.append(f'    <light pos="{c[0]:.2f} {c[1]:.2f} 2.35" dir="0 0 -1" '
                     f'directional="false" diffuse="0.40 0.37 0.33" '
                     f'attenuation="1 0.05 0.02" cutoff="72"/>\n')
    parts.append(FOOTER)
    return "".join(parts), start, goal, len(rooms), (W, H), task


HEADER = '''<mujoco model="random_house_{seed}">
  <!-- Generated by tools/make_house.py. Layout, room count, doorway positions
       and furniture are all chosen by the generator; the navigation stack is
       told none of it and arrives with a goal coordinate and a depth camera. -->
  <include file="../main_mujoco/chopped_dynamic.xml"/>
  <compiler meshdir="../main_mujoco/meshes/" texturedir="assets/"/>
  <include file="assets/household.xml"/>
  <option noslip_iterations="5"/>

  <statistic center="6 0 1.0" extent="8.0"/>
  <visual>
    <headlight diffuse="0.46 0.45 0.42" ambient="0.38 0.37 0.35" specular="0.05 0.05 0.05"/>
    <rgba haze="0.2 0.2 0.22 1"/>
    <global azimuth="150" elevation="-25" offwidth="1280" offheight="960"/>
    <map znear="0.01" zfar="60"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.62 0.68 0.76" rgb2="0.24 0.27 0.32" width="512" height="3072"/>
    <texture type="2d" name="t_floor" file="floor_oak.png"/>
    <texture type="2d" name="t_wall"  file="wall_plaster.png"/>
    <texture type="2d" name="t_sofa"  file="sofa_weave.png"/>
    <texture type="2d" name="t_walnut" file="wood_walnut.png"/>
    <texture type="2d" name="t_oak"   file="wood_oak.png"/>
    <texture type="2d" name="t_metal" file="metal_brushed.png"/>
    <mesh name="m_foliage" file="../../comp_vision_sim/assets/foliage.stl"/>
    <mesh name="m_pot"     file="../../comp_vision_sim/assets/pot.stl"/>

    <material name="floorwood" texture="t_floor" texuniform="true" texrepeat="1.6 1.6"
              reflectance="0.05" specular="0.18" shininess="0.25"/>
    <material name="wall"    texture="t_wall" texuniform="true" texrepeat="4 3" specular="0.05"/>
    <material name="wood"    texture="t_oak" texuniform="true" texrepeat="2 2" specular="0.3" shininess="0.35"/>
    <material name="darkwood" texture="t_walnut" texuniform="true" texrepeat="2 2" specular="0.25" shininess="0.3"/>
    <material name="fabric"  texture="t_sofa" texuniform="true" texrepeat="3 3" specular="0.04"/>
    <material name="cushion" texture="t_sofa" texuniform="true" texrepeat="2 2" rgba="1.22 1.22 1.22 1"
              specular="0.04"/>
    <material name="metal"   texture="t_metal" texuniform="true" texrepeat="1 1" specular="0.75" shininess="0.72"
              reflectance="0.12"/>
    <material name="carton"  rgba="0.72 0.58 0.38 1" specular="0.05"/>
    <!-- Seen in the baseline demo (seed 35) with handwrist's calibrated-colour
         estimator: it matched this foliage as the can (green) and estimated a
         0.47 m wide "can" on plant f4. Pick now finds items by what they are
         (handwrist.detection), which does not confuse a plant with a can; the
         colour estimator is kept only for comparison. -->
    <material name="plant"   rgba="0.22 0.42 0.20 1" specular="0.22" shininess="0.35"/>
    <material name="terracotta" rgba="0.68 0.38 0.26 1" specular="0.10"/>
    <material name="soil"    rgba="0.20 0.15 0.11 1" specular="0.02"/>
    <material name="basketwv" rgba="0.74 0.66 0.50 1" specular="0.06"/>
    <!-- item colours, unchanged from scene_home.xml. They colour the collision
         primitives, which are not rendered: cameras see each item's "look"
         geom (assets/household.xml). -->
    <material name="mug"    rgba="0.85 0.18 0.18 1"/>
    <material name="can"    rgba="0.15 0.70 0.25 1"/>
    <material name="remote" rgba="0.50 0.20 0.75 1"/>
    <material name="box"    rgba="0.10 0.65 0.70 1"/>
  </asset>

  <!-- Item contact settings, from scene_home.xml. The noslip pass stops a
       squeezed item creeping out of the fingers during a long carry, which is
       exactly what a cross-room fetch is. -->
  <default>
    <default class="item">
      <geom condim="4" friction="1.0 0.01 0.001" solref="0.005 1" priority="1" group="3"/>
    </default>
    <!-- what the cameras see (assets/household.xml): no mass, no contacts;
         group 2, which handwrist's footprints, surfaces and rays ignore -->
    <default class="look">
      <geom type="mesh" contype="0" conaffinity="0" group="2" mass="0"/>
    </default>
  </default>

  <worldbody>
    <light pos="6 0 6" dir="0 0 -1" directional="true" diffuse="0.22 0.21 0.19"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="floorwood"
          friction="1.6 0.01 0.001" condim="4"/>
'''

FOOTER = '''  </worldbody>
</mujoco>
'''


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(CVS / "random_house.xml"))
    a = p.parse_args()

    rng_seed = a.seed
    result = None
    while result is None:
        result = build(rng_seed)
        rng_seed += 1000
    xml, start, goal, nrooms, (W, H), task = result
    Path(a.out).write_text(xml, encoding="utf-8")
    print(f"seed {a.seed}: {nrooms} rooms in {W:.1f} x {H:.1f} m")
    print(f"start ({start[0]:.2f}, {start[1]:.2f})   goal ({goal[0]:.2f}, {goal[1]:.2f})"
          f"   {np.linalg.norm(goal-start):.1f} m apart")
    print(f"wrote {a.out}")
    print(f"items {task['items']} on a console at "
          f"({task['console'][0]:.2f}, {task['console'][1]:.2f}); "
          f"basket at ({task['basket'][0]:.2f}, {task['basket'][1]:.2f})")
    print(f"\n  python run_navigation.py --scene {Path(a.out).name} --detector geometric \\\n"
          f"      --goal {goal[0]:.2f},{goal[1]:.2f} --viewer")


if __name__ == "__main__":
    main()
