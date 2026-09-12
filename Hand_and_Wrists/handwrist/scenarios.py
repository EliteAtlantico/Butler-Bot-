"""Where things start: random but sensible placements for tests and demos.

Each object has a home surface in scene_home.xml. A scenario puts the object
somewhere random on that surface (random yaw too -- the wrist has to cope),
parks the robot 1.3-1.9 m away roughly facing it, and moves every other
object out of the way so a trial tests one grasp and nothing else.

Why that far: the head camera is tilted down 22 deg from 1.54 m up, so it
cannot see the floor nearer than ~1.25 m or a coffee-table top nearer than
~0.9 m. A robot that starts closer is blind to the thing it was asked to
pick. Starts are also rejected if the robot would stand in furniture or have
furniture between it and the item -- the last-metre approach does not steer
round obstacles.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .objects import CATALOGUE, set_object_pose
from .places import reset_drops

# object -> surface it is tested on
SURFACE = {
    "mug": "coffee_table",
    "can": "coffee_table",
    "remote": "coffee_table",
    "bottle": "side_table",
    "keys": "floor",
    "ball": "floor",
    "box": "floor",
}

COFFEE_TOP, COFFEE_NEAR_X = 0.40, 0.95     # top height, near edge (robot side)
SIDE_TOP, SIDE_NEAR_Y = 0.60, 1.18

# furniture footprints in scene_home.xml: (centre x, centre y, half x, half y)
FURNITURE = {
    "coffee_table": (1.25, 0.0, 0.30, 0.45),
    "side_table": (0.0, 1.4, 0.22, 0.22),
    "basket": (0.2, -1.3, 0.20, 0.15),
    "person": (-1.0, 1.55, 0.20, 0.30),      # body plus outstretched arm
}
ROBOT_CLEAR = 0.45        # m from the base centre to any furniture edge


def _dist_to_box(p, box):
    cx, cy, hx, hy = box
    dx = max(abs(p[0] - cx) - hx, 0.0)
    dy = max(abs(p[1] - cy) - hy, 0.0)
    return float(np.hypot(dx, dy))


def _clear(p, clearance=ROBOT_CLEAR):
    return all(_dist_to_box(p, b) >= clearance for b in FURNITURE.values())


def _route_clear(a, b, stop_short, clearance=0.35):
    """Straight route from a toward b, ignoring the last `stop_short` m."""
    v = np.asarray(b, float) - a
    n = float(np.linalg.norm(v))
    for s in np.arange(0.0, max(n - stop_short, 0.0), 0.05):
        if not _clear(a + v * (s / n), clearance):
            return False
    return True


def place_robot(bot, xy, yaw):
    """Teleport the robot, standing, and re-anchor odometry and the LQR."""
    q = bot.data.qpos
    q[0:2] = xy
    q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    bot.data.qvel[0:6] = 0.0
    mujoco.mj_forward(bot.model, bot.data)
    bot._odom_ref = bot.wheel_angles.copy()
    bot.balance.reset_reference(bot.state)


def park_others(bot, keep):
    """Line every other object up out of the way, behind the robot."""
    others = [n for n in CATALOGUE if n != keep]
    for i, n in enumerate(others):
        set_object_pose(bot.model, bot.data, n, [-3.5, -1.5 + 0.5 * i, 0.0005])


def _facing(robot, obj, rng, jitter=0.25):
    return float(np.arctan2(obj[1] - robot[1], obj[0] - robot[0])
                 + rng.uniform(-jitter, jitter))


def sample(name, rng):
    """(object xyz, object yaw, robot xy, robot yaw) for one random trial."""
    surface = SURFACE[name]
    yaw = float(rng.uniform(-np.pi, np.pi))
    for _ in range(200):
        if surface == "coffee_table":
            obj = np.array([COFFEE_NEAR_X + rng.uniform(0.07, 0.17),
                            rng.uniform(-0.30, 0.30), COFFEE_TOP + 0.0005])
            robot = np.array([rng.uniform(-0.75, -0.25),
                              obj[1] + rng.uniform(-0.4, 0.4)])
            stop = 0.7
        elif surface == "side_table":
            obj = np.array([rng.uniform(-0.12, 0.12),
                            SIDE_NEAR_Y + rng.uniform(0.05, 0.12), SIDE_TOP + 0.0005])
            robot = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.55, -0.15)])
            stop = 0.7
        else:
            obj = np.array([rng.uniform(-1.4, -0.8), rng.uniform(-0.6, 0.6), 0.0005])
            a = float(rng.uniform(-0.7, 0.7))
            robot = obj[:2] + rng.uniform(1.4, 1.8) * np.array([np.cos(a), np.sin(a)])
            stop = 0.3
        if _clear(robot) and _route_clear(robot, obj[:2], stop):
            return obj, yaw, robot, _facing(robot, obj, rng)
    raise RuntimeError(f"no clear start found for {name}")


TIDY_ITEMS = ("mug", "can", "remote")


def setup_tidy(bot, rng=None):
    """The coffee table with the mug, can and remote scattered along its near
    band (rng) or where the scene puts them (None), and the robot standing
    in front of it. Everything else stays where the scene has it."""
    bot.reset()
    reset_drops()
    if rng is not None:
        slots = rng.permutation([-0.30, 0.0, 0.30])
        for name, y in zip(TIDY_ITEMS, slots):
            pos = [COFFEE_NEAR_X + rng.uniform(0.08, 0.15), y + rng.uniform(-0.06, 0.06),
                   COFFEE_TOP + 0.0005]
            set_object_pose(bot.model, bot.data, name, pos, float(rng.uniform(-np.pi, np.pi)))
        robot = np.array([rng.uniform(-0.5, -0.1), rng.uniform(-0.3, 0.3)])
        place_robot(bot, robot, float(rng.uniform(-0.2, 0.2)))
    else:
        place_robot(bot, [0.0, 0.0], 0.0)
    mujoco.mj_forward(bot.model, bot.data)
    bot.balance.enable(bot.state)


def setup(bot, name, rng=None):
    """Reset and stage one trial. rng=None keeps the scene's own layout."""
    bot.reset()
    reset_drops()
    if rng is not None:
        obj, yaw, robot, ryaw = sample(name, rng)
        park_others(bot, name)
        set_object_pose(bot.model, bot.data, name, obj, yaw)
        place_robot(bot, robot, ryaw)
    else:
        place_robot(bot, [0.0, 0.0], 0.0)
    mujoco.mj_forward(bot.model, bot.data)
    bot.balance.enable(bot.state)
