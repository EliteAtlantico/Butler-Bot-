"""Where things start: random but sensible placements for tests and demos.

Each object has a home surface in scene_home.xml. A scenario puts the object
somewhere random on that surface (random yaw too -- the wrist has to cope),
parks the robot 1 m or so away roughly facing it, and moves every other
object out of the way so a trial tests one grasp and nothing else.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .objects import CATALOGUE, set_object_pose

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


def sample(name, rng):
    """(object xyz, object yaw, robot xy, robot yaw) for one random trial."""
    surface = SURFACE[name]
    yaw = float(rng.uniform(-np.pi, np.pi))
    if surface == "coffee_table":
        obj = np.array([COFFEE_NEAR_X + rng.uniform(0.07, 0.17),
                        rng.uniform(-0.30, 0.30), COFFEE_TOP + 0.0005])
        robot = np.array([rng.uniform(0.05, 0.35), obj[1] + rng.uniform(-0.25, 0.25)])
        ryaw = float(rng.uniform(-0.3, 0.3))
    elif surface == "side_table":
        obj = np.array([rng.uniform(-0.12, 0.12),
                        SIDE_NEAR_Y + rng.uniform(0.05, 0.12), SIDE_TOP + 0.0005])
        robot = np.array([rng.uniform(-0.3, 0.3), rng.uniform(0.25, 0.45)])
        ryaw = float(np.pi / 2 + rng.uniform(-0.3, 0.3))
    else:
        obj = np.array([rng.uniform(-1.4, -0.7), rng.uniform(-0.6, 0.6), 0.0005])
        a = float(rng.uniform(-0.8, 0.8))
        robot = obj[:2] + rng.uniform(0.9, 1.3) * np.array([np.cos(a), np.sin(a)])
        ryaw = float(np.arctan2(obj[1] - robot[1], obj[0] - robot[0])
                     + rng.uniform(-0.4, 0.4))
    return obj, yaw, robot, ryaw


def setup(bot, name, rng=None):
    """Reset and stage one trial. rng=None keeps the scene's own layout."""
    bot.reset()
    if rng is not None:
        obj, yaw, robot, ryaw = sample(name, rng)
        park_others(bot, name)
        set_object_pose(bot.model, bot.data, name, obj, yaw)
        place_robot(bot, robot, ryaw)
    else:
        place_robot(bot, [0.0, 0.0], 0.0)
    mujoco.mj_forward(bot.model, bot.data)
    bot.balance.enable(bot.state)
