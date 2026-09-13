#!/usr/bin/env python3
"""Watch the BracketBot pick something up.

    python run_pick.py --object remote              # viewer, scene's own layout
    python run_pick.py --object keys --random 3     # random placement, seed 3
    python run_pick.py --object mug --headless      # no window, just the log

Objects: mug, can, remote (coffee table), bottle (side table), keys, ball,
box (floor). In the viewer: Space pauses, R restarts the pick.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--object", "-o", default="remote")
    p.add_argument("--random", type=int, default=None, metavar="SEED",
                   help="random placement instead of the scene's layout")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--vision", action="store_true",
                   help="find the item with the head camera instead of being told")
    p.add_argument("--duration", "-d", type=float, default=None,
                   help="sim seconds to run (headless default 90; viewer: until closed)")
    p.add_argument("--speed", type=float, default=1.0, help="realtime factor (viewer)")
    p.add_argument("--trace", action="store_true", help="log the base every second (headless)")
    return p.parse_args()


def contacts_str(bot, pick):
    """Robot-vs-world contacts other than the wheels on the floor."""
    import mujoco

    m, d = bot.model, bot.data
    robot = pick.grippers["right"].robot_bodies
    name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or \
        f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])}#{g}"
    pairs = set()
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        r1, r2 = m.geom_bodyid[g1] in robot, m.geom_bodyid[g2] in robot
        if r1 == r2:
            continue
        rg, wg = (g1, g2) if r1 else (g2, g1)
        if name(wg) == "floor" and name(rg).endswith("_tire"):
            continue
        pairs.add(f"{name(rg)}~{name(wg)}")
    return ("  CONTACT " + ", ".join(sorted(pairs))) if pairs else ""


def main():
    args = parse_args()
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "wgl" if args.headless else "glfw"

    import numpy as np

    import handwrist
    from bracketbot_sim.robot import BracketBot
    from handwrist import scenarios
    from handwrist.skills import Pick

    bot = BracketBot(xml=str(handwrist.HOME_SCENE))

    def start():
        rng = None if args.random is None else np.random.default_rng(args.random)
        scenarios.setup(bot, args.object, rng)
        print(f"picking the {args.object}" + (" (finding it with the camera)"
                                              if args.vision else ""))
        if args.vision:
            from handwrist.detection import DetectionEstimator
            return Pick(bot, args.object, estimator=DetectionEstimator())
        return Pick(bot, args.object)

    pick = start()

    if args.headless:
        next_trace = 0.0
        duration = args.duration or 90.0
        while bot.time < duration and not pick.done:
            bot.step(0.1, controller=pick)
            if args.trace and bot.time >= next_trace:
                next_trace += 1.0
                ap = getattr(pick, "approach", None)
                err = pick.arm.grasp_pose[0] if pick.plan else None
                print(f"      t={bot.time:5.1f} {pick.phase:8s} "
                      f"{getattr(ap, 'stage', ''):6s} pos=({bot.position[0]:+.2f},"
                      f"{bot.position[1]:+.2f}) yaw={np.rad2deg(bot.yaw):+6.1f} "
                      f"pitch={np.rad2deg(bot.pitch):+5.1f} v={bot.ground_speed:.2f} "
                      f"lean={bot.com_lean * 1000:+.0f}mm "
                      f"site={np.round(err, 3) if err is not None else '-'}"
                      + contacts_str(bot, pick))
        why = pick.failure or f"ran out of time in {pick.phase}"
        print(f"{'SUCCESS' if pick.succeeded else 'FAILED: ' + why}"
              f"  t={bot.time:.1f}s  retries={pick.retries}")
        bot.close()
        return

    import glfw
    import mujoco.viewer

    state = {"paused": False, "restart": False}

    def on_key(key):
        if key in (glfw.KEY_SPACE, glfw.KEY_P):
            state["paused"] = not state["paused"]
        elif key == glfw.KEY_R:
            state["restart"] = True

    reported = False
    with mujoco.viewer.launch_passive(bot.model, bot.data, key_callback=on_key) as viewer:
        t_wall, t_sim = time.time(), bot.time
        while viewer.is_running():
            if state["restart"]:
                state["restart"] = False
                pick, reported = start(), False
                t_wall, t_sim = time.time(), bot.time
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                t_wall, t_sim = time.time(), bot.time
                continue
            if args.duration and bot.time > args.duration:
                break
            bot.step(0.02, controller=pick)
            viewer.sync()
            if pick.done and not reported:
                reported = True
                print("SUCCESS" if pick.succeeded else f"FAILED: {pick.failure}")
            lag = (bot.time - t_sim) / max(args.speed, 1e-6) - (time.time() - t_wall)
            if lag > 0:
                time.sleep(lag)
    bot.close()


if __name__ == "__main__":
    main()
