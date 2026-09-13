#!/usr/bin/env python3
"""Ask the BracketBot to do a chore.

    python run_task.py fetch --item remote            # find the remote, hand it to the person
    python run_task.py put --item remote --to basket  # pick it up, drop it in the basket
    python run_task.py tidy                           # coffee table -> basket, every item
    python run_task.py pick --item mug                # just pick it up

    --random SEED   scatter the item (fetch/put/pick) or the table (tidy)
    --headless      no window, just the log
    --truth         be told where items are instead of finding them with the camera

Items: mug, can, remote, bottle, ball.
Places: coffee_table, side_table, basket, person.
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
    p.add_argument("action", choices=["fetch", "put", "tidy", "pick"])
    p.add_argument("--item", default=None)
    p.add_argument("--to", default=None, help="place for put/fetch")
    p.add_argument("--surface", default="coffee_table", help="what tidy clears")
    p.add_argument("--into", default="basket", help="where tidy puts things")
    p.add_argument("--random", type=int, default=None, metavar="SEED")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--truth", action="store_true")
    p.add_argument("--duration", "-d", type=float, default=None,
                   help="sim seconds (headless default 360; viewer: until closed)")
    p.add_argument("--speed", type=float, default=1.0, help="realtime factor (viewer)")
    p.add_argument("--trace", type=float, default=0.0, metavar="SEC",
                   help="headless: log the robot every SEC seconds")
    return p.parse_args()


def trace_line(bot, task):
    """Pose, lean, the held item and every robot-vs-world contact."""
    import numpy as np

    from run_pick import contacts_str
    s = task.current or task.last
    line = (f"      t={bot.time:6.1f} {task.status[:46]:46s} "
            f"pos=({bot.position[0]:+.2f},{bot.position[1]:+.2f}) "
            f"yaw={np.rad2deg(bot.yaw):+5.0f} pitch={np.rad2deg(bot.pitch):+5.1f} "
            f"v={bot.ground_speed:.2f} lean={bot.com_lean * 1000:+.0f}mm")
    if s is not None:
        stage = getattr(getattr(s, "approach", None), "stage", "")
        side = getattr(s, "side", None) or (s.plan.side if getattr(s, "plan", None) else None)
        if side:
            hand = s.arms[side].grasp_pose[0]
            line += (f" {stage:5s} hand=({hand[0]:+.2f},{hand[1]:+.2f},{hand[2]:.2f})"
                     f" held={s.grippers[side].pinched_body()}")
        line += contacts_str(bot, s)
    return line


def main():
    args = parse_args()
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "wgl" if args.headless else "glfw"

    import numpy as np

    import handwrist
    from bracketbot_sim.robot import BracketBot
    from handwrist import scenarios
    from handwrist.skills import truth_estimator
    from handwrist.tasks import make_task

    if args.action != "tidy" and args.item is None:
        sys.exit(f"{args.action} needs --item")
    bot = BracketBot(xml=str(handwrist.HOME_SCENE))

    def start():
        rng = None if args.random is None else np.random.default_rng(args.random)
        if args.action == "tidy":
            scenarios.setup_tidy(bot, rng)
        else:
            scenarios.setup(bot, args.item, rng)
        task = make_task(bot, args.action, estimator=truth_estimator if args.truth else None,
                         item=args.item, to=args.to, surface=args.surface, into=args.into)
        print(f"task: {task.label}")
        return task

    def report(task):
        print(("SUCCESS" if task.succeeded else "FAILED") + f"  {task.status}"
              f"  (t={bot.time:.1f}s)")
        for label, ok, why in task.results:
            print(f"   {'ok  ' if ok else 'FAIL'} {label}" + ("" if ok else f" -- {why}"))

    task = start()
    if args.headless:
        duration = args.duration or 360.0
        next_trace = 0.0
        while bot.time < duration and not task.done:
            bot.step(0.1, controller=task)
            if args.trace and bot.time >= next_trace:
                next_trace = bot.time + args.trace
                print(trace_line(bot, task))
        if not task.done:
            print(f"(ran out of time: {task.status})")
        report(task)
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
                task, reported = start(), False
                t_wall, t_sim = time.time(), bot.time
            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                t_wall, t_sim = time.time(), bot.time
                continue
            if args.duration and bot.time > args.duration:
                break
            bot.step(0.02, controller=task)
            viewer.sync()
            if task.done and not reported:
                reported = True
                report(task)
            lag = (bot.time - t_sim) / max(args.speed, 1e-6) - (time.time() - t_wall)
            if lag > 0:
                time.sleep(lag)
    bot.close()


if __name__ == "__main__":
    main()
