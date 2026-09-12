#!/usr/bin/env python3
"""Run the simulated BracketBot.

    ./run.py                         # viewer, balancing, standing still
    ./run.py --algorithm square      # viewer, driving a square patrol
    ./run.py --algorithm avoid       # viewer, depth-camera obstacle avoidance
    ./run.py --headless -d 30 --algorithm avoid --shot cams.png
    ./run.py --cameras               # dump every camera view and exit

In the viewer, press Tab to open the control panel; the camera dropdown there
switches between the free camera and the robot's seven fixed cameras, so you
can watch through the head or a wrist while it drives.
"""
from __future__ import annotations

import argparse
import os
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--algorithm", "-a", default="stand",
                   choices=["stand", "square", "avoid", "waypoints", "drive", "spin"],
                   help="movement algorithm to run (default: stand)")
    p.add_argument("--scene", default="scene_dynamic.xml",
                   help="scene_dynamic.xml (with props) or scene_flat.xml")
    p.add_argument("--headless", action="store_true", help="no viewer window")
    p.add_argument("--duration", "-d", type=float, default=None,
                   help="seconds to run (headless default 30, viewer unlimited)")
    p.add_argument("--speed", type=float, default=1.0, help="realtime factor")
    p.add_argument("--no-balance", action="store_true",
                   help="leave the balance controller off (it will fall over)")
    p.add_argument("--cameras", action="store_true",
                   help="render every camera to a montage and exit")
    p.add_argument("--shot", default=None,
                   help="write a camera montage to this PNG when finished")
    return p.parse_args()


def make_algorithm(name):
    from bracketbot_sim import algorithms as alg
    if name == "stand":
        return alg.Stand()
    if name == "square":
        return alg.square_patrol(side=1.6)
    if name == "avoid":
        return alg.ObstacleAvoider()
    if name == "waypoints":
        return alg.WaypointFollower([(2.0, 0.0), (2.0, 2.0), (0.0, 2.0), (0.0, 0.0)])
    if name == "drive":
        return alg.Sequence(alg.Drive(0.4, 0.0, 5), alg.Drive(0.0, 0.9, 4),
                            alg.Drive(0.4, 0.0, 5), alg.Drive(0.0, -0.9, 4),
                            alg.Drive(0.0, 0.0, 3))
    if name == "spin":
        return alg.Drive(0.0, 1.0, 1e9)
    raise ValueError(name)


def montage(bot, path, width=320, height=240):
    """Every camera side by side, with depth false-coloured, as one PNG."""
    import numpy as np
    from PIL import Image

    tiles = []
    for name in bot.camera_names:
        tiles.append((name, bot.camera(name, width, height)))
    d = bot.depth("head_depth", width, height, max_range=8.0)
    finite = np.isfinite(d)
    norm = np.zeros_like(d)
    if finite.any():
        lo, hi = d[finite].min(), d[finite].max()
        norm[finite] = (d[finite] - lo) / max(hi - lo, 1e-6)
    # near = warm, far = cool, no-return = black
    rgb = np.zeros((*d.shape, 3), np.uint8)
    rgb[..., 0] = ((1 - norm) * 255 * finite).astype(np.uint8)
    rgb[..., 1] = ((1 - np.abs(norm - 0.5) * 2) * 255 * finite).astype(np.uint8)
    rgb[..., 2] = (norm * 255 * finite).astype(np.uint8)
    tiles.append(("head_depth (metres, red=near)", rgb))

    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * height), (20, 20, 24))
    for i, (name, img) in enumerate(tiles):
        sheet.paste(Image.fromarray(img), ((i % cols) * width, (i // cols) * height))
    sheet.save(path)
    print(f"wrote {path}  ({len(tiles)} views: "
          + ", ".join(n for n, _ in tiles) + ")")


def main():
    args = parse_args()
    # The offscreen camera renders need a GL backend. egl works headless; with a
    # viewer we must share the windowed backend instead.
    os.environ["MUJOCO_GL"] = "egl" if (args.headless or args.cameras) else "glfw"

    from bracketbot_sim.robot import BracketBot

    bot = BracketBot(xml=args.scene)
    print(bot.plant.describe())
    print("cameras:", ", ".join(bot.camera_names))

    if args.cameras:
        montage(bot, args.shot or "cameras.png")
        bot.close()
        return

    if not args.no_balance:
        bot.balance.enable(bot.state)
    algorithm = make_algorithm(args.algorithm)
    print(f"algorithm: {args.algorithm}")

    if args.headless:
        duration = args.duration or 30.0
        while bot.time < duration and not bot.fallen:
            bot.step(0.1, controller=algorithm)
        import numpy as np
        print(f"t={bot.time:.1f}s  pos={np.round(bot.position[:2], 3)}  "
              f"pitch={np.rad2deg(bot.pitch):+.2f}deg  fallen={bot.fallen}")
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(bot.model, bot.data) as viewer:
            start = time.time()
            while viewer.is_running():
                if args.duration and bot.time > args.duration:
                    break
                bot.step(0.02, controller=algorithm)
                viewer.sync()
                lag = bot.time / max(args.speed, 1e-6) - (time.time() - start)
                if lag > 0:
                    time.sleep(lag)

    if args.shot:
        montage(bot, args.shot)
    bot.close()


if __name__ == "__main__":
    main()
