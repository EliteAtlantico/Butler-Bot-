#!/usr/bin/env python3
"""Run the simulated BracketBot.

    ./run.py                         # viewer, balancing, standing still
    ./run.py --algorithm square      # viewer, driving a square patrol
    ./run.py --algorithm avoid       # viewer, depth-camera obstacle avoidance
    ./run.py --headless -d 30 --algorithm avoid --shot cams.png
    ./run.py --cameras               # dump every camera view and exit

RGB-D object detection and path planning live in ../comp_vision_sim; run
./run_navigation.py there.

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
                   choices=["stand", "square", "avoid", "waypoints", "drive",
                            "spin", "pick"],
                   help="movement algorithm to run (default: stand)")
    p.add_argument("--scene", default=None,
                   help="scene_dynamic.xml (props), scene_flat.xml (bare floor), "
                        "or scene_table.xml (obstacle + table + cube). "
                        "Defaults to scene_table.xml for --algorithm pick.")
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


DEFAULT_SCENE = {"pick": "scene_table.xml"}


def make_algorithm(name, bot=None):
    from bracketbot_sim import algorithms as alg
    if name == "pick":
        from bracketbot_sim.manipulation import PickCube
        return PickCube(bot)
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
    # Offscreen camera renders need a GL backend. On Linux they always go
    # through EGL, viewer or not: the passive viewer opens its own GLFW window
    # independently of MUJOCO_GL, and letting the offscreen renderers share that
    # windowed context segfaults the process the first time an algorithm asks
    # for a depth frame. Windows has no egl backend, so headless renders there
    # ride the same wgl context a viewer would use.
    if os.name == "nt":
        os.environ["MUJOCO_GL"] = "wgl" if (args.headless or args.cameras) else "glfw"
    else:
        os.environ["MUJOCO_GL"] = "egl"

    from bracketbot_sim.robot import BracketBot

    scene = args.scene or DEFAULT_SCENE.get(args.algorithm, "scene_dynamic.xml")
    bot = BracketBot(xml=scene)
    print(bot.plant.describe())
    print("cameras:", ", ".join(bot.camera_names))

    if args.cameras:
        montage(bot, args.shot or "cameras.png")
        bot.close()
        return

    if not args.no_balance:
        bot.balance.enable(bot.state)
    # PickCube reads the scene's cube and table when it is built, so the robot
    # has to exist first
    algorithm = make_algorithm(args.algorithm, bot)
    print(f"scene: {scene}   algorithm: {args.algorithm}")

    if args.headless:
        duration = args.duration or (120.0 if args.algorithm == "pick" else 30.0)
        while bot.time < duration and not bot.fallen:
            bot.step(0.1, controller=algorithm)
            if getattr(algorithm, "done", False):
                break
        import numpy as np
        print(f"t={bot.time:.1f}s  pos={np.round(bot.position[:2], 3)}  "
              f"pitch={np.rad2deg(bot.pitch):+.2f}deg  fallen={bot.fallen}")
    else:
        import glfw
        import mujoco.viewer

        # The passive viewer runs no physics of its own (run_physics_thread=
        # False): it is only a camera, and our loop below is the only thing
        # that advances time. The stock 'run/pause' button therefore does
        # nothing to the simulation, and the 'reset' button just rewinds the
        # rendered state without touching the controllers. We wire the keys
        # ourselves so they act on the real simulation.
        paused = {"v": False}
        current = {"algorithm": algorithm}

        def on_key(key):
            # GLFW key codes (press only; the passive bridge delivers repeats,
            # so guard against them).
            if key in (glfw.KEY_SPACE, glfw.KEY_P):
                paused["v"] = not paused["v"]
                print(("paused" if paused["v"] else "resumed") +
                      f"  t={bot.time:.2f}s  pos={bot.position[:2].round(2)}")
            elif key == glfw.KEY_R:
                bot.reset()           # re-anchors odometry + LQR refs
                # Rebind through the holder: assigning `algorithm` here would
                # only create a local, and the loop would keep running the
                # stale instance with its old clock and phase state.
                current["algorithm"] = make_algorithm(args.algorithm, bot)
                print(f"reset  t=0.00s")

        with mujoco.viewer.launch_passive(bot.model, bot.data,
                                          key_callback=on_key) as viewer:
            start = time.time()
            while viewer.is_running():
                if paused["v"]:
                    # Hold the scene still: render, but don't step physics.
                    viewer.sync()
                    time.sleep(0.02)
                    continue
                if args.duration and bot.time > args.duration:
                    break
                bot.step(0.02, controller=current["algorithm"])
                viewer.sync()
                lag = bot.time / max(args.speed, 1e-6) - (time.time() - start)
                if lag > 0:
                    time.sleep(lag)

    if args.shot:
        montage(bot, args.shot)
    bot.close()


if __name__ == "__main__":
    main()
