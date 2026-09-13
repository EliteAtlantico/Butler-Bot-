#!/usr/bin/env python3
"""Object detection + path planning from the simulated RGB-D camera.

    ./run_navigation.py                    # headless run, writes vision_nav.png
    ./run_navigation.py --viewer           # watch it in the MuJoCo viewer
    ./run_navigation.py --frames           # also dump per-frame perception PNGs
    ./run_navigation.py --detector yolo    # use the net from train_yolo.py

The robot spins once to map the room with its depth camera, finds the red
column, then A*s a path around the barrier it can see and drives it --
replanning as the map fills in.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The robot model and its BracketBot wrapper live next door; this package
# deliberately does not depend on them, so only the entry points bridge over.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "main_mujoco"))

# Ground truth for scoring the detector only -- the navigator never reads it.
# Extended objects are segments: scoring a 5.3 m wall against its centre point
# would report a metre of "error" for a perfectly good detection.
TRUE_OBJECTS = {
    "target":  [((6.0, 0.0), (6.0, 0.0))],
    "barrier": [((3.0, -4.0), (3.0, 1.3)), ((3.0, 2.7), (3.0, 5.0))],
    "pillar":  [((1.5, -1.3), (1.5, -1.3)), ((4.7, 3.1), (4.7, 3.1)),
                ((5.2, -2.6), (5.2, -2.6))],
}
TRUE_MARKERS = {"target": [(6.0, 0.0)],
                "barrier": [(3.0, -1.35), (3.0, 3.85)],
                "pillar": [(1.5, -1.3), (4.7, 3.1), (5.2, -2.6)]}


def scene_has_truth(scene_path) -> bool:
    """Ground truth above describes one specific course; scoring against it
    in any other scene would invent errors out of nothing."""
    return "obstacle_course" in str(scene_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=str(HERE / "obstacle_course.xml"))
    p.add_argument("--duration", "-d", type=float, default=90.0)
    p.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    p.add_argument("--speed", type=float, default=1.0, help="viewer realtime factor")
    p.add_argument("--out", default="vision_nav.png", help="diagnostic figure")
    p.add_argument("--frames", action="store_true",
                   help="also write rgbd_frame_XX.png every few seconds")
    p.add_argument("--seed-scan", type=float, default=1.0,
                   help="turns to spin while mapping before planning")
    p.add_argument("--detector", default="colour",
                   choices=["colour", "yolo", "geometric"],
                   help="colour thresholding, a YOLO net trained by "
                        "train_yolo.py, or 'geometric' -- colour-free "
                        "clustering that needs no palette and so works in an "
                        "unfamiliar scene")
    p.add_argument("--goal", default=None, metavar="X,Y",
                   help="drive to this world coordinate instead of looking for "
                        "a coloured target; needs no detector palette")
    p.add_argument("--camera", default=None,
                   help="camera to perceive through (default: auto-detected)")
    p.add_argument("--animate-movers", type=float, default=0.0, metavar="MPS",
                   help="shuttle every mocap body in the scene across the route "
                        "at this speed, to exercise moving obstacles")
    p.add_argument("--weights", default=None,
                   help="YOLO weights (default: runs/bracketbot_yolo/weights/best.pt)")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence")
    return p.parse_args()


def build_detector(args, info=None):
    if args.detector == "colour":
        return None
    if args.detector == "geometric":
        from vision_sim.perception import GeometricDetector
        det = GeometricDetector(
            floor_z=info.floor_z if info else 0.0,
            self_radius=(info.robot_radius + 0.25) if info else 0.55)
        print("detector: geometric (colour-free clustering)")
        return det
    from vision_sim.yolo_detector import YoloDetector
    # A fresh training run wins over the checked-in net, so retraining takes
    # effect without passing --weights; models/ is the fallback that makes a
    # clean clone work at all.
    trained = HERE / "runs" / "bracketbot_yolo" / "weights" / "best.pt"
    shipped = HERE / "models" / "bracketbot_yolo.pt"
    weights = args.weights or (trained if trained.exists() else shipped)
    det = YoloDetector(weights, conf=args.conf)
    print(f"detector: YOLO {weights} classes={list(det.names.values())}")
    return det


def annotate(rgb, detections, scale=3):
    """Draw detection boxes and labels onto the RGB frame."""
    from PIL import Image, ImageDraw
    colours = {"target": (255, 70, 70), "barrier": (255, 175, 40),
               "pillar": (90, 170, 255)}
    img = Image.fromarray(rgb).resize(
        (rgb.shape[1] * scale, rgb.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    for d in detections:
        u0, v0, u1, v1 = (v * scale for v in d.bbox)
        c = colours.get(d.label, (255, 255, 255))
        draw.rectangle([u0, v0, u1, v1], outline=c, width=2)
        tag = f"{d.label} {d.distance:.1f}m"
        tw = draw.textlength(tag)
        draw.rectangle([u0, max(v0 - 13, 0), u0 + tw + 6, max(v0 - 13, 0) + 13],
                       fill=c)
        draw.text((u0 + 3, max(v0 - 13, 0) + 1), tag, fill=(0, 0, 0))
    return img


def depth_image(depth, max_range=12.0):
    """Depth -> RGB, near=warm far=cool, no-return=black."""
    import numpy as np
    finite = np.isfinite(depth)
    norm = np.zeros_like(depth)
    if finite.any():
        norm[finite] = np.clip(depth[finite] / max_range, 0, 1)
    out = np.zeros((*depth.shape, 3), np.uint8)
    out[..., 0] = ((1 - norm) * 255 * finite).astype(np.uint8)
    out[..., 1] = ((1 - np.abs(norm - 0.5) * 2) * 255 * finite).astype(np.uint8)
    out[..., 2] = (norm * 255 * finite).astype(np.uint8)
    return out


def _point_segment_distance(p, a, b):
    import numpy as np
    p, a, b = np.asarray(p), np.asarray(a), np.asarray(b)
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def score_detections(detections):
    """Distance from each detection to the nearest true object of its class."""
    rows = []
    for d in detections:
        truths = TRUE_OBJECTS.get(d.label, [])
        if not truths:
            continue
        err = min(_point_segment_distance(d.position[:2], a, b)
                  for a, b in truths)
        rows.append((d, err))
    return rows


def figure(bot, nav, track, path_out, elapsed, truth=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Rectangle, Circle

    grid = nav.grid
    # The last frame is the robot's nose against the column; the frame that
    # saw the most objects is the one worth showing.
    obs = nav.best_obs if nav.best_obs is not None else nav.obs
    dets = nav.best_detections if nav.best_obs is not None else nav.detections
    fig = plt.figure(figsize=(16, 9), facecolor="#14161a")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], hspace=0.18, wspace=0.12)

    # --- RGB with detections -------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(annotate(obs.rgb, dets))
    ax.set_title(f"RGB + object detection (colour x geometry)  t={nav.best_time:.1f}s",
                 color="w", fontsize=11)
    ax.axis("off")

    # --- depth ----------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(depth_image(obs.depth, nav.max_range))
    ax.set_title("registered depth  (red = near, blue = far, black = no return)",
                 color="w", fontsize=11)
    ax.axis("off")

    # --- map + plan -------------------------------------------------------
    ax = fig.add_subplot(gs[:, 1])
    ax.set_facecolor("#14161a")
    occ = grid.occupied
    blocked = nav.blocked if nav.blocked is not None else occ
    ext = [grid.origin[0], grid.origin[0] + grid.size[0] * grid.resolution,
           grid.origin[1], grid.origin[1] + grid.size[1] * grid.resolution]

    shade = np.zeros((*grid.size, 4), np.float32)
    shade[grid.unknown] = (0.16, 0.17, 0.20, 1.0)          # never observed
    shade[grid.seen & ~occ] = (0.24, 0.30, 0.34, 1.0)      # observed free
    shade[blocked & ~occ] = (0.45, 0.32, 0.15, 1.0)        # inflation
    shade[occ] = (0.95, 0.62, 0.15, 1.0)                   # occupied
    ax.imshow(np.transpose(shade, (1, 0, 2)), origin="lower", extent=ext)

    if truth:
        for label, pts in TRUE_MARKERS.items():
            for i, (x, y) in enumerate(pts):
                ax.plot(x, y, "x", color="#6ef0a0", ms=9, mew=2,
                        label="ground truth" if (label == "target" and i == 0) else None)

    for d in dets:
        ax.add_patch(Circle(d.position[:2], 0.18, fill=False, lw=2,
                            color={"target": "#ff5555", "barrier": "#ffaa30",
                                   "pillar": "#5aa8ff"}.get(d.label, "w")))

    if len(track) > 1:
        ax.plot(track[:, 0], track[:, 1], "-", color="#7fd4ff", lw=2,
                label="driven track")
    if nav.path:
        p = np.array(nav.path)
        ax.plot(p[:, 0], p[:, 1], "--o", color="#ffe680", lw=1.6, ms=4,
                label="A* plan (last)")
    if nav.goal_xy is not None:
        ax.plot(*nav.goal_xy, "*", color="#ff5555", ms=18,
                label="detected goal")
    ax.plot(*bot.position[:2], "o", color="w", ms=9, label="robot")

    # Frame the grid the run actually used, not the original course's extent,
    # or a larger scene silently draws its track off the edge of the axes.
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_title("occupancy grid from depth + A* plan", color="w", fontsize=11)
    ax.tick_params(colors="#99a")
    for s in ax.spines.values():
        s.set_color("#33363d")
    leg = ax.legend(loc="upper left", fontsize=8, facecolor="#1c1f25",
                    edgecolor="#33363d")
    for t in leg.get_texts():
        t.set_color("#ccd")

    scored = score_detections(dets) if truth else []
    lines = [f"state: {nav.state}    sim {bot.time:.1f}s / wall {elapsed:.1f}s",
             f"occupied cells {int(occ.sum())}   observed {int(grid.seen.sum())}"
             f" / {grid.size[0] * grid.size[1]}"]
    if nav.goal_xy is not None and truth:
        err = np.linalg.norm(nav.goal_xy - np.array(TRUE_MARKERS["target"][0]))
        lines.append(f"goal estimate error {err:.2f} m")
    elif nav.goal_xy is not None:
        lines.append(f"goal ({nav.goal_xy[0]:.2f}, {nav.goal_xy[1]:.2f})")
    if scored:
        worst = max(e for _, e in scored)
        lines.append(f"{len(scored)} detections, worst position error {worst:.2f} m")
    fig.text(0.515, 0.035, "\n".join(lines), color="#aab", fontsize=9,
             family="monospace", va="bottom")

    fig.suptitle("BracketBot - RGB-D object detection and path planning in MuJoCo",
                 color="w", fontsize=14, y=0.97)
    fig.savefig(path_out, dpi=110, facecolor=fig.get_facecolor())
    print(f"wrote {path_out}")


def main():
    args = parse_args()
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer else (
        "wgl" if os.name == "nt" else "egl")

    import numpy as np
    from bracketbot_sim.robot import BracketBot
    from vision_sim.navigation import VisualNavigator

    from vision_sim.scene import SceneInfo

    bot = BracketBot(xml=args.scene)
    bot.balance.enable(bot.state)

    info = SceneInfo.from_model(bot.model, bot.data, camera=args.camera)
    goal = None
    if args.goal:
        goal = [float(v) for v in args.goal.replace(" ", "").split(",")[:2]]
    nav = VisualNavigator.for_bot(bot, goal=goal, camera=args.camera,
                                  scan_turns=args.seed_scan, verbose=True,
                                  detector=build_detector(args, info))
    print("cameras:", ", ".join(bot.camera_names))
    print(info.describe())
    print(f"scene: {args.scene}")
    print("goal: " + (f"coordinate {tuple(goal)}" if goal else
                      f"whatever the detector labels {nav.goal_label!r}"))

    track = []
    wall = time.time()
    frame_no, next_frame = 0, 0.0

    # Mocap bodies are kinematic: nothing in the physics moves them, so if the
    # scene has any and the caller asked for motion, we drive them ourselves.
    movers = [b for b in range(bot.model.nbody) if bot.model.body_mocapid[b] >= 0]
    home = {b: bot.model.body_pos[b].copy() for b in movers}
    if movers and args.animate_movers > 0:
        print(f"animating {len(movers)} mocap body(s) at "
              f"{args.animate_movers:.1f} m/s")

    def drive_movers(amplitude=3.0):
        if not movers or args.animate_movers <= 0:
            return
        phase = (bot.time * args.animate_movers / (2 * amplitude)) % 2.0
        offset = -amplitude + 2 * amplitude * (phase if phase < 1 else 2 - phase)
        for b in movers:
            p = home[b].copy()
            p[1] = offset
            bot.data.mocap_pos[bot.model.body_mocapid[b]] = p

    def tick():
        nonlocal frame_no, next_frame
        track.append(bot.position[:2].copy())
        if args.frames and nav.obs is not None and bot.time >= next_frame:
            next_frame = bot.time + 4.0
            annotate(nav.obs.rgb, nav.detections).save(
                f"rgbd_frame_{frame_no:02d}.png")
            frame_no += 1

    if args.viewer:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(bot.model, bot.data) as viewer:
            start = time.time()
            # Deliberately not stopping on nav.done: the navigator holds
            # station once it arrives, and closing the window at the moment
            # of success gives you nothing to look at.
            while viewer.is_running() and bot.time < args.duration \
                    and not bot.fallen:
                drive_movers()
                bot.step(0.05, controller=nav)
                tick()
                viewer.sync()
                lag = bot.time / max(args.speed, 1e-6) - (time.time() - start)
                if lag > 0:
                    time.sleep(lag)
    else:
        while bot.time < args.duration and not bot.fallen and not nav.done:
            drive_movers()
            bot.step(0.1, controller=nav)
            tick()

    elapsed = time.time() - wall
    track = np.array(track)
    truth = scene_has_truth(args.scene)
    print(f"\nstate={nav.state} fallen={bot.fallen} "
          f"sim={bot.time:.1f}s wall={elapsed:.1f}s")
    print(f"final position {np.round(bot.position[:2], 2)}")
    if nav.goal_xy is not None:
        print(f"  {np.linalg.norm(bot.position[:2] - nav.goal_xy):.2f} m from the goal")
    print(f"best frame t={nav.best_time:.1f}s, "
          f"{len(nav.best_detections)} objects"
          f"{':' if truth else ' (no ground truth for this scene)'}")
    if truth:
        for d, err in score_detections(nav.best_detections):
            print(f"  {d}  position error {err:.2f} m")

    if nav.obs is not None:
        figure(bot, nav, track, args.out, elapsed, truth=truth)
    bot.close()


if __name__ == "__main__":
    main()
