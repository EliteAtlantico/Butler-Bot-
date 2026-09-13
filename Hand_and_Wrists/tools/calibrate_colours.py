#!/usr/bin/env python3
"""Calibrate the colour windows vision.py uses to find each item.

    python tools/calibrate_colours.py            # writes handwrist/colours.json

Renders the head camera from many random viewpoints with MuJoCo's
segmentation renderer alongside the colour image, so every pixel is labelled
with the item it belongs to. From those labels it picks, per item, a hue
window and saturation / brightness floors, then reports how well the window
separates the item from everything else in the same frames.

The segmentation renders are only used here, offline -- the same idea as
comp_vision_sim's auto-labelled YOLO dataset. At run time the detector sees
nothing but colour and depth.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")

import mujoco  # noqa: E402

import handwrist  # noqa: E402
from bracketbot_sim.robot import BracketBot  # noqa: E402
from handwrist import scenarios  # noqa: E402
from handwrist.objects import CATALOGUE  # noqa: E402
from handwrist.vision import CAMERA, COLOURS_FILE, ColourWindow  # noqa: E402
from vision_sim.perception import rgb_to_hsv  # noqa: E402

W, H = 640, 480
VIEWS_PER_ITEM = 10
HUE_MARGIN = 8.0          # deg added either side of the item's 1-99 % hue range


def circ_window(h):
    """Hue window covering 1-99 % of `h` (degrees), handling the 0/360 wrap."""
    ref = np.rad2deg(np.arctan2(np.sin(np.deg2rad(h)).mean(), np.cos(np.deg2rad(h)).mean()))
    rel = (h - ref + 180.0) % 360.0 - 180.0
    lo, hi = np.percentile(rel, [1, 99])
    return ((ref + lo - HUE_MARGIN) % 360.0, (ref + hi + HUE_MARGIN) % 360.0)


def main():
    bot = BracketBot(xml=str(handwrist.HOME_SCENE))
    m, d = bot.model, bot.data
    seg = mujoco.Renderer(m, H, W)
    seg.enable_segmentation_rendering()
    bodies = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in CATALOGUE}

    item_hsv = {n: [] for n in CATALOGUE}
    frames = []                               # (hsv, body-id image) for scoring
    rng = np.random.default_rng(123)
    for name in CATALOGUE:
        for _ in range(VIEWS_PER_ITEM):
            scenarios.setup(bot, name, rng)
            mujoco.mj_forward(m, d)
            rgb = bot.camera(CAMERA, W, H)
            seg.update_scene(d, camera=CAMERA)
            s = seg.render()
            geom = s[..., 0]
            is_geom = (s[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (geom >= 0)
            body = np.where(is_geom, m.geom_bodyid[np.clip(geom, 0, None)], -1)
            hsv = rgb_to_hsv(rgb)
            item_hsv[name].append(hsv[body == bodies[name]])
            frames.append((hsv, body))

    windows = {}
    for name, chunks in item_hsv.items():
        a = np.vstack(chunks)
        if len(a) < 50:
            raise RuntimeError(f"{name}: only {len(a)} pixels seen; check scenarios")
        hue = circ_window(a[:, 0])
        sat_min = float(max(0.2, np.percentile(a[:, 1], 1) - 0.05))
        val_min = float(max(0.1, np.percentile(a[:, 2], 1) - 0.08))
        windows[name] = ColourWindow(tuple(round(x, 1) for x in hue),
                                     round(sat_min, 3), round(val_min, 3))

    print(f"{'item':7s} {'hue window':>16s} {'sat>=':>6s} {'val>=':>6s} "
          f"{'recall':>7s} {'precision':>9s}   ({len(frames)} frames)")
    for name, win in windows.items():
        tp = fp = fn = 0
        for hsv, body in frames:
            pred = win.mask(hsv)
            truth = body == bodies[name]
            tp += int((pred & truth).sum())
            fp += int((pred & ~truth).sum())
            fn += int((~pred & truth).sum())
        print(f"{name:7s} {win.hue[0]:7.1f}-{win.hue[1]:6.1f} {win.sat_min:6.2f} "
              f"{win.val_min:6.2f} {tp / max(tp + fn, 1):7.1%} {tp / max(tp + fp, 1):9.1%}")

    COLOURS_FILE.write_text(json.dumps(
        {n: {"hue": list(w.hue), "sat_min": w.sat_min, "val_min": w.val_min}
         for n, w in windows.items()}, indent=2) + "\n")
    print(f"\nwrote {COLOURS_FILE}")
    seg.close()
    bot.close()


if __name__ == "__main__":
    main()
