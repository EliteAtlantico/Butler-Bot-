#!/usr/bin/env python3
"""How good is the camera's estimate of each item? No driving involved.

    python eval_vision.py                 # 20 random placements per item
    python eval_vision.py -n 50 --objects remote ball

Stages the same random trials as eval_pick.py, looks once with the head
camera from the start pose, and compares the estimate with the simulator's
ground truth. For scale: a top grasp opens the fingers ~45 mm wider than the
item, so the centre can be ~20 mm off across the fingers before a finger
lands on top of it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    import handwrist
    from bracketbot_sim.robot import BracketBot
    from handwrist import scenarios
    from handwrist.objects import CATALOGUE, truth_estimate
    from handwrist.vision import CameraEstimator

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs="+", default=list(CATALOGUE), choices=list(CATALOGUE))
    p.add_argument("-n", "--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bot = BracketBot(xml=str(handwrist.HOME_SCENE))
    see = CameraEstimator()
    print(f"{'item':7s} {'seen':>6s} {'centre err mm':>22s} {'height mm':>10s} "
          f"{'width mm':>13s} {'axis err deg':>14s}")
    print(f"{'':7s} {'':>6s} {'mean':>7s}{'p90':>7s}{'max':>7s} {'mean':>10s} "
          f"{'bias':>6s}{'|err|':>7s} {'mean':>7s}{'max':>7s}")
    for name in args.objects:
        spec = CATALOGUE[name]
        rng = np.random.default_rng([args.seed, list(CATALOGUE).index(name)])
        xy, dz, dw, dyaw, seen = [], [], [], [], 0
        for _ in range(args.trials):
            scenarios.setup(bot, name, rng)
            truth = truth_estimate(bot.model, bot.data, spec)
            est = see(bot, spec)
            if est is None:
                continue
            seen += 1
            xy.append(np.linalg.norm(est.center[:2] - truth.center[:2]) * 1000)
            dz.append(abs(est.top_z - truth.top_z) * 1000)
            dw.append((est.width - truth.width) * 1000)
            if est.axis_yaw is not None and truth.axis_yaw is not None:
                period = np.pi if spec.aligned else 2 * np.pi
                e = (est.axis_yaw - truth.axis_yaw + period / 2) % period - period / 2
                dyaw.append(abs(np.rad2deg(e)))
        if not seen:
            print(f"{name:7s} {0:>3d}/{args.trials:<2d}  never seen")
            continue
        xy, dw = np.array(xy), np.array(dw)
        yaw = (f"{np.mean(dyaw):7.1f}{np.max(dyaw):7.1f}" if dyaw
               else f"{'-':>7s}{'-':>7s}")
        print(f"{name:7s} {seen:>3d}/{args.trials:<2d} {xy.mean():7.1f}{np.percentile(xy, 90):7.1f}"
              f"{xy.max():7.1f} {np.mean(dz):10.1f} {dw.mean():+6.1f}{np.abs(dw).mean():7.1f} {yaw}"
              + (f"   (axis found {len(dyaw)}/{seen})" if spec.handle_geom else ""))
    bot.close()


if __name__ == "__main__":
    main()
