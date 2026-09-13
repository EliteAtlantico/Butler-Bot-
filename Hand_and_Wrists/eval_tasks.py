#!/usr/bin/env python3
"""Chore benchmark: does the item end up where it was asked to go?

    python eval_tasks.py                        # basket + handover for every item, and tidy
    python eval_tasks.py --modes handover -n 6
    python eval_tasks.py --truth                # told where items are, no camera

Modes:
  basket    put <item> in the basket     (random item placement, robot 1.3-1.9 m away)
  handover  fetch <item> to the person   (same placements; ends on the person's palm)
  tidy      coffee table -> basket       (mug, can, remote scattered on the table)

A trial passes only if the task says it succeeded AND the simulator agrees
two seconds later: the item is inside the basket / resting on the palm (for
tidy: all three in the basket), and the robot is still standing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _where(m, d, name):
    """(centre of the gripped part, its height) for item `name`."""
    import mujoco

    from handwrist.objects import CATALOGUE
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, CATALOGUE[name].grasp_geom)
    return d.geom_xpos[g].copy()


def run_trial(job):
    mode, item, seed, vision, max_time = job
    import handwrist
    from bracketbot_sim.robot import BracketBot
    from handwrist import scenarios
    from handwrist.places import surface_of
    from handwrist.skills import truth_estimator
    from handwrist.tasks import make_task

    rng = np.random.default_rng(seed)
    bot = BracketBot(xml=str(handwrist.HOME_SCENE))
    m, d = bot.model, bot.data
    if vision:
        # items found by what they are; YOLO only, so a run does not depend on
        # whether an LLM server happens to be up
        from handwrist.detection import DetectionEstimator
        see = DetectionEstimator(backend="yolo")
    else:
        see = truth_estimator
    if mode == "tidy":
        scenarios.setup_tidy(bot, rng)
        task = make_task(bot, "tidy", estimator=see, verbose=False)
        items = list(scenarios.TIDY_ITEMS)
    else:
        scenarios.setup(bot, item, rng)
        action, to = ("put", "basket") if mode == "basket" else ("fetch", "person")
        task = make_task(bot, action, estimator=see, verbose=False, item=item, to=to)
        items = [item]

    wall = time.time()
    while bot.time < max_time and not task.done and not bot.fallen:
        bot.step(0.1, controller=task)
    bot.step(2.0, controller=task)                 # let things settle

    basket = surface_of(m, d, "basket")
    palm = surface_of(m, d, "person")
    placed = []
    for name in items:
        p = _where(m, d, name)
        if mode == "handover":
            ok = palm.contains(p, 0.01) and palm.top < p[2] < palm.top + 0.15
        else:
            # inside the basket's outline, resting in it -- possibly on top of
            # an item already there, which puts it a little above the rim
            ok = basket.contains(p, -0.005) and p[2] < basket.rim + 0.12
        placed.append(bool(ok))
    truth_ok = all(placed) and not bot.fallen
    out = dict(mode=mode, item=item or "+".join(items), seed=seed,
               success=bool(task.succeeded and truth_ok), task_says=task.succeeded,
               placed=f"{sum(placed)}/{len(placed)}", fallen=bool(bot.fallen),
               missed=[n for n, ok in zip(items, placed) if not ok],
               where={n: [round(float(v), 3) for v in _where(m, d, n)] for n in items},
               failure="" if task.succeeded and truth_ok else (
                   task.failure or ("task said done, sim disagrees" if task.succeeded
                                    else f"ran out of time: {task.status}")),
               sim_time=round(bot.time, 1), wall_time=round(time.time() - wall, 1),
               steps=[(lbl, ok) for lbl, ok, _ in task.results], note=task.note)
    bot.close()
    return out


def main():
    from handwrist.objects import CATALOGUE

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", nargs="+", default=["basket", "handover", "tidy"],
                   choices=["basket", "handover", "tidy"])
    p.add_argument("--objects", nargs="+", default=list(CATALOGUE), choices=list(CATALOGUE))
    p.add_argument("-n", "--trials", type=int, default=4, help="per item (tidy: total)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=6,
                   help="camera rendering needs a GL context per worker; >6 can fail")
    p.add_argument("--truth", action="store_true")
    p.add_argument("--out", default=str(HERE / "results" / "task_eval"))
    args = p.parse_args()

    jobs = []
    for mode in args.modes:
        if mode == "tidy":
            jobs += [("tidy", None, args.seed * 1000 + 900 + i, not args.truth, 420.0)
                     for i in range(args.trials)]
        else:
            jobs += [(mode, n, args.seed * 1000 + i, not args.truth, 180.0)
                     for n in args.objects for i in range(args.trials)]
    print(f"{len(jobs)} trials on {args.workers} workers "
          f"({'told where items are' if args.truth else 'camera'}) ...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_trial, jobs):
            results.append(r)
            print(f"  {r['mode']:8s} {r['item']:15s} seed {r['seed']:5d}  "
                  f"{'PASS' if r['success'] else 'FAIL'}  {r['sim_time']:5.1f}s  "
                  f"placed {r['placed']}  {r['failure']}"
                  + (f"  missed {r['missed']} at {[r['where'][n] for n in r['missed']]}"
                     if r["missed"] else ""), flush=True)

    lines = ["| Task | Item | Success | Mean time | Failures |", "|---|---|---|---|---|"]
    for mode in args.modes:
        keys = sorted({r["item"] for r in results if r["mode"] == mode},
                      key=lambda k: list(CATALOGUE).index(k) if k in CATALOGUE else 99)
        for k in keys:
            rs = [r for r in results if r["mode"] == mode and r["item"] == k]
            ok = [r for r in rs if r["success"]]
            fails = {}
            for r in rs:
                if not r["success"]:
                    fails[r["failure"]] = fails.get(r["failure"], 0) + 1
            tm = f"{np.mean([r['sim_time'] for r in ok]):.0f} s" if ok else "-"
            lines.append(f"| {mode} | {k} | {len(ok)}/{len(rs)} | {tm} | "
                         + ("; ".join(f"{a} x{b}" for a, b in fails.items()) or "-") + " |")
    total = sum(r["success"] for r in results)
    lines.append(f"| **all** | | **{total}/{len(results)}** | | |")
    table = "\n".join(lines)
    print("\n" + table + f"\n\n({time.time() - t0:.0f} s wall)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    out.with_suffix(".md").write_text(
        f"# Chore evaluation ({'told where items are' if args.truth else 'camera'})\n\n"
        f"{len(jobs)} trials, seed {args.seed}.\n\n{table}\n")
    print(f"wrote {out.with_suffix('.md')} and .json")


if __name__ == "__main__":
    main()
