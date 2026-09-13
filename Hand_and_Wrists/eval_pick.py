#!/usr/bin/env python3
"""Pick-success benchmark: random placements, success rate per object.

    python eval_pick.py                          # 8 trials of every object
    python eval_pick.py --vision                 # find items with the camera
    python eval_pick.py --objects remote ball -n 20
    python eval_pick.py --seed 7 --workers 4

Each trial resets the home scene, drops one object at a random spot and yaw
on its surface, parks the robot 1.3-1.9 m away, and runs the Pick skill with
no help. Without --vision the skill is told where the object is; with it, it
has to find it with the head camera. A trial passes only if the skill says
it is holding the object AND the simulator agrees: the object is >= 5 cm
above where it started, within 10 cm of the grasp site, and the robot is
still standing.
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

FURNITURE = ("coffee_table", "side_table", "basket")


def run_trial(job):
    name, seed, max_time, vision = job
    import mujoco

    import handwrist
    from bracketbot_sim.robot import BracketBot
    from handwrist import scenarios
    from handwrist.objects import CATALOGUE, truth_estimate
    from handwrist.skills import Pick, truth_estimator

    rng = np.random.default_rng(seed)
    bot = BracketBot(xml=str(handwrist.HOME_SCENE))
    scenarios.setup(bot, name, rng)
    m, d = bot.model, bot.data
    body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
    start_z = float(d.xpos[body][2])
    furniture = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f) for f in FURNITURE}
    truth = truth_estimate(m, d, CATALOGUE[name])

    if vision:
        # finds the object by what it is; YOLO only, so a run does not depend
        # on whether an LLM server happens to be up
        from handwrist.detection import DetectionEstimator
        estimator = DetectionEstimator(backend="yolo")
    else:
        estimator = truth_estimator
    pick = Pick(bot, name, estimator=estimator, verbose=False)
    robot = pick.grippers["right"].robot_bodies
    grip_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, pick.spec.grasp_geom)
    gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or \
        mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
    bumps = 0
    touched = {}                             # "robot part~thing" -> first phase seen
    wall = time.time()
    while bot.time < max_time and not pick.done and not bot.fallen:
        bot.step(0.1, controller=pick)
        bumped = False
        for i in range(d.ncon):
            g1, g2 = d.contact[i].geom1, d.contact[i].geom2
            b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
            if (b1 in robot) == (b2 in robot):
                continue
            rg, wg = (g1, g2) if b1 in robot else (g2, g1)
            if gname(wg) == "floor" and gname(rg).endswith("_tire"):
                continue
            touched.setdefault(f"{gname(rg)}~{gname(wg)}", pick.phase)
            bumped |= m.geom_bodyid[wg] in furniture
        bumps += bumped
    if pick.succeeded:                       # does it keep holding it?
        bot.step(1.0, controller=pick)

    # judge by the part that was gripped, not the body origin: a bottle's
    # origin is its base, 9 cm below where the fingers are
    grasp_site = pick.arm.grasp_pose[0] if pick.plan else d.geom_xpos[grip_geom]
    lifted = float(d.xpos[body][2] - start_z)
    near = float(np.linalg.norm(d.geom_xpos[grip_geom] - grasp_site))
    truth_ok = lifted > 0.05 and near < 0.10 and not bot.fallen
    plan = pick.plan
    est = pick.first_estimate
    est_err = (round(float(np.linalg.norm(est.center[:2] - truth.center[:2])) * 1000, 1)
               if est is not None else None)
    out = dict(
        object=name, seed=seed, success=bool(pick.succeeded and truth_ok),
        skill_says=pick.succeeded, truth_ok=truth_ok,
        failure=pick.failure or ("" if truth_ok else "skill said held, sim disagrees"
                                 if pick.succeeded else "timed out in " + pick.phase),
        sim_time=round(bot.time, 1), wall_time=round(time.time() - wall, 1),
        retries=pick.retries, lifted=round(lifted, 3), fallen=bool(bot.fallen),
        furniture_bumps=bumps, contacts=touched, estimate_err_mm=est_err,
        searched=pick._search_stops,
        failed_in=pick.history[-2][1] if pick.phase == "failed" and len(pick.history) > 1 else "",
        arm=plan.side if plan else "", grasp=plan.kind if plan else "",
        wrist_deg=round(float(np.rad2deg(plan.wrist_yaw)), 1) if plan else None,
    )
    bot.close()
    return out


def summarise(results):
    from handwrist.objects import CATALOGUE

    names = [n for n in CATALOGUE if any(r["object"] == n for r in results)]
    lines = ["| Object | Success | Mean time | Estimate error | Retries | Furniture bumps | Failures |",
             "|---|---|---|---|---|---|---|"]
    for n in names:
        rs = [r for r in results if r["object"] == n]
        ok = [r for r in rs if r["success"]]
        fails = {}
        for r in rs:
            if not r["success"]:
                fails[r["failure"]] = fails.get(r["failure"], 0) + 1
        t = np.mean([r["sim_time"] for r in ok]) if ok else float("nan")
        errs = [r["estimate_err_mm"] for r in rs if r["estimate_err_mm"] is not None]
        e = f"{np.mean(errs):.0f} mm" if errs else "-"
        lines.append(f"| {n} | {len(ok)}/{len(rs)} | {t:.1f} s | {e} | "
                     f"{sum(r['retries'] for r in rs)} | "
                     f"{sum(r['furniture_bumps'] > 0 for r in rs)} | "
                     + ("; ".join(f"{k} x{v}" for k, v in fails.items()) or "-") + " |")
    total = sum(r["success"] for r in results)
    lines.append(f"| **all** | **{total}/{len(results)}** | | | | | |")
    return "\n".join(lines)


def main():
    from handwrist.objects import CATALOGUE

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--objects", nargs="+", default=list(CATALOGUE), choices=list(CATALOGUE))
    p.add_argument("-n", "--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--max-time", type=float, default=90.0, help="sim seconds per trial")
    p.add_argument("--vision", action="store_true",
                   help="find items with the head camera instead of being told")
    p.add_argument("--out", default=None,
                   help="results path stem (default results/pick_eval[_vision])")
    args = p.parse_args()
    if args.out is None:
        args.out = str(HERE / "results" / ("pick_eval_vision" if args.vision else "pick_eval"))

    jobs = [(n, args.seed * 1000 + i, args.max_time, args.vision)
            for n in args.objects for i in range(args.trials)]
    print(f"{len(jobs)} trials on {args.workers} workers ...")
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for r in pool.map(run_trial, jobs):
            results.append(r)
            print(f"  {r['object']:7s} seed {r['seed']:5d}  "
                  f"{'PASS' if r['success'] else 'FAIL'}  {r['sim_time']:5.1f}s  "
                  f"{r['arm']:5s} {r['grasp']:4s} wrist {r['wrist_deg']}  "
                  f"retries {r['retries']}  {r['failure']}"
                  + (f" (in {r['failed_in']})" if r["failed_in"] else "")
                  + (f"  touched: {r['contacts']}" if not r["success"] else ""), flush=True)
    table = summarise(results)
    print("\n" + table + f"\n\n({time.time() - t0:.0f} s wall)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(results, indent=1))
    how = ("found with the head camera" if args.vision
           else "object positions from the simulator")
    out.with_suffix(".md").write_text(
        f"# Pick evaluation ({how})\n\n{len(jobs)} trials, seed {args.seed}, "
        f"{args.trials} per object.\n\n{table}\n")
    print(f"wrote {out.with_suffix('.md')} and .json")


if __name__ == "__main__":
    main()
