#!/usr/bin/env python3
"""Randomised house layouts: does navigation ever hit anything?

    ./stress_test.py                 # 12 random houses
    ./stress_test.py -n 40           # more
    ./stress_test.py --seed 7 --keep # keep the generated XML to inspect

Each trial builds a different floor plan -- random room size, one or two
interior walls with doorways at random positions and widths, and a random
spread of furniture -- then drops the robot somewhere free and asks for a
goal somewhere else free, which may be in another room, behind furniture, or
unreachable. The point is not that it always arrives; it is that it never
collides and never falls, whatever it is asked to do.

Collisions are read straight out of MuJoCo's contact list, so this measures
what actually touched rather than what the planner believed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "main_mujoco"))

import numpy as np

HEADER = """<mujoco model="house_{i}">
  <include file="../main_mujoco/chopped_dynamic.xml"/>
  <compiler meshdir="../main_mujoco/meshes/"/>
  <visual><global offwidth="1280" offheight="960"/><map znear="0.01" zfar="60"/></visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.5 0.54 0.58" rgb2="0.15 0.17 0.2" width="512" height="3072"/>
    <texture type="2d" name="f" builtin="checker" mark="edge" rgb1="0.50 0.39 0.27"
             rgb2="0.44 0.34 0.23" markrgb="0.33 0.25 0.17" width="300" height="300"/>
    <material name="f" texture="f" texuniform="true" texrepeat="12 12" reflectance="0.05"/>
    <material name="wall" rgba="0.86 0.84 0.80 1"/>
    <material name="wood" rgba="0.55 0.36 0.20 1"/>
    <material name="dark" rgba="0.36 0.24 0.14 1"/>
    <material name="fab"  rgba="0.35 0.40 0.48 1"/>
    <material name="met"  rgba="0.62 0.63 0.66 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="f"
          friction="1.6 0.01 0.001" condim="4"/>
"""
FOOTER = "  </worldbody>\n</mujoco>\n"


def _box(name, x, y, z, sx, sy, sz, mat):
    return (f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}">'
            f'<geom type="box" size="{sx:.3f} {sy:.3f} {sz:.3f}" material="{mat}"/>'
            f'</body>\n')


def _rects_overlap(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


def random_house(rng, i):
    """Returns (xml, start_xy, goal_xy, blockers) for one random floor plan."""
    W = rng.uniform(9.0, 15.0)
    H = rng.uniform(7.0, 11.0)
    x0, y0, x1, y1 = 0.0, -H / 2, W, H / 2
    parts = [HEADER.format(i=i)]
    blockers = []          # xy rectangles furniture/walls occupy

    t = 0.12               # wall half-thickness
    parts.append(_box("w_s", (x0 + x1) / 2, y0, 1.2, W / 2, t, 1.2, "wall"))
    parts.append(_box("w_n", (x0 + x1) / 2, y1, 1.2, W / 2, t, 1.2, "wall"))
    parts.append(_box("w_w", x0, 0, 1.2, t, H / 2, 1.2, "wall"))
    parts.append(_box("w_e", x1, 0, 1.2, t, H / 2, 1.2, "wall"))
    for r in ((x0 - t, y0 - t, x1 + t, y0 + t), (x0 - t, y1 - t, x1 + t, y1 + t),
              (x0 - t, y0 - t, x0 + t, y1 + t), (x1 - t, y0 - t, x1 + t, y1 + t)):
        blockers.append(r)

    # interior walls, each with one doorway wide enough to pass (>= 0.9 m)
    for k in range(rng.integers(1, 3)):
        wx = rng.uniform(x0 + 2.5, x1 - 2.5)
        door_w = rng.uniform(0.95, 1.7)
        door_y = rng.uniform(y0 + 1.2 + door_w / 2, y1 - 1.2 - door_w / 2)
        lo_lo, lo_hi = y0, door_y - door_w / 2
        hi_lo, hi_hi = door_y + door_w / 2, y1
        if lo_hi - lo_lo > 0.15:
            parts.append(_box(f"iw{k}a", wx, (lo_lo + lo_hi) / 2, 1.2,
                              t, (lo_hi - lo_lo) / 2, 1.2, "wall"))
            blockers.append((wx - t, lo_lo, wx + t, lo_hi))
        if hi_hi - hi_lo > 0.15:
            parts.append(_box(f"iw{k}b", wx, (hi_lo + hi_hi) / 2, 1.2,
                              t, (hi_hi - hi_lo) / 2, 1.2, "wall"))
            blockers.append((wx - t, hi_lo, wx + t, hi_hi))

    # furniture
    kinds = ["table", "sofa", "shelf", "crate", "bin", "chair"]
    for k in range(rng.integers(6, 14)):
        kind = kinds[rng.integers(0, len(kinds))]
        sx, sy, sz = {
            "table": (0.75, 0.45, 0.38), "sofa": (1.0, 0.42, 0.4),
            "shelf": (0.2, 0.85, 0.95), "crate": (0.3, 0.3, 0.3),
            "bin": (0.2, 0.2, 0.25), "chair": (0.22, 0.22, 0.45),
        }[kind]
        if rng.random() < 0.5:
            sx, sy = sy, sx
        mat = {"table": "wood", "sofa": "fab", "shelf": "dark",
               "crate": "wood", "bin": "met", "chair": "wood"}[kind]
        for _ in range(40):
            cx = rng.uniform(x0 + 0.8, x1 - 0.8)
            cy = rng.uniform(y0 + 0.8, y1 - 0.8)
            r = (cx - sx, cy - sy, cx + sx, cy + sy)
            if any(_rects_overlap(r, b, 0.75) for b in blockers):
                continue
            parts.append(_box(f"f{k}", cx, cy, sz, sx, sy, sz, mat))
            blockers.append(r)
            break

    # a start and a goal in free space, far apart
    def free_point():
        for _ in range(400):
            p = np.array([rng.uniform(x0 + 1.0, x1 - 1.0),
                          rng.uniform(y0 + 1.0, y1 - 1.0)])
            r = (p[0] - 0.55, p[1] - 0.55, p[0] + 0.55, p[1] + 0.55)
            if not any(_rects_overlap(r, b, 0.12) for b in blockers):
                return p
        return None

    start = free_point()
    goal = None
    for _ in range(60):
        g = free_point()
        if g is not None and start is not None and np.linalg.norm(g - start) > 0.55 * W:
            goal = g
            break
    if start is None or goal is None:
        return None
    return "".join(parts) + FOOTER, start, goal, blockers


def truth_reachable(bot, start, goal, res=0.05, clearance=0.30):
    """Is the goal reachable given PERFECT knowledge of the scene?

    Rasterises the real geometry rather than anything the camera saw, inflates
    by the robot's half-width, and flood-fills. Without this an unreachable
    goal and a planner that failed to find a real route look identical in the
    results table, and only one of them is a bug.
    """
    import mujoco
    from scipy.ndimage import binary_dilation, label as cc_label

    floor = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    lo = np.minimum(start, goal) - 12.0
    hi = np.maximum(start, goal) + 12.0
    nx, ny = int((hi[0] - lo[0]) / res), int((hi[1] - lo[1]) / res)
    occ = np.zeros((nx, ny), bool)

    for g in range(bot.model.ngeom):
        if g == floor or bot.model.body_weldid[bot.model.geom_bodyid[g]] != 0:
            continue
        h = np.abs(bot.data.geom_xmat[g].reshape(3, 3)) @ bot.model.geom_aabb[g, 3:6]
        c = bot.data.geom_xpos[g]
        if c[2] - h[2] > 1.63 or c[2] + h[2] < 0.10:   # over the robot, or flat
            continue
        a = ((c[:2] - h[:2]) - lo) / res
        b = ((c[:2] + h[:2]) - lo) / res
        x0, y0 = max(int(np.floor(a[0])), 0), max(int(np.floor(a[1])), 0)
        x1, y1 = min(int(np.ceil(b[0])), nx), min(int(np.ceil(b[1])), ny)
        if x1 > x0 and y1 > y0:
            occ[x0:x1, y0:y1] = True

    blocked = binary_dilation(occ, iterations=int(round(clearance / res)))
    free = ~blocked
    lab, _ = cc_label(free)
    sc = np.floor((np.asarray(start) - lo) / res).astype(int)
    gc = np.floor((np.asarray(goal) - lo) / res).astype(int)
    for c in (sc, gc):
        np.clip(c, [0, 0], [nx - 1, ny - 1], out=c)
    if lab[tuple(sc)] == 0:
        return None                      # start itself is inside inflation
    return bool(lab[tuple(sc)] == lab[tuple(gc)])


def run_trial(xml_text, start, goal, i, keep=False, duration=180.0):
    import mujoco
    from bracketbot_sim.robot import BracketBot
    from vision_sim.navigation import VisualNavigator
    from vision_sim.perception import GeometricDetector
    from vision_sim.scene import SceneInfo

    path = HERE / (f"house_{i}.xml" if keep else "_stress_tmp.xml")
    path.write_text(xml_text, encoding="utf-8")
    try:
        bot = BracketBot(xml=str(path))
        # drop the robot at the generated start pose
        bot.data.qpos[0:2] = start
        bot.data.qpos[2] = 0.30
        bot.data.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(bot.model, bot.data)
        bot.balance.enable(bot.state)

        floor = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        robot_g = {g for g in range(bot.model.ngeom)
                   if bot.model.body_weldid[bot.model.geom_bodyid[g]] != 0}
        scenery = {g for g in range(bot.model.ngeom)
                   if g not in robot_g and g != floor}

        reachable = truth_reachable(bot, start, goal)

        info = SceneInfo.from_model(bot.model, bot.data)
        det = GeometricDetector(floor_z=info.floor_z,
                                self_radius=info.robot_radius + 0.25)
        nav = VisualNavigator.for_bot(bot, goal=goal, detector=det)

        hits, worst_pen, t0 = 0, 0.0, time.time()
        while bot.time < duration and not bot.fallen and not nav.done:
            bot.step(0.05, controller=nav)
            for c in range(bot.data.ncon):
                con = bot.data.contact[c]
                a, b = con.geom1, con.geom2
                if (a in robot_g and b in scenery) or (b in robot_g and a in scenery):
                    hits += 1
                    worst_pen = max(worst_pen, -float(con.dist))
        return dict(i=i, state=nav.state, hits=hits, pen=worst_pen,
                    reachable=reachable,
                    fell=bool(bot.fallen), sim=bot.time, wall=time.time() - t0,
                    dist=float(np.linalg.norm(bot.position[:2] - goal)),
                    span=float(np.linalg.norm(np.asarray(goal) - np.asarray(start))),
                    bot=bot)
    finally:
        if not keep:
            path.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", type=int, default=12, help="number of houses")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep", action="store_true", help="keep generated XML")
    p.add_argument("--duration", type=float, default=180.0)
    a = p.parse_args()

    import os
    os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")

    rng = np.random.default_rng(a.seed)
    rows = []
    print(f"{'#':>3} | {'span':>5} | {'reach':>5} | {'result':9} | {'left':>5} | "
          f"{'CONTACTS':>8} | {'pen':>7} | {'sim':>6} | verdict")
    print("-" * 92)
    for i in range(a.n):
        house = None
        while house is None:
            house = random_house(rng, i)
        xml_text, start, goal, _ = house
        r = run_trial(xml_text, start, goal, i, keep=a.keep, duration=a.duration)
        bot = r.pop("bot")
        bot.close()
        rows.append(r)
        reach = {True: "yes", False: "no", None: "?"}[r["reachable"]]
        ok_arrive = r["state"] == "arrived"
        if r["fell"]:
            verdict = "FELL"
        elif r["hits"] > 0:
            verdict = "TOUCHED"
        elif ok_arrive:
            verdict = "ok"
        elif r["reachable"] is False:
            verdict = "ok (correctly refused)"
        else:
            verdict = "MISSED reachable goal"
        r["verdict"] = verdict
        print(f"{i:>3} | {r['span']:>4.1f}m | {reach:>5} | {r['state']:9} | "
              f"{r['dist']:>4.1f}m | {r['hits']:>8} | {r['pen']*1000:>5.1f}mm | "
              f"{r['sim']:>5.1f}s | {verdict}", flush=True)

    arrived = sum(r["state"] == "arrived" for r in rows)
    touched = sum(r["hits"] > 0 for r in rows)
    fell = sum(r["fell"] for r in rows)
    reachable = [r for r in rows if r["reachable"] is True]
    got = sum(r["state"] == "arrived" for r in reachable)
    missed = [r["i"] for r in rows if r["verdict"] == "MISSED reachable goal"]
    print("-" * 92)
    print(f"contact-free {len(rows)-touched}/{len(rows)}   falls {fell}/{len(rows)}")
    print(f"arrived {arrived}/{len(rows)} overall; "
          f"{got}/{len(reachable)} of the goals that were actually reachable")
    if missed:
        print(f"missed reachable goals: houses {missed}")
    if touched:
        worst = max(rows, key=lambda r: r["pen"])
        print(f"worst penetration {worst['pen']*1000:.1f} mm in house {worst['i']}")


if __name__ == "__main__":
    main()
