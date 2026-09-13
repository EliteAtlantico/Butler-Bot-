#!/usr/bin/env python3
"""The baseline simulation: tidy a procedurally generated house.

    python baseline_demo.py                 # headless, writes a report
    python baseline_demo.py --viewer        # watch it
    python baseline_demo.py --seed 41 --items 3

Four things at once, none of them staged:

  1. cross-room navigation  -- the items and the basket are in different
     rooms, so every fetch drives through a doorway;
  2. pick and place         -- a variety of objects off a console and off the
     floor, each carried to the basket and dropped in;
  3. obstacle avoidance     -- sofas, tables, shelves, plants and bins are
     scattered between the two, and the route has to go round them;
  4. a house it has never seen -- layout, doorways and furniture come from
     tools/make_house.py, and the robot is told only where things are, not
     what the building looks like.

Nothing is hand-placed for the robot's benefit, and the run reports what it
actually achieved rather than asserting it worked.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
for _p in (HERE, REPO, REPO / "main_mujoco", REPO / "Hand_and_Wrists", HERE / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=35)
    p.add_argument("--items", type=int, default=4, help="how many to tidy away")
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--scene", default=None, help="reuse an existing house")
    p.add_argument("--budget", type=float, default=150.0,
                   help="sim seconds allowed per leg")
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["MUJOCO_GL"] = "glfw" if args.viewer else "wgl" if os.name == "nt" else "egl"

    import numpy as np
    import mujoco
    import make_house

    # ---------------------------------------------------------- the house
    if args.scene:
        scene = Path(args.scene)
        task = None
    else:
        seed, built = args.seed, None
        while built is None:
            built = make_house.build(seed)
            seed += 1000
        xml, start, goal, nrooms, (W, H), task = built
        scene = HERE / "baseline_house.xml"
        scene.write_text(xml, encoding="utf-8")
        print(f"house: seed {args.seed}, {nrooms} rooms in {W:.1f} x {H:.1f} m")
        print(f"  items {task['items']} on a console at "
              f"({task['console'][0]:.1f}, {task['console'][1]:.1f})")
        print(f"  basket at ({task['basket'][0]:.1f}, {task['basket'][1]:.1f}), "
              f"{np.hypot(*task['basket']):.1f} m from the robot")

    from bracketbot_sim.robot import BracketBot
    from vision_sim.navigation import VisualNavigator
    from vision_sim.perception import GeometricDetector
    from vision_sim.scene import SceneInfo
    from integration.pick_adapter import make_pick
    from handwrist.place import Place

    bot = BracketBot(xml=str(scene))
    bot.balance.enable(bot.state)
    info = SceneInfo.from_model(bot.model, bot.data)
    print(info.describe())

    basket_xy = np.array(task["basket"]) if task else None
    items = (task["items"] if task else ["mug", "can", "remote", "box"])[:args.items]

    # ------------------------------------------------------------ plumbing
    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(bot.model, bot.data)
    wall0, t_wall = time.time(), time.time()

    def run(controller, done, budget, label):
        """Step until `done()` or the budget runs out. One place that pumps
        the viewer, so every leg is watchable rather than only the last."""
        nonlocal t_wall
        t0 = bot.time
        while bot.time - t0 < budget and not bot.fallen and not done():
            if viewer is not None and not viewer.is_running():
                return "window closed"
            bot.step(0.05 if viewer else 0.1, controller=controller)
            if viewer is not None:
                viewer.sync()
                lag = (bot.time - t0) / max(args.speed, 1e-6) - (time.time() - t_wall)
                if lag > 0:
                    time.sleep(min(lag, 0.05))
        t_wall = time.time()
        if bot.fallen:
            return "fell over"
        return None if done() else f"ran out of time in {label}"

    def body_xy(name):
        b = mujoco.mj_name2id(bot.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return bot.data.xpos[b][:2].copy()

    def tick(dt=0.02):
        """One small step, keeping the viewer in sync. Backing away runs on
        this fixed tick so it behaves the same with or without a viewer --
        the legs step 0.05 s with one and 0.1 s without, and the same
        back-away once freed the robot in one and not the other."""
        t = time.time()
        bot.step(dt)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, dt / max(args.speed, 1e-6) - (time.time() - t)))

    def back_away(distance=0.45, speed=0.15, stall_after=3.0, pulse=0.20, max_pulses=3):
        """Reverse straight out from the furniture the arm just worked at,
        before anything turns on the spot.

        Pick parks the base at arm's reach -- measured 0.10 m from the console
        face, the hull touching it -- and turning there swings the hull's
        corner into the furniture and stalls for the rest of the run.

        Reverse, and keep checking whether it is actually going backwards.
        While it is, leave it alone. Only when it has made no backward
        progress for `stall_after` seconds try something else: a balancing
        robot leans back by first rolling FORWARD, which the furniture can
        block, and then a reverse command just sits there (measured 0.00 m,
        the balance loop pressing the hull in harder). The something else is
        a short backward wheel pulse with the balance loop paused -- measured
        from the post-pick pose AT REST, -2 N m for 0.2 s rolls the hull ~8 cm
        clear with a peak lean of 0.12 rad (it falls at 0.6), still holding
        the item -- then straight back to reversing and watching.

        "Not moving" needs patience. After a pulse -- and from standstill -- the
        balance loop has to lean back before it reverses, and it leans back by
        rolling forward first, so for a second or two it is not going
        backwards even though it is about to. Measured from the post-pick pose
        over three lead-ins (at rest, straight after reversing, after a pause):
        with a 1 s patience the pulse freed it in none, 2 s in some, 3 s in
        all three, every one 0.45 m out with a peak lean of 0.13 rad or less
        and the item still held. Longer pulses are not a safe escalation --
        0.30 s tipped it over -- so the pulse stays at the measured length and
        is cut short if the robot leans past 0.25 rad.

        Progress is read off the wheel encoders, which a real robot has.
        """
        nonlocal t_wall
        b = bot.balance
        origin, heading = bot.position[:2].copy(), np.array([np.cos(bot.yaw), np.sin(bot.yaw)])
        odo0 = bot.odometry
        mark_odo, mark_t, t0 = bot.odometry, bot.time, bot.time
        tried = 0
        bot.drive(-speed, 0.0)
        while odo0 - bot.odometry < distance and not bot.fallen and bot.time - t0 < 20.0:
            if viewer is not None and not viewer.is_running():
                break
            tick()
            if mark_odo - bot.odometry >= 0.01:
                mark_odo, mark_t = bot.odometry, bot.time      # going backwards: keep going
            elif bot.time - mark_t >= stall_after:
                if tried == max_pulses:
                    break                                       # out of things to try
                tried += 1
                b.disable()
                peak = 0.0
                for _ in range(int(round(pulse / bot.dt))):
                    bot.set_wheel_torque(-2.0, -2.0)
                    mujoco.mj_step(bot.model, bot.data)
                    peak = max(peak, abs(bot.pitch))
                    if peak > 0.25:
                        break                                   # tipping: stop pushing
                print(f"  no backward progress for {stall_after:.1f} s: "
                      f"backward wheel pulse {tried} ({pulse:.2f} s, peak lean {peak:.2f} rad)")
                if viewer is not None:
                    viewer.sync()
                b.trim_integral = 0.0
                b.enable(bot.state)                             # re-seeds the references here
                bot.drive(-speed, 0.0)
                mark_odo, mark_t = bot.odometry, bot.time      # and watch again
        bot.drive(0.0, 0.0)
        b.reset_reference(bot.state)
        for _ in range(50):
            tick()
        t_wall = time.time()
        truth = float((origin - bot.position[:2]) @ heading)
        print(f"  backed away {odo0 - bot.odometry:.2f} m by the wheels ({truth:.2f} m in the sim)"
              + (f" after {tried} pulse(s)" if tried else "")
              + ("" if odo0 - bot.odometry >= distance else "  -- STILL STUCK"))

    # One grid for the whole run. A fresh navigator per leg would throw the
    # map away and have to rediscover the same walls each time; keeping it
    # means the first trip pays for the scan and later trips plan straight
    # into what the robot already knows -- which is also why the later legs
    # can spin a quarter turn instead of a full one.
    shared_grid = None

    def drive_to(xy, label, first, stop=0.35):
        nonlocal shared_grid
        det = GeometricDetector(floor_z=info.floor_z,
                                self_radius=info.robot_radius + 0.25)
        nav = VisualNavigator.for_bot(bot, goal=np.asarray(xy), detector=det,
                                      grid=shared_grid, stop_distance=stop,
                                      scan_turns=1.0 if first else 0.25)
        shared_grid = nav.grid
        return nav, run(nav, lambda: nav.done, args.budget, label)

    # ------------------------------------------------------------- the run
    log = []
    first_leg = True
    console_xy = np.array(task["console"]) if task else None

    def approach_point(item, here):
        """Where to stand to be able to reach the item.

        Driving at the item itself stops the robot a metre away on whatever
        side it happened to come from -- sometimes behind the console, where
        the arm cannot park and the grasp planner has nothing to offer. Items
        on the console are reachable only from its open (-y) face, so aim at
        that face.

        A floor item can be approached from anywhere, but has to be far enough
        back to be inside the head camera's view -- and the last metre in is
        Pick's own straight-line drive, which does not avoid obstacles. Taking
        simply the side the robot came from once put the stand-off on a table's
        corner; driving in along the table's edge, the stowed hand hooked the
        tabletop for 33 s until Pick timed out. So prefer that side, but take
        the nearest bearing whose stand-off and straight run in are clear on
        the navigator's own map, with room for the arm.
        """
        p = body_xy(item)
        if console_xy is not None and abs(p[1] - console_xy[1]) < 0.5 \
                and abs(p[0] - console_xy[0]) < 1.0:
            return np.array([p[0], console_xy[1] - 1.05])      # the open face
        d = here - p
        base = float(np.arctan2(d[1], d[0])) if np.linalg.norm(d) > 1e-6 else 0.0
        if shared_grid is not None:
            blocked, _ = shared_grid.costmap(robot_radius=info.robot_radius + 0.15)
            for off in np.deg2rad([0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180]):
                u = np.array([np.cos(base + off), np.sin(base + off)])
                stand = p + u * 1.35
                run_in = stand[None, :] - u[None, :] * np.linspace(0.0, 0.90, 19)[:, None]
                cells = shared_grid.to_cell(run_in)
                inside = shared_grid.inside(cells)
                if inside.all() and not blocked[cells[:, 0], cells[:, 1]].any():
                    if off != 0:
                        print(f"  the side it came from is not clear; approaching from "
                              f"{np.degrees(off):+.0f} deg round instead")
                    return stand
            print("  no clear side on the map; approaching from the side it came from")
        return p + np.array([np.cos(base), np.sin(base)]) * 1.35

    for item in items:
        target = approach_point(item, bot.position[:2])
        print(f"\n--- {item} ---")
        print(f"  fetch: standing off at ({target[0]:.2f}, {target[1]:.2f}) "
              f"for an item at ({body_xy(item)[0]:.2f}, {body_xy(item)[1]:.2f})")
        nav, why = drive_to(target, f"driving to the {item}", first_leg)
        first_leg = False
        if why or nav.state != nav.ARRIVED:
            log.append((item, "navigate-to-item", why or nav.state))
            print(f"  FAILED: {why or nav.state}")
            if why == "window closed":
                break
            continue

        pick = make_pick(bot, item, use_camera=True)
        why = run(pick, lambda: pick.done, args.budget, f"picking the {item}")
        if why or not pick.succeeded:
            log.append((item, "pick", why or pick.failure))
            print(f"  FAILED to pick: {why or pick.failure}")
            if why == "window closed":
                break
            continue
        print(f"  picked up the {item}")
        back_away()

        print(f"  carry: driving to the basket ({basket_xy[0]:.2f}, {basket_xy[1]:.2f})")
        nav, why = drive_to(basket_xy, "carrying to the basket", False, stop=0.85)
        if why or nav.state != nav.ARRIVED:
            log.append((item, "navigate-to-basket", why or nav.state))
            print(f"  FAILED: {why or nav.state}")
            if why == "window closed":
                break
            continue

        # Place takes the finished Pick, not the item name: it inherits which
        # arm is holding what, the grasp frame and how high the grip sits above
        # the object's base, none of which it could recover from a string.
        place = Place(bot, pick, "basket")
        why = run(place, lambda: place.done, args.budget, f"placing the {item}")
        ok = (not why) and place.succeeded
        log.append((item, "done" if ok else "place", "ok" if ok else (why or place.failure)))
        print(f"  {'PUT IT IN THE BASKET' if ok else 'FAILED to place: ' + str(why or place.failure)}")
        if why == "window closed":
            break

    # ------------------------------------------------------------- report
    print("\n" + "=" * 62)
    print(f"{'item':8s} {'stage reached':22s} why")
    print("-" * 62)
    for item, stage, why in log:
        print(f"{item:8s} {stage:22s} {why}")
    done = sum(1 for _, s, _ in log if s == "done")
    print("-" * 62)
    print(f"{done}/{len(items)} tidied away   sim {bot.time:.0f}s   "
          f"wall {time.time()-wall0:.0f}s   fallen={bot.fallen}")
    if basket_xy is not None:
        for item in items:
            d = float(np.linalg.norm(body_xy(item) - basket_xy))
            print(f"  {item:8s} ended {d:5.2f} m from the basket"
                  + ("   IN" if d < 0.35 else ""))
    if viewer is not None:
        print("\nclose the viewer window to finish")
        while viewer.is_running():
            bot.step(0.05)
            viewer.sync()
            time.sleep(0.01)
        viewer.close()
    bot.close()


if __name__ == "__main__":
    main()
