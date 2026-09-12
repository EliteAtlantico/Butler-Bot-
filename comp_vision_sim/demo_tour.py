#!/usr/bin/env python3
"""Watch every navigation capability in the MuJoCo viewer, one after another.

    ./demo_tour.py                 # all four scenarios
    ./demo_tour.py --only 3        # just the third
    ./demo_tour.py --list          # what the tour contains

Each scenario opens its own viewer window; close it to move to the next.
Inside a window, Tab opens the control panel -- switch the camera dropdown to
`head_depth` to see exactly what the navigator is planning from.

The last scenario is a known failure, and is included on purpose: it is the
honest edge of what this stack does.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCENARIOS = [
    {
        "name": "Colour detection + path planning",
        "scene": "obstacle_course.xml",
        "args": ["--detector", "colour", "--duration", "70"],
        "watch": [
            "spins ~9 s first -- that is mapping, not indecision;"
            " it can only plan around what it has looked at",
            "drives NORTH-EAST, not at the red column: the amber barrier"
            " blocks the direct route, so A* takes the gap at y~1.9",
            "the column is visible OVER the barrier, which is why it can"
            " localise a goal it has no straight path to",
        ],
    },
    {
        "name": "YOLO detection (trained on auto-labelled sim frames)",
        "scene": "obstacle_course.xml",
        "args": ["--detector", "yolo", "--duration", "70"],
        "watch": [
            "same route, but object classes now come from a neural net",
            "trained only on MuJoCo segmentation renders -- no hand labels",
            "mAP50 0.99; the console prints each detection's class and range",
        ],
    },
    {
        "name": "An unfamiliar scene, coordinate goal, no colour palette",
        "scene": "test_arena.xml",
        "args": ["--detector", "geometric", "--goal", "14,0", "--duration", "120"],
        "watch": [
            "18x16 m -- far outside the grid size that used to be hard-coded",
            "green/purple props the colour detector knows nothing about, so"
            " detection is pure geometry here",
            "S-bend: through the first gap at y~+1.75, then back across to"
            " the second at y~-1.75",
        ],
    },
    {
        "name": "Furnished apartment: real objects, doorway, no palette",
        "scene": "apartment.xml",
        "args": ["--detector", "geometric", "--goal", "9,-2", "--duration", "150"],
        "watch": [
            "furniture built from composed primitives -- table, chairs, sofa,"
            " shelving, plant, cartons -- not coloured cylinders",
            "the doorway at x=5 is the ONLY route between the two rooms",
            "the lintel above that door sits at 1.9 m and the robot is 1.63 m,"
            " so it drives under it; the grid ignores anything above head"
            " height, which is what keeps the doorway open on the map",
            "thin chair and table legs are only a few pixels wide at range --"
            " the hardest thing in the scene to see",
        ],
    },
    {
        "name": "Moving obstacle -- KNOWN FAILURE, shown deliberately",
        "scene": "moving_obstacle.xml",
        "args": ["--detector", "geometric", "--goal", "8,0",
                 "--animate-movers", "0.8", "--duration", "60"],
        "watch": [
            "the orange block shuttles across the route",
            "the robot drives straight into it and falls over",
            "why: a moving object never persists in the occupancy grid --"
            " each frame marks one cell, the next ray clears it, so the"
            " planner sees empty floor. Parked in the same spot it IS avoided",
            "fixing this needs tracking + prediction, which the stack has not"
            " got yet",
        ],
    },
]


def banner(i, s):
    line = "=" * 74
    print(f"\n{line}\n  [{i}/{len(SCENARIOS)}]  {s['name']}\n"
          f"  scene: {s['scene']}\n{line}")
    for w in s["watch"]:
        print(f"   * {w}")
    print("   (close the viewer window to continue)\n", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", type=int, help="run a single scenario, 1-based")
    p.add_argument("--list", action="store_true", help="list and exit")
    p.add_argument("--headless", action="store_true",
                   help="no viewer; just run them and print the outcomes")
    a = p.parse_args()

    if a.list:
        for i, s in enumerate(SCENARIOS, 1):
            print(f"  {i}. {s['name']}  [{s['scene']}]")
        return

    chosen = ([SCENARIOS[a.only - 1]] if a.only else SCENARIOS)
    start = a.only or 1
    for i, s in enumerate(chosen, start):
        banner(i, s)
        cmd = [sys.executable, "-u", str(HERE / "run_navigation.py"),
               "--scene", str(HERE / s["scene"]), *s["args"]]
        if not a.headless:
            cmd.append("--viewer")
        cmd += ["--out", str(HERE / f"tour_{i}.png")]
        subprocess.run(cmd, cwd=str(HERE))

    print("\ntour complete.")


if __name__ == "__main__":
    main()
