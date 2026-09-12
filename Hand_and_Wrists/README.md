# Hand & Wrists

Arm, wrist and gripper control for the BracketBot: it works out how to hold
each household item, drives to where it can reach it, picks it up and checks
that it is actually holding it.

Everything here builds on the robot in `../main_mujoco` (model, balance
controller, arm IK) without editing any of it.

```powershell
cd Hand_and_Wrists
python run_pick.py --object remote              # watch it in the viewer
python run_pick.py --object keys --random 3     # random placement (seed 3)
python run_pick.py --object mug --headless      # no window, just the log
python eval_pick.py                             # success-rate benchmark
```

Objects: `mug`, `can`, `remote` (coffee table), `bottle` (side table),
`keys`, `ball`, `box` (floor). In the viewer, Space pauses and R restarts.

## Results

`eval_pick.py -n 12`: 84 trials, each with the object at a random spot and
random rotation on its surface and the robot parked ~1 m away. A trial passes
only if the robot says it is holding the object **and** the simulator agrees
(object lifted >= 5 cm, still between the fingers, robot upright).

| Object | Grasp | Success | Mean time |
|---|---|---|---|
| mug | from above, fingers clear of the handle | 12/12 | 30.3 s |
| can | from above | 12/12 | 27.5 s |
| bottle | from the side | 12/12 | 27.5 s |
| remote | from above, wrist turned to its long axis | 12/12 | 28.0 s |
| keys | off the floor, wrist aligned | 12/12 | 35.1 s |
| ball | off the floor | 10/12 | 32.9 s |
| box | off the floor, wrist aligned | 12/12 | 35.3 s |
| **all** | | **82/84** | |

Full per-trial data: `results/pick_eval.md` and `results/pick_eval.json`.

## How it works

```
estimate ──> planner ──> Pick skill ─────────────────────────────────────────>
(where is it,  (which arm,   approach -> settle -> pregrasp -> insert -> close
 how big, which wrist angle,    -> lift -> stow -> verify   (retry on failure)
 way it points) where to park)
```

| File | What it does |
|---|---|
| `handwrist/objects.py` | Per-item grasp knowledge (from above or the side, align the wrist or not, keep clear of the handle) and `ObjectEstimate`, the planner's input. |
| `handwrist/gripper.py` | Finger-gap calibration measured from the model, and touch sensing: "holding" means both pads touch the same object *and* the fingers stalled short of the close command. |
| `handwrist/grasping.py` | Enumerates every way to grasp an item (each table edge or floor heading, each arm, each equivalent wrist angle), checks each with the arm IK from the base pose it implies, and ranks them by driving, turning and reach. |
| `handwrist/skills.py` | `Pick`, the state machine, and `ApproachPose`, the last-metre parking controller. |
| `handwrist/scenarios.py` | Random placements for tests and demos. |
| `scenes/scene_home.xml` | Living room: coffee table, side table, basket, seven items. Reuses `main_mujoco/chopped_dynamic.xml`. |
| `run_pick.py` / `eval_pick.py` | Demo and benchmark. |

### Reach

Measured with the arm IK from a parked base (grasp-site position, 6 mm tolerance):

| Gripper | Forward reach | Height |
|---|---|---|
| pointing down | 0.15–0.40 m | floor to ~0.8 m |
| pointing forward (side grasp) | 0.15–0.50 m | floor to ~0.9 m |

The base hull sticks out 0.094 m ahead of the mast, so a top grasp can reach
about 0.2 m in from a table edge. The planner derives the parking spot from
the table's geometry and rejects anything out of reach.

## Things that had to be got right

Each of these was a failure first, found by the benchmark.

* **Parking a balancing robot.** It has to lean back before it can brake,
  so it overshoots: 0.24 m when told to stop from 0.2 m/s, 0.36 m from
  0.3 m/s (about 1.2 s × speed), and a speed-toward-goal loop on top of that
  rocks ±0.2 m indefinitely. `ApproachPose` cruises at a fixed speed, starts
  braking at 1.4 s × *measured* speed with the balance loop's reference
  pinned on the goal, and always creeps the last leg into furniture.
* **Turning on the spot has a dead band.** Tyre scrub stalls the yaw loop
  5–8° short. A yaw-rate command with a 0.15 rad/s floor, then pinning the
  reference inside 0.03 rad, lands within 2°.
* **Move the hand in a straight line toward things.** Handing the IK the
  final pose makes every joint slew at its own rate, and the hand sweeps a
  curve. That dipped 8 cm below target and clipped the mug, and the arm's
  stiff servos then shoved the whole robot back 40 cm.
* **...but lift directly away from them.** Stepping the target up off the
  floor let the IK drift against a joint limit, where "hand up" came out as
  "mast down". The hand pressed into the floor and propped the robot off its
  wheels.
* **Check reach from where the base actually stopped.** Parking is good to
  a few cm, so `Pick` re-runs the IK from the real pose before reaching, and
  re-parks if needed.
* **Never drive with the arm out.** It moves the CoM enough that the base
  wobbles and parks badly, so a retry brings the arm home first.

## Limitations

* **Object positions come from the simulator.** Recognising objects with
  the cameras is the next phase. It plugs in through `Pick(..., estimator=...)`,
  which returns the same `ObjectEstimate`.
* `ApproachPose` has no obstacle avoidance. It is meant for the last metre;
  longer routes should use `main_mujoco`'s `NavigateTo` first.
* The ball occasionally falls: it can roll, and a nudge from the hull
  during the reach or a lift at full stretch tips the robot.
* The onboard IK library (`libhybrid_ik_lib.so`, RelaxedIK in Rust) is
  built for the robot's ARM64 computer and cannot load on an x86 PC. The
  simulation uses `main_mujoco`'s damped-least-squares IK instead.
