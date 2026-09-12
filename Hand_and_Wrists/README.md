# Hand & Wrists

Arm, wrist and gripper control for the BracketBot. It finds a household item
with its head camera, works out how to hold it, drives to where it can reach
it, picks it up, and checks that it is actually holding it.

Everything here builds on the robot in `../main_mujoco` (model, balance
controller, arm IK) and the camera plumbing in `../comp_vision_sim`, without
editing either.

```powershell
cd Hand_and_Wrists
python run_pick.py --object remote --vision          # find it with the camera, pick it (viewer)
python run_pick.py --object keys --vision --random 3 # random placement (seed 3)
python run_pick.py --object mug --headless           # no window; position given by the sim
python eval_pick.py --vision                         # pick benchmark, camera only
python eval_vision.py                                # how accurate is the camera estimate?
```

Objects: `mug`, `can`, `remote` (coffee table), `bottle` (side table),
`keys`, `ball`, `box` (floor). In the viewer, Space pauses and R restarts.

## Results

84 trials, 12 per item. Each trial puts the item at a random spot and random
rotation on its surface and parks the robot 1.3–1.9 m away, roughly facing
it. A trial passes only if the robot says it is holding the item **and** the
simulator agrees: lifted at least 5 cm, still between the fingers, robot
upright.

| Item | Grasp | Found with the camera | Told where it is |
|---|---|---|---|
| mug | from above, fingers clear of the handle | 12/12 | 12/12 |
| can | from above | 12/12 | 12/12 |
| bottle | from the side | 12/12 | 12/12 |
| remote | from above, wrist turned to its long axis | 12/12 | 12/12 |
| keys | off the floor, wrist aligned | 11/12 | 12/12 |
| ball | off the floor | 12/12 | 12/12 |
| box | off the floor, wrist aligned | 12/12 | 11/12 |
| **all** | | **83/84** | **83/84** |

Per-trial data: `results/pick_eval_vision.*` (camera) and `results/pick_eval.*`
(told where it is).

### How good is the camera estimate?

`eval_vision.py`, 20 random views per item, compared with ground truth:

| Item | Centre error (mean / max) | Width error | Axis error (mean / max) |
|---|---|---|---|
| mug | 3.2 / 4.5 mm | −1.7 mm | 2.3° / 5.1° (handle seen 17/20) |
| can | 3.0 / 4.2 mm | −1.6 mm | – |
| bottle | 3.2 / 5.3 mm | +0.4 mm | – |
| remote | 2.7 / 5.2 mm | +3.2 mm | 0.4° / 0.7° |
| keys | 2.4 / 3.2 mm | +3.5 mm | 1.8° / 3.7° |
| ball | 9.0 / 12.1 mm | 0.0 mm | – |
| box | 2.3 / 2.9 mm | +3.6 mm | 0.6° / 1.6° |

For scale, the fingers open ~45 mm wider than the item, so the centre can be
~20 mm off before a fingertip lands on it.

## How it works

```
 look ─────> estimate ─────> planner ─────> Pick skill ───────────────────────────────>
(head RGB-D,  (where, how big,  (which arm,    approach -> settle -> pregrasp -> insert
 turn to       which way it      wrist angle,    -> close -> lift -> stow -> verify
 search)       points)           where to park)  (on failure: back up, look again, retry)
```

| File | What it does |
|---|---|
| `handwrist/vision.py` | Finds an item in the head camera: colour mask, biggest 3-D cluster, then geometry from the points. |
| `handwrist/objects.py` | Per-item grasp knowledge (from above or the side, align the wrist or not, keep clear of the handle) and `ObjectEstimate`, the planner's input. |
| `handwrist/gripper.py` | Finger-gap calibration measured from the model, and touch sensing: "holding" means both pads touch the same object *and* the fingers stalled short of the close command. |
| `handwrist/grasping.py` | Enumerates every way to grasp an item (each table edge or floor heading, each arm, each equivalent wrist angle), checks each with the arm IK from the base pose it implies, and ranks them. |
| `handwrist/skills.py` | `Pick`, the state machine, and `ApproachPose`, the last-metre parking controller. |
| `handwrist/scenarios.py` | Random placements for tests and demos, rejecting starts in or behind furniture. |
| `handwrist/colours.json` | Colour windows per item, written by `tools/calibrate_colours.py`. |
| `scenes/scene_home.xml` | Living room: coffee table, side table, basket, seven items. Reuses `main_mujoco/chopped_dynamic.xml`. |

### Seeing

`vision_sim.perception.observe()` (the vision team's code) renders the head
depth camera's colour and depth images from one pose and lifts every pixel
into world coordinates. On top of that:

1. **Colour mask.** Hue, saturation and brightness windows per item. They
   are calibrated by `tools/calibrate_colours.py` from segmentation renders,
   which label every pixel with the object it belongs to, giving 98–99 %
   recall. The renders are only used offline; at run time the detector sees
   colour and depth, nothing else. The wooden floor shares the keys' orange
   hue but is far less saturated, and that is what separates them.
2. **Biggest 3-D cluster**, so stray edge pixels do not drag the estimate.
3. **Geometry from the top face.** From a camera 1.5 m up, the whole top of
   an upright item is visible, so its outline gives the centre, width, length
   and long axis. The side facing the camera would only give half the item and
   bias every centre toward the robot. Round items also use the silhouette
   width per height slice, because a bottle's "top face" is its 32 mm neck on
   a 70 mm body. The mug's handle is whatever sticks out past the round body.

`vision_sim.detect()` is not used: it is tuned for navigation-scale obstacles,
drops anything under 12 cm tall, and clusters on a 30 cm grid. That removes
every floor item and merges the three on the coffee table.

**The head camera has a blind zone.** It is tilted 22° down from 1.54 m, so it
cannot see the floor nearer than ~1.25 m or a coffee-table top nearer than
~0.9 m. The robot therefore looks from a distance, turns on the spot in 40°
steps if the item is not in view, and grasps on that estimate. After a failed
grasp it reverses out of the blind zone and looks again, because the item has
usually been nudged.

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
* **Lift with the mast, not the IK.** Asked for "hand 12 cm higher" off the
  floor, the IK kept answering "mast down, shoulder up": it prices a metre of
  mast like a radian of shoulder. The hand pressed the item into the floor,
  propped the robot off its wheels, and it fell. The robot's own onboard IK
  (`constants.py`) penalises the mast 50× more than the wrist for the same
  reason.
* **Leave room for the lean.** Reaching to the floor shifts the CoM and the
  base creeps ~7 cm toward the item. Parking at 0.31 m instead of 0.28 m
  took floor picks from 32/36 to 35/36. Re-anchoring the balance loop at
  contact instead made it far worse (3/36).
* **Read the surface height from what is under the item, not from the ray.**
  Stepping a ray through the remote restarted it inside the tabletop, which
  reported its underside, 3 cm too low. That drove the fingertips into the
  table: 0/12 on the remote with vision, 12/12 after the fix.
* **Check reach from where the base actually stopped**, and **never drive
  with the arm out**.

## Limitations

* The robot's position comes from the simulator, as it does for the rest of
  the team's stack; there is no localisation.
* `ApproachPose` has no obstacle avoidance. It is meant for the last metre;
  longer routes should use `main_mujoco`'s `NavigateTo` first.
* Colour windows are per item, so two items of the same colour would
  confuse the detector. The team's YOLO detector returns the same kind of
  detection and could replace the colour step.
* **The wrist cameras are unusable in the current model.**
  `build_dynamic_model.py` places each one at its body's origin, which the
  CAD export left ~0.8 m from the hand, so it films the arm and table edge.
  Fixing it in that script would give the grasp a close-up view.
* The onboard IK library (`libhybrid_ik_lib.so`, RelaxedIK in Rust) is built
  for the robot's ARM64 computer and its Python binding refuses to load
  off Linux, so the simulation uses `main_mujoco`'s damped-least-squares IK.
