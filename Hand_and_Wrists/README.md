# Hand & Wrists

Arm, wrist and gripper control for the BracketBot, and the household chores
built on it. The robot finds an item with its head camera, works out how to
hold it, drives to where it can reach it, picks it up, carries it, and puts
it down where it was asked: on a table, in the laundry basket, or in a
person's hand.

Everything here builds on the robot in `../main_mujoco` (model, balance
controller, arm IK) and the camera plumbing in `../comp_vision_sim`, without
editing either.

```powershell
cd Hand_and_Wrists
python run_task.py fetch --item remote            # find the remote, hand it to the person
python run_task.py put --item remote --to basket  # pick it up, put it in the basket
python run_task.py tidy                           # every item on the coffee table -> basket
python run_task.py pick --item mug                # just pick it up

python run_pick.py --object remote --vision       # the pick on its own, in the viewer
python eval_tasks.py                              # chore benchmark
python eval_pick.py --vision                      # pick benchmark
python eval_vision.py                             # how accurate is the camera estimate?
```

Add `--random SEED` to scatter the item (or the coffee table for `tidy`),
`--headless` for no window, `--truth` to be told where items are instead of
finding them. In the viewer, Space pauses and R restarts.

Items: `mug`, `can`, `remote` (coffee table), `bottle` (side table),
`ball` (floor). Places: `coffee_table`, `side_table`, `basket`,
`person`.

## Results

Each trial puts the item at a random spot and rotation on its surface and
parks the robot 1.3–1.9 m away. A trial passes only if the robot says it
succeeded **and** the simulator agrees afterwards.

**Chores** (`eval_tasks.py`, 2 trials per item and task, 2 tidy-ups):

| Chore | Found with the camera | Told where it is |
|---|---|---|
| put it in the basket (7 items) | 13/14 | 13/14 |
| hand it to the person (7 items) | 13/14 | 13/14 |
| tidy the coffee table (mug, can, remote) | 2/2 | 2/2 |
| **all** | **28/30** | **28/30** |

**Picking** (`eval_pick.py`, 12 per item):

| Item | Grasp | Camera | Told |
|---|---|---|---|
| mug | from above, fingers clear of the handle | 12/12 | 11/12 |
| can | from above | 12/12 | 12/12 |
| bottle | from the side | 12/12 | 12/12 |
| remote | from above, wrist turned to its long axis | 12/12 | 12/12 |
| ball | off the floor | 12/12 | 12/12 |
| **all** | | **83/84** | **83/84** |

Furniture bumps during picks: 0 of 84 when told, 1 of 84 with the camera,
down from several per item before the base's creep was planned for (see
below).

**Camera accuracy** (`eval_vision.py`, 20 random views per item): centre error
2–5 mm (the ball 8–9 mm), width within 4 mm, axis within 5°. The fingers open
~45 mm wider than the item, so ~20 mm of error is tolerated.

Per-trial data: `results/`.

## Chores, for other front ends

A chore is started by name with keyword arguments -- the same shape as a
parsed command -- so a CLI, a benchmark, the phone remote or a voice
interface can start one without knowing how it is done:

```python
from handwrist.tasks import make_task

task = make_task(bot, "fetch", item="remote")            # hand it to the person
task = make_task(bot, "put", item="remote", to="basket")
task = make_task(bot, "tidy", surface="coffee_table", into="basket")
task = make_task(bot, "pick", item="mug")

while not task.done:
    bot.step(0.1, controller=task)

task.status      # one line, e.g. "fetch the remote: put the remote to the person: lower"
task.succeeded   # True / False
task.failure     # why, in words, e.g. "pick up the remote: could not find the remote"
task.results     # every step and how it went
```

An unknown item, place or action raises `ValueError` with a readable message
("I don't know an item called 'phone'; I know ball, bottle, ..."), suitable
for reading back to whoever asked. `handwrist.tasks.ACTIONS` lists what is
available.

The LLM search on `feature/llm-scene-reasoning` stops the robot 1.0 m from
what it found. That is inside the head camera's blind zone for floor items;
`Pick` handles it by backing up until it can see the item again.

## Any object, any scene

The chores above know six items and four places by name, and find the items
by what they are: an open-vocabulary detector is asked for "a remote" or "a mug",
not for a colour. The same skills also take objects and places they have never
seen, which is how the LLM agent (`../robot_agent`) drives them:

```python
from handwrist.detection import DetectionEstimator
from handwrist.objects import ObjectSpec
from handwrist.skills import Pick
from handwrist.place import Place
from handwrist.surfaces import find_surfaces, obstacle_boxes

see = DetectionEstimator()                                   # YOLO-World, else the vision LLM
pick = Pick(bot, ObjectSpec("tv remote", "", aligned=True, shape="box"), estimator=see)
...
shelf = {s.name: s for s in find_surfaces(bot.model, bot.data)}["sideboard"]
place = Place(bot, pick, shelf.spec(obstacles=obstacle_boxes(bot.model, bot.data, "sideboard")))
```

| File | What it adds |
|---|---|
| `handwrist/detection.py` | `DetectionEstimator`, the default for `Pick` and the chores: an open-vocabulary box (YOLO-World when confident, the vision LLM otherwise) plus depth, measured with `CameraEstimator`'s geometry. Each catalogue item is asked for under the names a detector knows it by (`QUERIES`); a colour in the request ("the red mug") only ranks the candidates. |
| `handwrist/surfaces.py` | `find_surfaces`: every uncovered upward-facing top at arm height, and containers, read from the scene. Reproduces `PLACES` exactly in the living room. |

`Pick` accepts an `ObjectSpec` as well as a catalogue name, and `Place` /
`PlacePlanner` accept a `PlaceSpec` carrying its own surface, obstacles and an
optional spot, as well as a `PLACES` name. Existing calls are unchanged.

## How it works

```
find ──> estimate ──> plan ──> Pick ──────────────────────────> Place ────────────────────>
(head     (where, how   (arm,    approach, settle, reach, close,   back off, drive over, lower
 RGB-D)    big, which    wrist,   verify by touch, lift, stow       until touchdown, let go,
           way)          parking) (on failure: back up, look again) retreat
```

| File | What it does |
|---|---|
| `handwrist/vision.py` | The geometry both estimators share (biggest 3-D cluster, then size and axis from the points), and `CameraEstimator`, the original calibrated-colour finder, kept for comparison. |
| `handwrist/objects.py` | Per-item grasp knowledge (from above or the side, align the wrist, keep clear of the handle) and `ObjectEstimate`. |
| `handwrist/gripper.py` | Finger-gap calibration from the model, and touch sensing: holding = both pads on the same object and the fingers stalled short of the close command. |
| `handwrist/grasping.py` | Enumerates every way to grasp (table edges or floor headings × arm × wrist angle × sideways shift of the base), ranks them, and IK-checks the best. |
| `handwrist/skills.py` | `Pick`, the shared `ArmSkill` machinery, and `ApproachPose`, the last-metre parking controller. |
| `handwrist/places.py` / `place.py` | Named places read from the scene, where to park to put something down, and the `Place` skill. |
| `handwrist/tasks.py` | Chores: `make_task`, `fetch`, `put`, `tidy`, `pick`. |
| `handwrist/scenarios.py` | Random placements for tests and demos. |
| `scenes/scene_home.xml` | Living room: coffee table, side table, laundry basket, a person with a hand held out, six items. Each item and the person are dressed in a realistic textured model (`comp_vision_sim/assets/household.xml`, visual only); the collision primitives underneath are unchanged. |
| `tools/calibrate_colours.py` | Calibrates `CameraEstimator`'s colour windows (`handwrist/colours.json`) from segmentation renders. Calibrated on the old flat-colour items; rerun it before comparing against the realistic ones. |

### Seeing

`vision_sim.perception.observe()` (the vision team's code) renders the head
depth camera's colour and depth images from one pose and lifts every pixel
into world coordinates. On top of that:

1. **What it is.** YOLO-World boxes the item by name -- a glazed mug with a
   handle, a printed soda can, a TV remote with its keypad -- and the depth
   pixels in the box above the surface it stands on are the item. A colour
   named in the request re-ranks the boxes; it never finds one. (The older
   `CameraEstimator` masked calibrated per-item colour windows instead: a
   robot that only knew "the purple thing" could not pick up a real remote.)
2. **Biggest 3-D cluster**, so stray edge pixels do not drag the estimate.
3. **Geometry from the top face.** From 1.5 m up the whole top of an upright
   item is visible, and its outline gives centre, width, length and axis.
   Round items also use the silhouette width per height slice (a bottle's
   "top" is its neck); the mug's handle is what sticks out past the body.

The head camera is tilted 22° down from 1.54 m, so it cannot see the floor
nearer than ~1.25 m or a coffee-table top nearer than ~0.9 m. The robot
looks from a distance, turns on the spot in 40° steps if the item is not in
view, and after a failed grasp reverses out of the blind zone to look again.

### Reach

| Gripper | Forward reach | Height |
|---|---|---|
| pointing down | 0.15–0.40 m | floor to ~0.8 m |
| pointing forward (side grasp) | 0.15–0.50 m | floor to ~0.9 m |

The arm reaches high better further out, so the place planner stands further
back when the closest spot fails its IK check.

## Things that had to be got right

Each of these was a failure first, found by the benchmarks.

**Driving a balancing robot**
* **Parking.** It has to lean back before it can brake, so it overshoots
  (~1.2 s × speed). `ApproachPose` cruises at a fixed speed, brakes at
  1.4 s × measured speed with the balance reference pinned on the goal, and
  creeps the last leg.
* **Turning on the spot** stalls 5–8° short (tyre scrub). A yaw-rate floor
  lands within 2°; a turn that stalls within 9° is accepted.
* **It cannot back off something it is touching.** To reverse it first leans
  back -- by rolling forward -- and an obstacle against the hull blocks that:
  measured, the wheels sat still while the controller pressed the hull into a
  table leg harder and harder. Pressed against scenery close to its goal, it
  accepts where it is; otherwise it biases the lean the other way for 0.6 s
  so the wheels roll back. The bias lives on the balance controller with an
  expiry time -- an orphaned one once sent the robot 5 m across the room.
* **The base creeps 8–20 cm forward while the arm works**, under the tabletop
  and into a leg. The planner leaves a creep corridor clear of anything low,
  and can shift the base sideways so it sits clear of a corner leg while the
  arm reaches across. Furniture bumps during picks went to zero.
* **Back away before turning** after a pick, and never drive with the arm
  out. The last-metre route is checked against the known furniture.

**The hand**
* **Move the hand in a straight line toward things**, but **lift with the
  mast alone**: asked to raise the hand off the floor, the IK lowered the mast
  and swung the shoulder, pressing the item into the floor. (The robot's own
  onboard IK weights the mast 50× the wrist for the same reason.)
* **Grip creep is a simulator artifact, fixed in the scene.** MuJoCo's soft
  friction let a squeezed item slide out at ~0.5 mm/s, enough to drop flat
  items on a 30 s carry; `noslip_iterations="5"` in `scene_home.xml` removes it.
* **Put things down until they touch down**, not to a planned height:
  pushing on after touchdown propped the robot up on the item and it fell.
* **Lower into the basket** rather than dropping from above the rim (items
  bounced out), and use a different spot for each item so they sit side by
  side.
* **Read the support height from the geometry under the item**, not a ray
  hit: a ray stepped through the remote reported the tabletop's underside.

## Limitations

* An item right against a table's corner leg can leave no parking spot with
  both reach and creep room; the pick then stops with "parked out of reach"
  rather than wedging the base (the one miss in the "told" benchmarks).
* `ApproachPose` checks its straight-line route against known furniture but
  does not plan detours; long routes should use `main_mujoco`'s `NavigateTo`.
* Robot pose comes from the simulator, as for the rest of the team's stack.
* Finding items needs YOLO-World (`ultralytics`, `clip`) or a vision LLM; the
  colour estimator is still there without either.
* The keys and the box were taken out of the simulation: at pick range they
  were a few pixels across in the head camera, and YOLO-World boxed them
  confidently in under a third of views (the five remaining items: every view).
* The wrist cameras are unusable in the current model: `build_dynamic_model.py`
  puts each at its link's origin, ~0.8 m from the hand. A fix is prepared but
  not yet merged into `main`.
* The onboard IK library (`libhybrid_ik_lib.so`) is built for the robot's
  ARM64 computer and its Python binding refuses to load off Linux, so the
  simulation uses `main_mujoco`'s damped-least-squares IK.
