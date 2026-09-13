# BracketBot agent

The local LLM runs the whole robot through **tool calls**, in any scene. You
say what you want; the model picks a tool, the robot does it in the
simulator, the model reads how it went and decides what to do next, until it
can answer you.

```
python -m robot_agent "put the remote in the basket, then hand me the can"
python -m robot_agent --scene apartment "move the ball on the floor onto a chair"
python -m robot_agent                 # type requests one after another
python -m robot_agent --voice         # speak them (Whisper, runs locally)
python -m robot_agent --viewer        # watch it in the MuJoCo viewer
python -m robot_agent --list-tools
```

Needs the llama-server running with tool support (`--jinja`), by default
`http://localhost:8080/v1`, and for detection `ultralytics` (YOLO-World; the
weights are in `weights/`). Voice needs `comp_vision_sim/requirements-voice.txt`.
The conversation carries over between requests.

Scenes: `living_room` (`Hand_and_Wrists/scenes/scene_home.xml`, default),
`apartment` (`comp_vision_sim/home_search.xml`), or the path of any scene that
includes the robot.

## No fixed chores

The chores in `Hand_and_Wrists` (fetch, put, tidy) chain the same steps every
time for six known items and four known places. Here the model gets those
steps as tools and decides how each is done, so it can do things nobody wrote a
chore for, with objects and furniture it has never seen:

```
find       look_around, detect_objects, search_for,     every candidate in the room, with
           describe_view                                size and what it rests on
position   go_near, go_to_surface, go_to, move, turn    stand where the camera can see it
measure    inspect_object     centre, width, length, height, axis, what it rests on
decide     plan_grasp         top or side, which arm, align the wrist, grip height,
                              handle, squeeze -- checked with IK, nothing moves
act        pick_up            the same choices, run by handwrist's Pick
put down   list_surfaces      every surface in the scene, set-on or drop-into
           place_held_item    handwrist's Place, on any of them, optionally at x, y
```

What makes that work in other scenes:

* **Objects by name, not by colour.** `handwrist.detection.DetectionEstimator`
  boxes the object with YOLO-World when it is confident, otherwise the vision
  LLM, then measures it from depth with the colour estimator's own geometry.
  Measured on the living-room items from benchmark viewpoints (then seven, since
  cut to five): centre error median 3 mm (worst 9 mm), width within 4 mm. For
  something the detector has no word for, `looks_like="small object on the
  floor"` tells it what to ask for.
* **Surfaces from the scene's furniture.** `handwrist.surfaces.find_surfaces`
  reads every uncovered, upward-facing top at arm height, and containers (a
  floor with walls). In the living room it reproduces the four hand-written
  places exactly; the apartment gets its table, chairs, sofa, coffee table,
  bin and sideboard with no configuration.
* **Grasps chosen by the model.** `ObjectSpec` -- top/side, aligned wrist,
  handle, grip height, squeeze -- used to be a table row per item. The model
  fills it in, `plan_grasp` tells it whether it works and why not ("0.30 m in
  from the edge: out of reach from every side").

Every tool returns `ok`, the reason when it failed (and for `pick_up`, the
phases it got through), and the robot's pose, and never raises. A fallen robot
refuses motion tools. Each hand holds one object, so two can be carried at once:
`pick_up` uses the free arm (leaving the holding arm, and its grip, alone), and
`place_held_item` takes `object` to say which to put down. Loose items are dropped
onto their surfaces when a scene loads (`handwrist.surfaces.settle_loose_items`),
so no grasp starts on an item resting exactly in a table top.

Guards learned from live runs:

* `go_to` refuses a spot the robot cannot stand in and names what is in the way
  and the nearest free spot ("too close to the cartons; nearest spot (0.41,
  2.45)"); `go_near` is the way to approach an object. A model once sent the
  same unreachable `go_to` thirty times.
* The agent loop refuses a third identical call that already failed twice and
  tells the model to change something.
* A straight short drive is used only when the line clears all furniture;
  otherwise the A* navigator plans it. Clipping a carton's corner knocked the
  robot over.
* Detections match words: "box" also matches a stack of cartons. `look_around`
  and `search_for` report size and support, and the prompt tells the model to
  reject candidates that contradict the request.

## Limits

* The sim advances only while a tool runs, so the robot waits balanced while
  the model thinks.
* Surfaces come from the scene model, as the grasp planner's support and
  obstacle checks already do; objects come from the camera.
* `pick_up`'s last-metre approach drives straight: get within about 2 m with
  `go_to` / `go_to_surface` first (it refuses from further away).
* The head camera cannot see the floor within ~1.2 m or a table top within
  ~0.9 m; the tools say so when an object is boxed but too close to measure.
* Placement tested in the apartment: a floor ball carried through the doorway
  onto the bin works. Small chairs do not: the seat's legs are closer together
  than the base is wide, so it cannot park, and on one run releasing onto a
  seat levered the robot off its wheels. Surfaces at or above the carry height
  (~0.72 m: the dining table) are refused with that reason: the stowed hand
  hits their edge, and carrying higher made the robot fall.
* The phone remote (`remote_control/`) is separate and unchanged.

## Tests

```
python -m pytest tests/test_robot_agent.py -m "not integration"   # loop, plumbing, schemas
python -m pytest tests/test_robot_agent.py -m integration         # the robot, both scenes
```
