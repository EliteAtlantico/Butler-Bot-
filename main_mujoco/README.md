# BracketBot — MuJoCo simulation

A simulatable BracketBot built from the `chopped_urdf_v2` URDF: floating base,
driven wheels, collision geometry, seven cameras (RGB + depth), an LQR balance
controller ported from the real robot, and a place to plug in movement
algorithms.

```
./run.py --algorithm avoid          # drives itself around using the depth camera
./run.py --algorithm pick           # goes round an obstacle and picks a cube off a table
```

## What was wrong with the URDF

The shipped model is a statue. Worth knowing, because it explains the rebuild:

| | |
|---|---|
| Base structure | one chain of rigidly-welded cover parts — no floating base |
| Wheels | welded to the head, no hinges |
| Collision geometry | none at all (every geom `contype=0 conaffinity=0`) |
| Mass | 0.29 kg total across 53 links (Onshape exported volumes, not masses) |
| Actuators / sensors / cameras | none |

`build_dynamic_model.py` fixes all of that and writes `chopped_dynamic.xml`.
Re-run it if you change the source URDF:

```
.venv/bin/python build_dynamic_model.py
```

## Files

```
chopped_urdf_v2.xml     original visual-only MJCF (converted from the URDF)
scene.xml               original, floor + lighting          <- unchanged, for reference
build_dynamic_model.py  the rebuild
chopped_dynamic.xml     generated: the simulatable robot
scene_dynamic.xml       robot + floor + props to drive around
scene_flat.xml          robot + bare floor (for controller work)
scene_table.xml         obstacle + 0.30 m block table + 45 mm cube to pick
run.py                  entry point
bracketbot_sim/
  robot.py              BracketBot API, BalanceController, ODriveSim
  lqr.py                BracketBot's LQR, parameterised
  plant.py              measures the plant parameters out of the model
  algorithms.py         movement algorithms, LocalMap, NavigateTo
  kinematics.py         damped-least-squares arm IK
  manipulation.py       ArmController, PickCube
```

## The robot

| | |
|---|---|
| Total mass | 10.18 kg (3.38 chassis + 2×2.2 wheels + 2×1.2 arms) |
| Wheel radius / track | 0.0846 m / 0.322 m |
| CoM height above axle | 0.282 m |
| DOF | freejoint + 2 wheels + 18 arm joints |
| Actuators | 2 wheel torque motors, 18 arm position servos |
| Sensors | IMU (quat/gyro/accel), wheel encoders, base pos/vel |
| Cameras | `head_rgb`, `head_depth`, `head_stereo_left/right`, `wrist_cam_left/right`, `chase` |
| Gripper | 13 mm closed to 195 mm open, friction pads at the fingertips |

Wheel radius and the mass budget come from the real robot's `lib/lqr.py`
(`R=0.0846`, `Mr=2.2`, `Mp=4-0.62`). Everything else — inertias, CoM, track —
is measured out of the model by `plant.py`, so the controller is derived from
the plant it actually drives rather than from v1 hardware constants.

## Balancing

BracketBot is an inverted pendulum. `bracketbot_sim/lqr.py` is their LQR with
the plant parameters made arguments and `control.lqr` swapped for
`scipy.solve_continuous_are`. State is `[x, ẋ, pitch, pitch_rate, yaw,
yaw_rate]`, input is `[pitch_torque, yaw_torque]`, and the wheels are torque
actuators so the balance loop and the velocity loop can share them.

Measured tracking, from a standstill (`scene_flat.xml`):

| Command | Achieved | Pitch | Roll |
|---|---|---|---|
| 0.5 m/s forward | 0.47 m/s | −1.9° | 0.0° |
| 0.8 m/s forward | 0.78 m/s | −1.7° | 0.0° |
| −0.4 m/s reverse | −0.42 m/s | −0.7° | 0.0° |
| 1.0 rad/s spin | 1.00 rad/s | −1.5° | 0.0° |
| 0.4 m/s + 0.6 rad/s arc | 0.32 m/s, 0.60 rad/s | −1.2° | 0.0° |

Three things that are easy to get wrong here, all of which cost a fall:

* **Pitch must be measured in the body frame**, from the gravity vector
  (`bot.gravity_body`), not from the world-frame up-vector. The world-frame
  version is correct only while the robot faces +x and quietly degrades as it
  turns, so the robot balances fine until you ask it to spin and then tips over.
* **The velocity setpoint is a ramp and needs a governor.** If the robot can't
  keep up — saturated, wheel slipping, nose against a pillar — the reference
  runs away, the error integrates, the actuators sit on their limits, and it
  falls. `max_lead` / `max_yaw_lead` cap how far the reference may lead reality.
* **Yaw gets a smaller torque budget than pitch** (`max_yaw_torque`). Both
  commands share two motors; a yaw term big enough to spin briskly will saturate
  a wheel and take the balance loop's authority with it.

Upright is not pitch = 0 — the CoM sits slightly forward of the axle.
`plant.py` computes the geometric trim, and the controller leaks residual
position error into it (integral action) so the robot holds station instead of
parking a fixed distance away.

## Cameras and depth

```python
from bracketbot_sim.robot import BracketBot
bot = BracketBot()

rgb   = bot.camera("head_rgb", 640, 480)       # uint8 [H, W, 3]
depth = bot.depth("head_depth", 640, 480)      # float32 [H, W], METRES, inf = no return
left, right = bot.stereo()                     # 60 mm baseline pair
pts   = bot.point_cloud("head_depth")          # Nx3 in the camera frame
every = bot.all_cameras()                      # {name: rgb} for all seven
```

`bot.depth()` returns real metres, not the raw buffer. The head cameras sit at
z = 1.54 m; the depth camera is tilted **22° down** because a level camera at
that height looks straight over anything shorter than a coffee table. RGB and
stereo stay level.

The image-left half of the depth frame is the robot's **left** (verified against
an obstacle at a known bearing — the camera's x axis points right, which makes
it tempting to assume the opposite).

Rendering needs a GL backend: `MUJOCO_GL=egl` headless, `glfw` with a viewer.
`run.py` sets it for you. Call `bot.close()` when done.

## Movement algorithms

An algorithm is any callable `f(bot, t)` that calls `bot.drive(v, w)`. It runs
once per physics step, on top of the balance loop — you ask for a velocity, the
LQR works out how to stay upright while delivering it.

```python
def wander(bot, t):
    d_left, d_right = ...                      # your perception
    bot.drive(0.3, 0.6 if d_left > d_right else -0.6)

bot.step(30.0, controller=wander)
```

Included in `algorithms.py`: `Stand`, `Drive(v, w, duration)`,
`WaypointFollower`, `square_patrol`, `ObstacleAvoider` (depth-camera reactive
navigation), and `Sequence` to chain them.

Verified: the waypoint follower closes a 1.5 m square in 22 s; the obstacle
avoider drives autonomously for 39 s through `scene_dynamic.xml` without
falling and without coming closer than 0.82 m to any prop.

## Reaching and picking

`scene_table.xml` is the manipulation scene: an obstacle squarely on the route,
a 0.30 m block acting as a table, and a 45 mm cube sitting on top of it.

```
./run.py --algorithm pick               # navigate round the obstacle, pick the cube
```

`PickCube` runs as an ordinary `f(bot, t)` algorithm and sequences:
`navigate` (obstacle-aware) -> `align` -> `pregrasp` -> `descend` -> `grasp` ->
`lift`. The grasp is a real friction grasp between two fingertip pads -- there
is no weld constraint holding the cube on.

### Measured results

Eight cube placements across the reachable band, each run end to end from the
start pose (navigate round the obstacle, then pick):

| Cube placement | Result | Lift | Time |
|---|---|---|---|
| centre (the demo) | PASS | +0.212 m | 43.3 s |
| +y 0.10 m | PASS | +0.206 m | 29.1 s |
| -y 0.10 m | PASS | +0.205 m | 28.8 s |
| -y 0.06 m | PASS | +0.206 m | 29.1 s |
| +y 0.06 m | PASS | +0.215 m | 31.5 s |
| near edge | PASS | +0.203 m | 38.5 s |
| +y near edge | PASS | +0.214 m | 31.0 s |
| further in (stand-off clamped to reach) | PASS | +0.212 m | 54.8 s |

**8/8**, no contact with the obstacle, the table, or the robot's own body in
any run. Obstacle clearance was 0.43-0.47 m throughout (the pillar is 0.18 m in
radius).

### The arm

`kinematics.py` solves damped-least-squares IK on a grasp site built midway
between the fingertips, using the seven joints of one arm (the prismatic mast
plus six revolute) against a 6-DOF target. The site frames are built
world-axis-aligned at the home pose, so target orientation `identity` means
"gripper pointing straight down, fingers straddling along world y", and
`rot_z(theta)` spins the grasp about the vertical.

Measured reach envelope, gripper vertical, from a grasp site that rests 0.71 m
up (`pos_err < 4 mm`):

| Forward reach | Result |
|---|---|
| up to 0.40 m | solid at any height from -0.10 m to -0.45 m |
| 0.44 m | marginal (~8 mm error) |
| 0.48 m and beyond | unreachable |

Lateral tolerance is about +/-0.12 m. **This envelope is why the table is a
0.30 m block and not a 0.45 m one.** With a deeper table, the dead space
(table half-depth + base half-width) exceeds what the arm can reach past, and
the robot cannot both keep its nose out of the table and get a gripper over the
cube. `PickCube` derives its stand-off from the table's own geometry and warns
when a cube placement needs more reach than the arm has, instead of driving
into the table.

### Picking from a balancing base is the hard part

The base does not hold still. Extending a 1.2 kg arm 0.35 m forward moves the
CoM several centimetres, the balance loop answers by driving the wheels, and
the base makes a ~0.17 m excursion. Four things make it work anyway:

* **IK is re-solved every control tick against the world-frame target**, not
  solved once and replayed. The arm tracks the cube while the base moves under
  it.
* **The balance loop is told about the arm.** `bot.com_lean` measures the
  sprung CoM's offset from the axle in the base frame and feeds it forward as a
  pitch trim. Leaving the controller to discover the shift through accumulated
  position error lets the base creep 0.15-0.2 m first. This alone improved
  station-holding from 0.12 m of steady-state offset to 0.002 m.
* **Gain scheduling.** A second, position-stiff LQR gain set (`Q_STATION`) is
  swapped in while the arm works. The driving gains weight position lightly on
  purpose -- a balancing robot that fights every centimetre drives badly.
* **The descent waits for the base to settle.** Starting down during the
  excursion means the gripper arrives late and knocks the cube off the table.

### Navigation

`NavigateTo` picks a heading by testing corridors, not by balancing gradients.
For each candidate heading it sweeps the robot's width forward through a local
obstacle map and takes the heading closest to the goal that is clear for the
full lookahead. Things that had to be got right, each of which was a failure
first:

* **A local map, not just the current frame.** The head camera is 1.54 m up and
  tilted 22 deg down, so an obstacle shorter than the robot leaves the frame
  entirely as it gets close -- at 0.5 m a 0.9 m pillar is invisible. Steering on
  the current frame drives confidently into something it saw clearly two seconds
  ago.
* **Clear the map by ray-casting, not by ageing.** A short timeout makes the
  robot forget the obstacle it is pressed against, because it cannot re-observe
  it. Cells are dropped only when the camera looks through where they should be
  and sees further.
* **Self-filtering.** A camera on a mast sees the robot's own arms. Without a
  self-filter the map is permanently full of obstacles 0.15 m ahead.
* **Clearance over a corridor, not a cone.** An obstacle 0.3 m away and 33 deg
  off the nose is in the robot's path but outside any cone narrow enough to be
  useful at 2 m.
* **A pose controller for the last half metre.** Heading-to-goal swings through
  180 deg on the smallest overshoot; the polar (rho, alpha, beta) controller
  converges position and orientation together.
* **Stuck detection on actual motion.** The velocity *command*, and the balance
  loop's odometry (which reads non-zero from slipping wheels), both look like
  motion while the robot pushes uselessly into a pillar.

## Running the real robot's code

`bot.odrive` implements the same calls as quickstart's `lib/odrive_uart.ODriveUART`
(`set_speed_mps_left/right`, `get_position_turns_left/right`, `start_*`,
`enable_velocity_mode_*`, `clear_errors_*`), closing a PI velocity loop onto
torque the way the real ODrive closes one onto current. Call
`bot.odrive.update(bot.dt)` each step and leave the balance controller off to
drive it exactly as the hardware examples do.

## Not done

* No ToF (VL53L5CX) stand-in — `bot.point_cloud()` covers the same ground.
* Terrain is a flat plane with four props; no slopes, steps, or walls yet.
* The arms are held by position servos and are not used by any algorithm.
* Arm masses (1.2 kg each) are an estimate — the real LQR constants predate the
  v2 arms, so there is no published figure to match.
