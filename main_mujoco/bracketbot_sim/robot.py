"""BracketBot simulation API.

Wraps the MuJoCo model in something shaped like the real robot's software:

    bot = BracketBot()
    bot.balance.enable()
    bot.drive(0.3, 0.0)            # m/s forward, rad/s yaw
    rgb   = bot.camera("head_rgb")
    depth = bot.depth("head_depth")     # metres, float32
    bot.step(0.05)

Sign conventions (verified against the model geometry):
  * +x is forward, +y is the robot's left, +z is up
  * the wheel hinges spin about +y and a POSITIVE joint velocity rolls the
    robot forward (verified by pulsing the actuators in the built model)
  * pitch > 0 means leaning forward, yaw follows the right-hand rule about +z
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# Pick an offscreen GL backend before mujoco is imported, unless the caller
# has already chosen one. egl works headless; glfw needs a display.
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from .lqr import LQR_gains, PlantParams  # noqa: E402
from .plant import measure_plant, sprung_bodies, sprung_com  # noqa: E402

DEFAULT_XML = Path(__file__).resolve().parent.parent / "scene_dynamic.xml"

# LQR weights: [x, x_dot, pitch, pitch_rate, yaw, yaw_rate] and [pitch_t, yaw_t]
Q_DIAG = (60.0, 30.0, 260.0, 20.0, 40.0, 10.0)
R_DIAG = (18.0, 1.0)
# A second, position-stiff gain set for holding still while the arm works.
# The driving gains weight position lightly on purpose -- a balancing robot
# that fights every centimetre drives badly -- but that softness lets the
# base overshoot ~0.25 m when the arm swings its CoM forward, which is the
# difference between reaching over a table and headbutting it.
Q_STATION = (700.0, 160.0, 260.0, 22.0, 40.0, 10.0)
R_STATION = (18.0, 1.0)


class BalanceController:
    """LQR balance + velocity/yaw tracking, the sim's stand-in for the real
    robot's balance loop.

    The LQR regulates the full state to zero, so driving is done by walking the
    position and yaw *references* forward at the commanded rates -- the
    controller then chases a setpoint that is always moving, which is what
    produces sustained motion instead of a one-shot lurch.
    """

    def __init__(self, plant: PlantParams, q_diag=Q_DIAG, r_diag=R_DIAG,
                 q_station=Q_STATION, r_station=R_STATION):
        self.plant = plant
        self.K_drive = LQR_gains(q_diag, r_diag, plant)
        self.K_station = LQR_gains(q_station, r_station, plant)
        self.K = self.K_drive
        self.enabled = False
        self.v_cmd = 0.0        # m/s
        self.w_cmd = 0.0        # rad/s
        self.x_ref = 0.0
        self.yaw_ref = 0.0
        self.pitch_trim = plant.trim  # upright != pitch 0; see plant.py
        self.trim_feedforward = True   # recompute the trim from the live CoM
        self.trim_integral = 0.0
        self.max_torque = 15.0
        # Yaw gets a smaller torque budget than pitch on purpose. The two
        # commands share the same two motors, and a yaw term big enough to spin
        # the robot briskly will happily saturate a wheel and take the pitch
        # loop's authority with it -- at which point it falls over. Balance
        # wins ties.
        self.max_yaw_torque = 4.0
        # Slow auto-trim: the geometric trim from plant.py is only a starting
        # guess (the arms sag under their position actuators, which moves the
        # CoM). Leaking the residual POSITION error into the trim is integral
        # action on x -- without it the robot balances happily but holds
        # station a fixed distance from where it was asked to stand.
        self.auto_trim = True
        self.trim_gain = 0.012
        self.trim_limit = 0.05
        self._base_trim = plant.trim
        # Reference governor. The setpoints are ramps, so if the robot cannot
        # keep up (saturated torque, a wheel slipping, driving into a pillar)
        # the reference runs away and the error integrates without bound until
        # the actuators sit on their limits and it falls over. Capping how far
        # the reference may lead the measured state is the anti-windup.
        self.max_lead = 0.30       # m
        self.max_yaw_lead = 0.35   # rad

    def enable(self, state=None):
        """Start balancing, seeding the references from the current state."""
        self.enabled = True
        if state is not None:
            self.x_ref = state[0]
            self.yaw_ref = state[4]

    def disable(self):
        self.enabled = False

    def station_gains(self, on=True):
        """Swap between driving gains and position-stiff station gains."""
        self.K = self.K_station if on else self.K_drive

    def reset_reference(self, state):
        self.x_ref = state[0]
        self.yaw_ref = state[4]

    def __call__(self, state, dt, bot=None):
        """state -> (pitch_torque, yaw_torque)."""
        if not self.enabled:
            return 0.0, 0.0
        if self.trim_feedforward and bot is not None:
            self._base_trim = -float(np.arctan2(bot.com_lean, self.plant.L))
        else:
            self._base_trim = self.plant.trim
        self.x_ref += self.v_cmd * dt
        self.yaw_ref += self.w_cmd * dt
        self.x_ref = float(np.clip(self.x_ref, state[0] - self.max_lead,
                                   state[0] + self.max_lead))
        yaw_err = _wrap(self.yaw_ref - state[4])
        if abs(yaw_err) > self.max_yaw_lead:
            self.yaw_ref = state[4] + np.sign(yaw_err) * self.max_yaw_lead

        err = np.array(state, float)
        err[0] -= self.x_ref
        err[1] -= self.v_cmd
        err[2] -= self.pitch_trim
        err[4] = _wrap(err[4] - self.yaw_ref)
        err[5] -= self.w_cmd

        if self.auto_trim:
            self.trim_integral = float(np.clip(
                self.trim_integral - self.trim_gain * err[0] * dt,
                -self.trim_limit, self.trim_limit))
        self.pitch_trim = self._base_trim + self.trim_integral

        u = -self.K @ err
        return (float(np.clip(u[0], -self.max_torque, self.max_torque)),
                float(np.clip(u[1], -self.max_yaw_torque, self.max_yaw_torque)))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BracketBot:
    def __init__(self, xml=DEFAULT_XML, q_diag=Q_DIAG, r_diag=R_DIAG,
                 arm_pose=None):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self._wheel_act = [self._act_id(f"{s}_wheel") for s in ("left", "right")]
        self._wheel_jnt = [self._jnt_id(f"{s}_wheel_joint") for s in ("left", "right")]
        self._chassis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                          "chassis")
        self.arm_joints = [n for n in self._joint_names()
                           if n not in ("left_wheel_joint", "right_wheel_joint")]

        self._sprung = sprung_bodies(self.model)
        self._wheel_body = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                              f"{s}_wheel") for s in ("left", "right")]
        self._renderers: dict[tuple, mujoco.Renderer] = {}
        self._depth_renderers: dict[tuple, mujoco.Renderer] = {}

        self.reset(arm_pose)
        self.plant = measure_plant(self.model, self.data)
        self.balance = BalanceController(self.plant, q_diag, r_diag)
        self.odrive = ODriveSim(self)

    # ------------------------------------------------------------------ ids
    def _act_id(self, n):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def _jnt_id(self, n):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)

    def _joint_names(self):
        out = []
        for j in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n and self.model.jnt_type[j] in (mujoco.mjtJoint.mjJNT_HINGE,
                                                mujoco.mjtJoint.mjJNT_SLIDE):
                out.append(n)
        return out

    # ---------------------------------------------------------------- state
    def reset(self, arm_pose=None):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.001          # rest the tyres on the floor
        if arm_pose:
            for name, val in arm_pose.items():
                self.set_arm_target(name, val)
                adr = self.model.jnt_qposadr[self._jnt_id(name)]
                self.data.qpos[adr] = val
        mujoco.mj_forward(self.model, self.data)
        self._odom_ref = self.wheel_angles.copy()
        if hasattr(self, "balance"):
            self.balance.reset_reference(self.state)

    @property
    def time(self):
        return float(self.data.time)

    @property
    def rotation(self):
        return self.data.xmat[self._chassis].reshape(3, 3)

    @property
    def gravity_body(self):
        """World up-vector expressed in the body frame -- what an IMU sees."""
        return self.rotation.T @ np.array([0.0, 0.0, 1.0])

    @property
    def pitch(self):
        """Lean angle about the body y axis, + is forward.

        Measured from the gravity vector in the BODY frame, which makes it
        yaw-invariant. Taking it from the world-frame up-vector instead reads
        correctly only while the robot faces +x and silently decays to nonsense
        as it turns -- which shows up as the robot falling over during spins.
        """
        u = self.gravity_body
        return float(np.arctan2(-u[0], u[2]))

    @property
    def roll(self):
        """Lean angle about the body x axis, + is toward the robot's left."""
        u = self.gravity_body
        return float(np.arctan2(u[1], u[2]))

    @property
    def yaw(self):
        fwd = self.rotation @ np.array([1.0, 0.0, 0.0])
        return float(np.arctan2(fwd[1], fwd[0]))

    @property
    def angular_velocity(self):
        """Body-frame angular velocity from the gyro (rad/s)."""
        return self.data.sensor("imu_gyro").data.copy()

    @property
    def wheel_angles(self):
        return np.array([self.data.qpos[self.model.jnt_qposadr[j]]
                         for j in self._wheel_jnt])

    @property
    def wheel_rates(self):
        return np.array([self.data.qvel[self.model.jnt_dofadr[j]]
                         for j in self._wheel_jnt])

    @property
    def wheel_speeds_mps(self):
        """Ground speed of each wheel [left, right], + is forward."""
        return self.plant.R * self.wheel_rates

    @property
    def odometry(self):
        """Forward distance travelled since reset, from the encoders (m)."""
        return float(self.plant.R * np.mean(self.wheel_angles - self._odom_ref))

    @property
    def forward_velocity(self):
        return float(np.mean(self.wheel_speeds_mps))

    @property
    def position(self):
        """Ground-truth chassis position (m). Not available on hardware."""
        return self.data.xpos[self._chassis].copy()

    @property
    def state(self):
        """LQR state [x, x_dot, pitch, pitch_rate, yaw, yaw_rate]."""
        w = self.angular_velocity
        return np.array([self.odometry, self.forward_velocity,
                         self.pitch, float(w[1]),
                         self.yaw, float((self.rotation @ w)[2])],
                        dtype=float)

    @property
    def com_lean(self):
        """Forward offset of the sprung CoM from the wheel axle, in the base
        frame (metres). Positive means the mass is ahead of the wheels.

        This is how the balance loop finds out that the arm moved. Reaching
        forward with a 1.2 kg arm shifts the CoM by several centimetres, and a
        controller that only discovers this through accumulated position error
        lets the base creep 0.15-0.2 m before it catches up -- straight into
        the table it was reaching over.
        """
        com = sprung_com(self.model, self.data, self._sprung)
        axle = 0.5 * (self.data.xpos[self._wheel_body[0]]
                      + self.data.xpos[self._wheel_body[1]])
        return float((self.rotation.T @ (com - axle))[0])

    @property
    def ground_speed(self):
        """True horizontal speed of the base (m/s). Not available on hardware;
        the wheels' odometry reads non-zero when they slip."""
        return float(np.linalg.norm(self.data.qvel[0:2]))

    @property
    def fallen(self):
        return abs(self.pitch) > 0.6

    # ------------------------------------------------------------- commands
    def drive(self, v_mps=0.0, w_rads=0.0):
        """Command forward speed and yaw rate through the balance controller."""
        self.balance.v_cmd = float(v_mps)
        self.balance.w_cmd = float(w_rads)

    def set_wheel_torque(self, left_nm, right_nm):
        """Raw torque, + is forward drive on that wheel."""
        lim = self.balance.max_torque
        self.data.ctrl[self._wheel_act[0]] = np.clip(left_nm, -lim, lim)
        self.data.ctrl[self._wheel_act[1]] = np.clip(right_nm, -lim, lim)

    def set_arm_target(self, joint, value):
        self.data.ctrl[self._act_id(f"act_{joint}")] = float(value)

    def joint_position(self, name):
        return float(self.data.qpos[self.model.jnt_qposadr[self._jnt_id(name)]])

    def joint_velocity(self, name):
        return float(self.data.qvel[self.model.jnt_dofadr[self._jnt_id(name)]])

    def body_position(self, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"no body named {name!r}")
        return self.data.xpos[bid].copy()

    def site_position(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
        return self.data.site_xpos[sid].copy()

    def set_arm_pose(self, pose: dict):
        for k, v in pose.items():
            self.set_arm_target(k, v)

    # ----------------------------------------------------------------- step
    def step(self, duration=None, controller=None):
        """Advance the sim. `duration` in seconds (default one timestep).

        `controller(bot, t)` is called once per physics step, before the
        balance loop -- that is where a movement algorithm lives.
        """
        n = 1 if duration is None else max(1, int(round(duration / self.dt)))
        for _ in range(n):
            if controller is not None:
                controller(self, self.time)
            tau_pitch, tau_yaw = self.balance(self.state, self.dt, self)
            # pitch torque drives both wheels together, yaw torque differentially
            self.set_wheel_torque(tau_pitch / 2 - tau_yaw / 2,
                                  tau_pitch / 2 + tau_yaw / 2)
            mujoco.mj_step(self.model, self.data)
        return self.state

    # -------------------------------------------------------------- cameras
    @property
    def camera_names(self):
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                for i in range(self.model.ncam)]

    def _renderer(self, cache, height, width, depth):
        key = (height, width)
        if key not in cache:
            r = mujoco.Renderer(self.model, height=height, width=width)
            if depth:
                r.enable_depth_rendering()
            cache[key] = r
        return cache[key]

    def camera(self, name="head_rgb", width=640, height=480):
        """RGB image as uint8 [H, W, 3]."""
        r = self._renderer(self._renderers, height, width, depth=False)
        r.update_scene(self.data, camera=name)
        return r.render()

    def depth(self, name="head_depth", width=640, height=480, max_range=10.0):
        """Depth image in METRES as float32 [H, W]; beyond max_range -> inf."""
        r = self._renderer(self._depth_renderers, height, width, depth=True)
        r.update_scene(self.data, camera=name)
        d = r.render().astype(np.float32)
        d[d > max_range] = np.inf
        return d

    def stereo(self, width=640, height=480):
        return (self.camera("head_stereo_left", width, height),
                self.camera("head_stereo_right", width, height))

    def all_cameras(self, width=320, height=240):
        """Every camera at once: {name: rgb}, plus depth for head_depth."""
        return {n: self.camera(n, width, height) for n in self.camera_names}

    def camera_pose(self, name):
        """(position, rotation matrix) of a camera in world coordinates."""
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        return self.data.cam_xpos[cid].copy(), self.data.cam_xmat[cid].reshape(3, 3).copy()

    def point_cloud_world(self, name="head_depth", width=160, height=120,
                          max_range=6.0):
        """Depth image as Nx3 points in WORLD coordinates."""
        pts = self.point_cloud(name, width, height, max_range)
        p, R = self.camera_pose(name)
        return pts @ R.T + p

    def point_cloud(self, name="head_depth", width=160, height=120,
                    max_range=6.0):
        """Depth image -> Nx3 points in the camera frame (x right, y up, -z fwd)."""
        d = self.depth(name, width, height, max_range)
        cid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        fovy = np.deg2rad(self.model.cam_fovy[cid])
        f = 0.5 * height / np.tan(fovy / 2)
        v, u = np.mgrid[0:height, 0:width]
        keep = np.isfinite(d)
        d, u, v = d[keep], u[keep], v[keep]
        x = (u - width / 2) * d / f
        y = -(v - height / 2) * d / f
        return np.stack([x, y, -d], -1)

    def close(self):
        """Release the GL contexts. Safe to call twice; call it before exit --
        left to the garbage collector these tear down after EGL itself has
        gone and spray harmless-but-alarming EGLError tracebacks."""
        for cache in (self._renderers, self._depth_renderers):
            for r in cache.values():
                try:
                    r.close()
                except Exception:
                    pass
            cache.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ODriveSim:
    """Drop-in stand-in for quickstart's `lib/odrive_uart.ODriveUART`.

    The real ODrive runs a velocity loop onto motor current; this does the same
    thing with a PI loop onto actuator torque, so examples written against the
    hardware API drive the simulated robot unchanged.
    """

    def __init__(self, bot: BracketBot, kp=28.0, ki=90.0, torque_limit=15.0):
        self.bot = bot
        self.kp, self.ki, self.limit = kp, ki, torque_limit
        self.target = np.zeros(2)   # [left, right] m/s
        self._integral = np.zeros(2)
        self.active = False

    # -- lifecycle calls the examples make; no-ops here --------------------
    def start_left(self): self.active = True
    def start_right(self): self.active = True
    def enable_velocity_mode_left(self): pass
    def enable_velocity_mode_right(self): pass
    def disable_watchdog_left(self): pass
    def disable_watchdog_right(self): pass
    def clear_errors_left(self): self._integral[:] = 0
    def clear_errors_right(self): self._integral[:] = 0

    # -- commands ----------------------------------------------------------
    def set_speed_mps_left(self, v): self.target[0] = float(v)
    def set_speed_mps_right(self, v): self.target[1] = float(v)

    def get_position_turns_left(self):
        return float(self.bot.wheel_angles[0] / (2 * np.pi))

    def get_position_turns_right(self):
        return float(self.bot.wheel_angles[1] / (2 * np.pi))

    def get_speed_mps_left(self): return float(self.bot.wheel_speeds_mps[0])
    def get_speed_mps_right(self): return float(self.bot.wheel_speeds_mps[1])

    def update(self, dt):
        """Run one PI step; call this every physics step when in velocity mode."""
        err = self.target - self.bot.wheel_speeds_mps
        self._integral = np.clip(self._integral + err * dt, -self.limit,
                                 self.limit)
        tau = self.kp * err + self.ki * self._integral
        tau = np.clip(tau, -self.limit, self.limit)
        self.bot.set_wheel_torque(tau[0], tau[1])
