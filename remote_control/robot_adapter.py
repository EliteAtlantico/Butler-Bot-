"""Adapters between the remote server and existing BracketBot interfaces."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from io import BytesIO
import os
from pathlib import Path
import struct
import sys
import threading
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MUJOCO = PROJECT_ROOT / "main_mujoco"
DEFAULT_SCENE = PROJECT_ROOT / "Hand_and_Wrists" / "scenes" / "scene_home.xml"

USER_CAMERAS = (
    # A third-person view that follows the robot around the room (the model's
    # trackcom `chase` camera): what is going on, not just what the robot sees.
    # Shown from above and behind the robot, over the walls: see display_camera().
    {"id": "scene", "label": "Scene View", "model_name": "chase", "rgbd": False,
     "follow_from": "head_depth"},
    # Shown with the head depth sensor's downward tilt: see display_camera().
    {"id": "head-left", "label": "Left Head RGB-D",
     "model_name": "head_stereo_left", "rgbd": True, "tilt_from": "head_depth"},
    {"id": "head-right", "label": "Right Head RGB-D",
     "model_name": "head_stereo_right", "rgbd": True, "tilt_from": "head_depth"},
    {"id": "wrist-left", "label": "Left Wrist Camera",
     "model_name": "wrist_cam_left", "rgbd": False},
    {"id": "wrist-right", "label": "Right Wrist Camera",
     "model_name": "wrist_cam_right", "rgbd": False},
)


def display_camera(model, data, camera: dict):
    """What the live view renders for `camera`: the model camera by name, or -- for
    a view with `tilt_from` -- a free camera at that camera's position looking along
    the `tilt_from` camera's direction.

    The stereo head cameras in chopped_dynamic.xml look dead level from 1.54 m, so
    with a 55 deg field of view no floor nearer than about 3 m is in frame: the
    view was sky above an empty plane, and the table right in front of the robot
    was out of shot. The head depth camera the robot perceives through is tilted
    22 deg down. The display borrows that tilt and leaves the robot model's sensors
    as they are.
    """
    import math

    import mujoco
    import numpy as np

    follow_from = camera.get("follow_from")
    if follow_from:
        return _follow_camera(model, data, follow_from)
    tilt_from = camera.get("tilt_from")
    if not tilt_from:
        return camera["model_name"]
    at = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera["model_name"])
    tilt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, tilt_from)
    forward = -np.asarray(data.cam_xmat[tilt]).reshape(3, 3)[:, 2]    # cameras look along -z
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    view.distance = 1.0
    view.lookat[:] = np.asarray(data.cam_xpos[at]) + forward
    view.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
    view.elevation = math.degrees(math.asin(max(-1.0, min(1.0, float(forward[2])))))
    return view


FOLLOW_AHEAD = 1.0         # m in front of the robot the view is centred on
FOLLOW_DISTANCE = 3.6      # m from that point
FOLLOW_ELEVATION = -70.0   # deg: the eye is ~4 m up, above 2.4 m house walls


def _follow_camera(model, data, facing_camera: str):
    """A free camera behind the robot, looking down at it along its heading.

    The model's `chase` camera trails 3 m back at 1.6 m high. In the living room
    that is open floor. In the house (home_search.xml) it sits inside a wall and the
    scene view was a grey slab. Looking down steeply from above wall height sees
    into whichever room the robot is in, since the scenes have no ceilings. Centring
    the view a metre ahead keeps a wall just behind the robot out of the frame. The
    heading comes from `facing_camera` (the head depth sensor), so it turns with
    the robot."""
    import math

    import mujoco
    import numpy as np

    facing = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, facing_camera)
    forward = -np.asarray(data.cam_xmat[facing]).reshape(3, 3)[:, 2]
    chassis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "chassis")
    root = int(model.body_rootid[chassis]) if chassis >= 0 else 0
    view = mujoco.MjvCamera()
    view.type = mujoco.mjtCamera.mjCAMERA_FREE
    level = np.array([forward[0], forward[1], 0.0])
    level /= max(float(np.linalg.norm(level)), 1e-9)
    view.lookat[:] = data.subtree_com[root] + FOLLOW_AHEAD * level
    view.distance = FOLLOW_DISTANCE
    view.azimuth = math.degrees(math.atan2(forward[1], forward[0]))
    view.elevation = FOLLOW_ELEVATION
    return view


@dataclass(frozen=True)
class JointCapability:
    name: str
    label: str
    side: str
    minimum: float
    maximum: float
    step: float
    unit: str


class SimulationRobotAdapter:
    """Uses the repository's BracketBot API without changing it."""

    def __init__(self, scene: str | Path = DEFAULT_SCENE):
        os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")
        if str(MAIN_MUJOCO) not in sys.path:
            sys.path.insert(0, str(MAIN_MUJOCO))

        import mujoco
        from bracketbot_sim.robot import BracketBot

        self._mujoco = mujoco
        self.scene = Path(scene).resolve()
        self._setup(BracketBot(xml=self.scene), owns_bot=True)
        # Drop the scene's loose items onto real contacts before anyone drives or grasps.
        hand_path = PROJECT_ROOT / "Hand_and_Wrists"
        if str(hand_path) not in sys.path:
            sys.path.insert(0, str(hand_path))
        from handwrist.surfaces import settle_loose_items
        with self._lock:
            settle_loose_items(self.bot)

    @classmethod
    def attach(cls, bot) -> "SimulationRobotAdapter":
        """Drive a BracketBot that is already running, instead of building one.

        For handing a live robot over mid-run: the operator then drives the
        robot that was already in the scene, not a fresh one at the origin.
        The caller keeps ownership, so close() stops the robot but leaves its
        GL contexts for the caller to release. MUJOCO_GL is left alone -- it
        was chosen when the bot was built.
        """
        import mujoco

        self = cls.__new__(cls)
        self._mujoco = mujoco
        self.scene = None
        self._setup(bot, owns_bot=False)
        return self

    @property
    def lock(self):
        """Taken by every physics step and camera render on this robot."""
        return self._lock

    def _setup(self, bot, owns_bot: bool):
        self._lock = threading.RLock()
        self._control_renderers_closed = False
        self.bot = bot
        self._owns_bot = owns_bot
        # Seeds the balance references from where the robot stands now, so an
        # attached robot holds its current spot rather than a stale reference.
        self.bot.balance.enable(self.bot.state)
        self._joints = self._discover_joints()
        self._gripper_joints = {
            "left": ("left_left_gripper", "left_right_gripper"),
            "right": ("right_left_gripper", "right_right_gripper"),
        }
        available = set(self.bot.camera_names)
        missing = [name for camera in USER_CAMERAS
                   for name in (camera["model_name"], camera.get("tilt_from"), camera.get("follow_from"))
                   if name and name not in available]
        if missing:
            raise ValueError(f"Scene is missing required robot cameras: {', '.join(missing)}")
        # A free camera takes its field of view from the model's visual settings, not
        # from a camera. The live view renders a private copy so the tilted head views
        # get the stereo cameras' 55 deg without changing anyone else's model (the viewer).
        import copy
        self._display_model = copy.deepcopy(self.bot.model)
        head = next(camera for camera in USER_CAMERAS if camera.get("tilt_from"))
        head_id = self._mujoco.mj_name2id(self.bot.model, self._mujoco.mjtObj.mjOBJ_CAMERA,
                                          head["model_name"])
        self._display_model.vis.global_.fovy = float(self.bot.model.cam_fovy[head_id])
        self._camera_condition = threading.Condition()
        self._camera_wakeup = threading.Event()
        self._camera_stop = threading.Event()
        self._active_camera = None
        self._camera_generation = 0
        self._frame_sequence = 0
        self._latest_frame = None
        self._last_delivered_sequence = 0
        self._dropped_frames = 0
        self._render_times = deque(maxlen=90)
        self._last_camera_request = 0.0
        self._render_thread = threading.Thread(
            target=self._render_loop, name="butlerbot-camera", daemon=True)
        self._render_thread.start()

    def _discover_joints(self) -> list[JointCapability]:
        result: list[JointCapability] = []
        for name in self.bot.arm_joints:
            if "gripper" in name:
                continue
            joint_id = self.bot._jnt_id(name)
            lower, upper = (float(v) for v in self.bot.model.jnt_range[joint_id])
            side = "left" if name.startswith("lj") else "right"
            index = int(name[2:])
            kind = "Lift" if index == 0 else f"Joint {index}"
            unit = "m" if index == 0 else "rad"
            result.append(JointCapability(
                name=name,
                label=f"{side.title()} {kind}",
                side=side,
                minimum=lower,
                maximum=upper,
                step=0.01,
                unit=unit,
            ))
        return result

    @property
    def capabilities(self) -> dict[str, object]:
        return {
            "backend": "simulation",
            "drive": True,
            "arm_joints": [asdict(joint) for joint in self._joints],
            "grippers": ["left", "right"],
            "cameras": list(self.bot.camera_names),
            "mobile_cameras": [dict(camera) for camera in USER_CAMERAS],
            "model_cameras": list(self.bot.camera_names),
            "perception_camera": "head_depth",
            "autonomous_tasks": ["pick", "fetch", "put", "tidy"],
        }

    def step(self, duration: float, linear_mps: float, angular_rads: float):
        with self._lock:
            self.bot.drive(linear_mps, angular_rads)
            self.bot.step(duration)

    def step_controller(self, duration: float, controller):
        """Step an existing navigation/manipulation controller on this bot."""
        with self._lock:
            self.bot.step(duration, controller=controller)

    def stop(self):
        with self._lock:
            self.bot.drive(0.0, 0.0)

    def hold_position(self):
        """Stop the base and freeze every arm target at its measured position."""
        with self._lock:
            self.bot.drive(0.0, 0.0)
            for name in self.bot.arm_joints:
                joint_id = self.bot._jnt_id(name)
                address = self.bot.model.jnt_qposadr[joint_id]
                self.bot.set_arm_target(name, float(self.bot.data.qpos[address]))

    def cancel_chore(self, controller=None):
        """Use the chore controller's own cleanup hooks, then freeze motion."""
        with self._lock:
            current = getattr(controller, "current", None) or controller
            manipulating = getattr(current, "_manipulating", None)
            if callable(manipulating):
                manipulating(False)
            try:
                from handwrist.skills import expire_nudge
                expire_nudge(self.bot, force=True)
            except ImportError:
                pass
            self.hold_position()

    def make_chore(self, action: str, **arguments):
        """Build the merged Hand_and_Wrists high-level task for this same bot."""
        hand_path = PROJECT_ROOT / "Hand_and_Wrists"
        vision_path = PROJECT_ROOT / "comp_vision_sim"
        for path in (hand_path, vision_path):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        from handwrist.tasks import make_task

        with self._lock:
            return make_task(self.bot, action, verbose=False, **arguments)

    def set_arm_target(self, joint: str, value: float) -> float:
        capability = next((item for item in self._joints if item.name == joint), None)
        if capability is None:
            raise ValueError(f"Unsupported arm joint: {joint}")
        target = max(capability.minimum, min(capability.maximum, float(value)))
        with self._lock:
            self.bot.set_arm_target(joint, target)
        return target

    def set_gripper(self, side: str, action: str):
        sides = tuple(self._gripper_joints) if side == "both" else (side,)
        if any(item not in self._gripper_joints for item in sides):
            raise ValueError("Gripper side must be left, right, or both.")
        if action not in {"open", "close", "hold"}:
            raise ValueError("Gripper action must be open, close, or hold.")

        with self._lock:
            for selected_side in sides:
                for joint in self._gripper_joints[selected_side]:
                    if action == "hold":
                        joint_id = self.bot._jnt_id(joint)
                        address = self.bot.model.jnt_qposadr[joint_id]
                        target = float(self.bot.data.qpos[address])
                    else:
                        target = 0.0 if action == "open" else 1.0
                    self.bot.set_arm_target(joint, target)

    def telemetry(self) -> dict[str, object]:
        with self._lock:
            joint_positions = {}
            for capability in self._joints:
                joint_id = self.bot._jnt_id(capability.name)
                address = self.bot.model.jnt_qposadr[joint_id]
                joint_positions[capability.name] = float(self.bot.data.qpos[address])
            return {
                "simulation_time": self.bot.time,
                "forward_velocity": self.bot.forward_velocity,
                "pitch_radians": self.bot.pitch,
                "fallen": self.bot.fallen,
                "position": [float(v) for v in self.bot.position],
                "arm_positions": joint_positions,
            }

    def camera_diagnostics(self) -> dict[str, object]:
        with self._camera_condition:
            latest = self._latest_frame
            active = self._active_camera
            if latest is None or "error" in latest:
                return {
                    "active_camera": active[0] if active else None,
                    "mode": active[4] if active else None,
                    "render_fps": 0.0,
                    "frame_age_ms": None,
                    "dropped_frames": self._dropped_frames,
                }
            return {
                "active_camera": latest["camera"],
                "mode": latest["mode"],
                "render_fps": latest["render_fps"],
                "frame_age_ms": max(
                    0.0, (time.monotonic() - latest["captured_monotonic"]) * 1000.0),
                "dropped_frames": latest["dropped_frames"],
                "sequence": latest["sequence"],
            }

    def camera_bmp(self, name: str, width: int = 480, height: int = 360) -> bytes:
        """Preserve the raw-camera snapshot API used by live-robot handoff."""
        if name not in self.bot.camera_names:
            raise ValueError(f"Unknown camera: {name}")
        with self._lock:
            rgb = self.bot.camera(name, width=width, height=height)
        return _rgb_to_bmp(rgb)

    def camera_frame(self, name: str, width: int = 320, height: int = 240,
                     mode: str = "rgb", after: int = 0,
                     timeout: float = 1.0, activate: bool = True,
                     stream_id: object | None = None) -> dict[str, object]:
        """Return only the newest active-camera frame, never a queued old frame."""
        camera = next((item for item in USER_CAMERAS if item["id"] == name), None)
        if camera is None:
            raise ValueError(f"Unknown user camera: {name}")
        if mode not in {"rgb", "depth"}:
            raise ValueError("Camera mode must be rgb or depth.")
        if mode == "depth" and not camera["rgbd"]:
            raise ValueError("Depth is available only for the head cameras.")

        configuration = (camera["id"], camera["model_name"], int(width),
                         int(height), mode)
        started = time.monotonic()
        deadline = started + max(0.05, float(timeout))
        with self._camera_condition:
            was_idle = started - self._last_camera_request > 1.5
            self._last_camera_request = started
            configuration_changed = configuration != self._active_camera
            if configuration_changed:
                if not activate:
                    raise RuntimeError("Camera stream was superseded by a newer selection.")
                self._active_camera = configuration
                self._camera_generation += 1
                self._latest_frame = None
                after = 0
                self._camera_wakeup.set()
            elif was_idle:
                self._camera_wakeup.set()
            generation = self._camera_generation
            while not self._camera_stop.is_set():
                latest = self._latest_frame
                if latest is not None and latest["generation"] == generation:
                    if "error" in latest:
                        raise RuntimeError(latest["error"])
                    if latest["sequence"] <= after:
                        latest = None
                if latest is not None:
                    self._last_delivered_sequence = max(
                        self._last_delivered_sequence, latest["sequence"])
                    result = dict(latest)
                    result["server_wait_ms"] = (
                        time.monotonic() - started) * 1000.0
                    return result
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("No fresh camera frame was available.")
                self._camera_condition.wait(remaining)
        raise RuntimeError("Camera renderer has stopped.")

    def _render_loop(self):
        """Continuously replace one active-camera frame on the WGL thread."""
        renderers = {}
        try:
            while not self._camera_stop.is_set():
                with self._camera_condition:
                    configuration = self._active_camera
                    generation = self._camera_generation
                    last_request = self._last_camera_request
                if configuration is None or time.monotonic() - last_request > 1.5:
                    self._camera_wakeup.wait(0.5)
                    self._camera_wakeup.clear()
                    continue

                camera_id, _, width, height, mode = configuration
                view = next(item for item in USER_CAMERAS if item["id"] == camera_id)
                render_started = time.monotonic()
                try:
                    key = (width, height, mode)
                    renderer = renderers.get(key)
                    if renderer is None:
                        renderer = self._mujoco.Renderer(
                            self._display_model, height=height, width=width)
                        if mode == "depth":
                            renderer.enable_depth_rendering()
                        renderers[key] = renderer
                    with self._lock:
                        renderer.update_scene(
                            self.bot.data,
                            camera=display_camera(self.bot.model, self.bot.data, view))
                    frame = renderer.render().copy()
                    if mode == "depth":
                        frame[frame > 8.0] = float("inf")
                        frame = _depth_to_rgb(frame, max_range=8.0)
                    image = _rgb_to_jpeg(frame, quality=62)
                    completed = time.monotonic()
                    with self._camera_condition:
                        if (configuration != self._active_camera
                                or generation != self._camera_generation):
                            self._dropped_frames += 1
                            continue
                        previous = self._latest_frame
                        if (previous is not None
                                and previous["sequence"] > self._last_delivered_sequence):
                            self._dropped_frames += 1
                        self._frame_sequence += 1
                        self._render_times.append(completed)
                        while (self._render_times
                               and completed - self._render_times[0] > 1.0):
                            self._render_times.popleft()
                        render_fps = (len(self._render_times) - 1) / max(
                            completed - self._render_times[0], 0.001)
                        self._latest_frame = {
                            "data": image,
                            "content_type": "image/jpeg",
                            "generation": generation,
                            "sequence": self._frame_sequence,
                            "captured_monotonic": completed,
                            "render_ms": (completed - render_started) * 1000.0,
                            "render_fps": render_fps,
                            "dropped_frames": self._dropped_frames,
                            "camera": configuration[0],
                            "mode": mode,
                        }
                        self._camera_condition.notify_all()
                except Exception as error:
                    with self._camera_condition:
                        self._latest_frame = {
                            "error": str(error),
                            "generation": generation,
                            "sequence": self._frame_sequence,
                        }
                        self._camera_condition.notify_all()

                elapsed = time.monotonic() - render_started
                self._camera_wakeup.wait(max(0.0, (1.0 / 30.0) - elapsed))
                self._camera_wakeup.clear()
        finally:
            for renderer in renderers.values():
                try:
                    renderer.close()
                except Exception:
                    pass

    def close_control_renderers(self):
        """Close perception GL contexts on the thread that created them."""
        with self._lock:
            if self._owns_bot and not self._control_renderers_closed:
                self.bot.close()
                self._control_renderers_closed = True

    def close(self):
        if self._render_thread.is_alive():
            self._camera_stop.set()
            self._camera_wakeup.set()
            with self._camera_condition:
                self._camera_condition.notify_all()
            self._render_thread.join(timeout=8.0)
        with self._lock:
            self.bot.drive(0.0, 0.0)
            # An attached bot belongs to whoever built it; closing its GL
            # contexts here as well is a double teardown.
            if self._owns_bot and not self._control_renderers_closed:
                self.bot.close()
                self._control_renderers_closed = True


def _rgb_to_jpeg(rgb, *, quality: int) -> bytes:
    """Fast, compact live-view encoding; perception keeps the original arrays."""
    from PIL import Image

    output = BytesIO()
    Image.fromarray(rgb, mode="RGB").save(
        output, format="JPEG", quality=quality, subsampling=2, optimize=False)
    return output.getvalue()


def _rgb_to_bmp(rgb) -> bytes:
    """Encode an RGB snapshot for the teammate handoff's legacy camera API."""
    import numpy as np

    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("Expected an RGB camera frame.")
    row_bytes = width * 3
    padding = (4 - row_bytes % 4) % 4
    rows = []
    for row in np.asarray(rgb, dtype=np.uint8)[::-1]:
        rows.append(np.ascontiguousarray(row[:, ::-1]).tobytes())
        if padding:
            rows.append(b"\0" * padding)
    pixels = b"".join(rows)
    offset = 14 + 40
    header = struct.pack("<2sIHHI", b"BM", offset + len(pixels), 0, 0, offset)
    dib = struct.pack("<IIIHHIIIIII", 40, width, height, 1, 24, 0,
                      len(pixels), 2835, 2835, 0, 0)
    return header + dib + pixels


def _depth_to_rgb(depth, *, max_range: float):
    """Make an honest, fixed-scale visualization of a MuJoCo depth frame."""
    import numpy as np

    values = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(values)
    intensity = np.zeros(values.shape, dtype=np.uint8)
    scaled = 1.0 - np.clip((values - 0.15) / (max_range - 0.15), 0.0, 1.0)
    intensity[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
    return np.stack((intensity, intensity, intensity), axis=-1)
