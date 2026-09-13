"""Adapters between the remote server and existing BracketBot interfaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import struct
import sys
import threading


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_MUJOCO = PROJECT_ROOT / "main_mujoco"
DEFAULT_SCENE = MAIN_MUJOCO / "scene_flat.xml"


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
        self._lock = threading.RLock()
        self.bot = BracketBot(xml=Path(scene).resolve())
        self.bot.balance.enable(self.bot.state)
        self._joints = self._discover_joints()
        self._gripper_joints = {
            "left": ("left_left_gripper", "left_right_gripper"),
            "right": ("right_left_gripper", "right_right_gripper"),
        }

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
        }

    def step(self, duration: float, linear_mps: float, angular_rads: float):
        with self._lock:
            self.bot.drive(linear_mps, angular_rads)
            self.bot.step(duration)

    def stop(self):
        with self._lock:
            self.bot.drive(0.0, 0.0)

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

    def camera_bmp(self, name: str, width: int = 480, height: int = 360) -> bytes:
        if name not in self.bot.camera_names:
            raise ValueError(f"Unknown camera: {name}")
        with self._lock:
            rgb = self.bot.camera(name, width=width, height=height)
        return _rgb_to_bmp(rgb)

    def close(self):
        with self._lock:
            self.bot.drive(0.0, 0.0)
            self.bot.close()


def _rgb_to_bmp(rgb) -> bytes:
    """Encode a uint8 RGB array as a browser-compatible 24-bit BMP."""
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

