from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_mujoco"
VISION = ROOT / "comp_vision_sim"

for path in (str(MAIN), str(VISION), str(ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")


@pytest.fixture(scope="session")
def model_path() -> Path:
    return MAIN / "scene_dynamic.xml"


@pytest.fixture(scope="session")
def table_model_path() -> Path:
    return MAIN / "scene_table.xml"


@pytest.fixture
def drive_bot():
    import numpy as np

    class Bot:
        def __init__(self):
            self.position = np.array([0.0, 0.0, 0.0])
            self.yaw = 0.0
            self.ground_speed = 0.0
            self.commands = []
            self.depth_calls = 0
            self.cloud = np.empty((0, 3))

        def drive(self, v=0.0, w=0.0):
            self.commands.append((float(v), float(w)))

        def depth(self, _name, width, height, max_range):
            self.depth_calls += 1
            return np.full((height, width), max_range, dtype=float)

        def point_cloud_world(self, *_args, **_kwargs):
            return self.cloud.copy()

    return Bot()


@pytest.fixture(scope="session")
def robot():
    from bracketbot_sim.robot import BracketBot

    bot = BracketBot(xml=MAIN / "scene_dynamic.xml")
    yield bot
    bot.close()
