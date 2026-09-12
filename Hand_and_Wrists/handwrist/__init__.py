"""Hands and wrists for the BracketBot: grasp planning and pick skills.

The pipeline, in the order the data flows:

    objects    what each household item is and how to hold it
    gripper    finger-gap calibration and pad contact sensing
    grasping   object estimate -> which arm, wrist angle, grasp pose, where to park
    skills     Pick: approach -> reach -> close -> verify -> lift -> stow

The robot itself (model, balance controller, arm IK) lives in ../../main_mujoco
and is imported from there, not copied.
"""
import os
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
HAND_AND_WRISTS = PACKAGE_DIR.parent
REPO = HAND_AND_WRISTS.parent
MAIN_MUJOCO = REPO / "main_mujoco"
SCENES = HAND_AND_WRISTS / "scenes"
HOME_SCENE = SCENES / "scene_home.xml"

if str(MAIN_MUJOCO) not in sys.path:
    sys.path.insert(0, str(MAIN_MUJOCO))

# robot.py defaults MUJOCO_GL to egl, which Windows does not have. Callers that
# open a viewer set glfw themselves before importing this package.
if os.name == "nt":
    os.environ.setdefault("MUJOCO_GL", "wgl")
