# Hand & Wrists

Arm, wrist and gripper control for the BracketBot: recognising household
objects and picking them up. Everything for the `feature/manipulation` branch
lives in this folder.

The robot model itself stays in `../main_mujoco`; nothing here edits those
files. Gripper changes (finger pads, friction) are applied on top of
`main_mujoco/chopped_dynamic.xml` when the model loads, so rebuilding the base
model never wipes them out.

Planned contents:

- arm control: move a hand to a 3D point (inverse kinematics), open/close the
  gripper, detect a successful grasp
- a household scene with graspable objects
- perception-to-grasp: find an object with the head camera, reach, grab, lift
- `run_pick.py`: demo script with viewer and video recording
