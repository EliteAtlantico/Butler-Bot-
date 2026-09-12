# Testing

The test suite covers the MuJoCo robot/control stack, dynamic model builder,
vision pipeline, occupancy mapping and A* planning, scene introspection,
kinematics/manipulation, YOLO integration, every MJCF scene, and all four
command-line programs.

Create an isolated environment and install the pinned dependencies:

```powershell
python -m venv .venv-tests
.venv-tests\Scripts\python -m pip install -r requirements-test.txt
```

Run the complete suite:

```powershell
.venv-tests\Scripts\python -m pytest
```

Run it with branch coverage:

```powershell
.venv-tests\Scripts\python -m pytest --cov=main_mujoco --cov=comp_vision_sim --cov-branch --cov-report=term-missing
```

On Linux/macOS, replace `.venv-tests\Scripts\python` with
`.venv-tests/bin/python`. Tests marked `integration` compile or exercise real
MuJoCo models; tests marked `rendering` create an off-screen GL context; and
tests marked `slow` exercise model generation or dataset generation.
