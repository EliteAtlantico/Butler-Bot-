# HANDOFF — LLM scene-reasoning goal detector (pick up from here)

Branch: `feature/llm-scene-reasoning` (pushed to origin, tip `fbc40a4`).
Goal: let the BracketBot look at a head-camera frame, use the LOCAL LLM to
reason about where the goal (red column) is, convert that into a world
position, and drive to it.

## What exists (all committed on the branch)

- `comp_vision_sim/vision_sim/llm_reasoner.py`
  - `LlmGoalDetector` — callable detector `obs -> list[Detection]`. Sends the
    head RGB frame to the local llama-server (OpenAI-compatible) as a base64
    image, the model reasons about the scene and returns compact JSON
    (scene, goal_found, goal pixel, confidence, a `movement` sentence, other
    objects). Detector converts pixel -> world by fusing with the registered
    depth image (median of the goal's visible surface, exactly the method in
    `yolo_detector.py`); falls back to ray-to-ground when no depth under the
    pixel. Returns a `Detection` on the SAME contract as `perception.detect` /
    `YoloDetector`, so `VisualNavigator`'s A*/drive loop consumes it unchanged.
  - `LLMClient` — minimal chat client. KEY quirks that MUST stay:
    - Appends `\n\n/no_think` to the prompt AND sends `"enable_thinking":
      false`. Without this the Qwen build spends the whole token budget on
      internal reasoning and returns EMPTY content.
    - `max_tokens` default raised to 2048 (1024 caused one truncated-JSON
      failure during the long run; the error path catches it and reuses the
      last cached detection).
  - Gating: queries are rate-limited by SIM time (`t`), with a wall-clock
    fallback when `t` is None. The whole sim loop blocks on the LLM call, so
    the robot cannot move during a query regardless.

- `comp_vision_sim/run_navigation.py` — added `--detector llm` and
  `--llm-url` (default `http://localhost:8080/v1`), `--llm-model` (default
  `Qwen/Qwen3.8-27B`), `--llm-period` (sim s between queries, default 3.0),
  `--llm-conf` (min confidence, default 0.40).

- `comp_vision_sim/vision_sim/navigation.py` — `sense()` now passes `t` into
  the detector: `self.detector(self.obs, robot_yaw=bot.yaw, t=t)`.

- `comp_vision_sim/vision_sim/perception.py` — `detect()` signature gained
  `**_` so it accepts the new `t` kwarg (no-op for the colour detector).
  YOLO and LLM detectors already had `**_`.

- `comp_vision_sim/test_llm_reasoner.py` — offline test (no server needed):
  forward-projects the true goal into pixels, stubs the model to return them,
  checks the round-trip. PASSES: position error 0.145 m, bearing on axis,
  depth-fusion found 238 px; goal_found=false -> no detection; JSON extraction
  robust to fences/trailing text.

- `vision_sim/__init__.py` — documents the new module.

## How to run

```
cd comp_vision_sim
../.venv/bin/python run_navigation.py --detector llm --llm-period 3.0 --llm-conf 0.40 -d 90
```
(venv = `/home/kc/Projects/Team-14-Battle-of-The-Schools/.venv`, Python 3.14,
mujoco 3.13, numpy, PIL, requests all present.) Headless uses EGL; add
`--viewer` for the MuJoCo window. Use `-u` for unbuffered logs.

The LLM server: `http://localhost:8080` (llama-server, `Qwen/Qwen3.8-27B`
GGUF Q4, multimodal, 131k ctx). It is ALREADY RUNNING ON THE GPU — do NOT
restart it (user controls its lifecycle). Confirmed on RTX 5090, ~27 GB VRAM.

## Verified (real execution, this session)

- Offline geometry test: PASS (0.145 m round-trip).
- `py_compile` clean on all touched files.
- End-to-end headless `--detector llm` run (killed by user before it finished,
  NOT at arrival): the full loop worked —
  - SCAN: robot spun 360°; LLM correctly reported "no red goal in view,
    rotate to scan" when the goal rotated out of frame, and locked on when it
    came back.
  - Goal estimate from LLM: **(6.05, 0.01)** vs true **(6.0, 0.0)** -> within
    5 cm. Logged `t=9.4s scan done, goal at (6.05, 0.01), 3 waypoints`.
  - NAVIGATE: robot threaded the barrier gap; per-frame LLM reasoning was
    sensible and pixel-tracking was correct (target pixel centred ~150-160 as
    the robot closed in; movement text shifted "rotate ~20 deg right" ->
    "advance straight while keeping the red cylinder centred"). Ran ~15+ min
    (each query 7-17 s on the 27B model; ~30 gated queries over 90 sim s).
  - One query returned truncated JSON (old 1024 cap) -> error path reused the
    last good detection (graceful degradation worked).

## What the next LLM should do / known gaps

1. **REBASE or merge `origin/main` into the branch FIRST.** While this work
   was happening, `main` advanced past the branch base (`ce3a3d8`) and added:
   - `2c0d9ef` Fix sim-time re-seeds that fired on every physics step
   - `6e370d7` Restore CRLF line endings in navigation.py
   Both touch the clock-gating / navigation.py area this branch edits. Re-
   verify the detector still works after the merge (run `test_llm_reasoner.py`
   and a short `--detector llm` run). Note: line endings — `navigation.py` /
   `perception.py` use CRLF; keep that consistent on edit.
2. **Get a clean full run to ARRIVED** and confirm the final report prints
   `state=arrived` and `final position ... N.NN m from the column` (stop
   distance 1.0 m). The previous run was killed during final approach before
   the arrival line was printed.
3. **Confirm the default colour path is unaffected** by the `**_` addition:
   a short `run_navigation.py -d 14` should still scan, find the goal, and
   arrive. (py_compile passed; a full colour run was not completed before
   handoff.)
4. Optional polish: `max_tokens` is now 2048 — confirm no more truncated-JSON
   failures in a fresh run. Could add `--llm-max-tokens` flag if desired.
5. The `movement` string from the model is currently only logged (verbose) —
   the actual driving uses the model's pixel -> world position via the
   existing planner. If a "use the model's movement text directly" variant is
   wanted, that is a separate design decision; the current approach
   (pixel->world, then A*/follow) is the one built and tested.

## Design rationale (why it's shaped this way)

- The user asked the robot to "look at a scene, reason about object location,
  convert to movement data, move toward where it thinks an object may be."
  The implementation uses the EXISTING detector seam (`detector(obs,
  robot_yaw=...) -> list[Detection]`) that `VisualNavigator` already supports
  for colour + YOLO, so the LLM is a third detector and the proven depth-map +
  A* + follow loop is reused untouched. This gives reason->location->movement
  while the planner handles obstacle avoidance.
- The LLM is a slow, deep look (~10 s/query), not a per-frame classifier, so
  it is time-gated and caches its last detection between queries.

## Key files / paths

- Repo: `/home/kc/Projects/Team-14-Battle-of-The-Schools`
- New module: `comp_vision_sim/vision_sim/llm_reasoner.py`
- Entry: `comp_vision_sim/run_navigation.py`
- Test: `comp_vision_sim/test_llm_reasoner.py`
- LLM server: `http://localhost:8080` (do not restart)
- venv: `../.venv/bin/python` from `comp_vision_sim/`
- Remote: `git@github.com:EliteAtlantico/Team-14-Battle-of-The-Schools.git`
