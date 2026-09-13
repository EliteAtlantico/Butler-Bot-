# Phase 5 handoff — Seam D: give-up → remote control

Repo: `EliteAtlantico/Team-14-Battle-of-The-Schools`
Branch: **`main-merge-6pm`** @ `78b95cb` — everything below is pushed, tree clean.
`main` is frozen at `a795600` for the duration of the integration.

Read `plan(1).md` first, especially **§0.1 (ask, do not resolve ambiguity
yourself)** and **§0.2 (take the features as they are; do not build what is
missing)**. Those outrank everything here.

---

## 0. What is already done

| Phase | State |
|---|---|
| 0 — `.gitattributes`, line endings | done, `103d97b` |
| 1 — four merges | done, `e8452ff` / `697d8fb` / `17c602e` (main was already merged) |
| 1b/2 — bug fixes, test collection | done, `72e9441` |
| 3 — consistency fixes | done, `08ca476` |
| 4 — Seam C (arrival → arm) | done, `1beb2b4` + `78b95cb` |
| **5 — Seam D (give-up → remote)** | **yours** |
| 6 — seam tests | not started |
| 7 — CI | not started |

Decisions already taken by the team (do not relitigate):

- **D5 — Pick only.** Chores (place / hand-over / tidy) exist on the branch and
  work, but the integration exposes `Pick` alone. Adding an intent field to
  `llm_command` would be a new LLM capability, which §7 puts out of scope.
- **D2 — `comp_vision_sim/home_search.xml` is the end-to-end demo scene.** It
  now has both explorable rooms and the seven pickable objects.
- **D1 — one navigation camera** (`head_depth`). Nothing in the tree ever
  claimed otherwise; no edit was needed.
- The sideboard was lowered to **0.55 m** so the keys are actually reachable.

Still open and **yours to ask about**: **D3 (remote-control lifetime)**, D4
(YOLO weights), D6 (dependency pins), D7 (delete `tianjun-computer-vision`).

---

## 1. Your task, concretely

plan.md §5 Phase 5:

- [ ] Add `SimulationRobotAdapter.attach(bot)` beside the existing constructor,
      so the server drives an already-running `BracketBot` instead of building
      a second one. Keep the current constructor working for standalone use.
- [ ] Replace the two `print()` calls in `hand_off_to_remote_control`
      (`comp_vision_sim/run_navigation.py:230`) with: start the server against
      the live bot, print the URL, block until the operator releases.
- [ ] Confirm the phone UI's camera selector can name `head_depth` and both
      wrist cams. **Probably already true — see §3.**
- [ ] Lifetime per **D3** — ask first.

**Acceptance:** a run that cannot find its target ends at a reachable local
URL, and driving from the UI moves the same robot whose pose the navigator was
reporting — asserted by a one-instance test, not by eye.

---

## 2. The two-robot problem, precisely

`remote_control/robot_adapter.py:32`:

```python
def __init__(self, scene: str | Path = DEFAULT_SCENE):   # DEFAULT_SCENE = main_mujoco/scene_flat.xml
    os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")
    if str(MAIN_MUJOCO) not in sys.path:
        sys.path.insert(0, str(MAIN_MUJOCO))
    import mujoco
    from bracketbot_sim.robot import BracketBot
    self._mujoco = mujoco
    self._lock = threading.RLock()
    self.bot = BracketBot(xml=Path(scene).resolve())     # <-- builds its own robot
    self.bot.balance.enable(self.bot.state)
    self._joints = self._discover_joints()
    self._gripper_joints = {...}
```

So starting the server mid-run gives you **two independent MuJoCo sims**, and
the operator drives a robot that is not the one that got lost — in a different
scene, at the origin, with an empty map.

`attach()` has to do everything `__init__` does except build the bot. Things
that are easy to miss:

- **`close()` must not close a bot it does not own.** Today
  (`robot_adapter.py:141`) it calls `self.bot.close()`, which tears down the GL
  contexts. In attached mode the navigator still owns the bot and
  `run_navigation` closes it at the end; closing twice is how you get the EGL
  teardown tracebacks `robot.py:close` warns about.
- **`balance.enable()`** is already on for a bot arriving from the navigator.
  Re-enabling reseeds the references (`BalanceController.enable` takes the
  current state), which is probably what you want after a give-up, but it is a
  behavioural choice — decide deliberately.
- **`MUJOCO_GL` is already set** by the time you attach, and the navigator may
  be using `glfw` (viewer) rather than the adapter's `egl`/`wgl` default.
  `setdefault` will not fight it; do not overwrite it.
- **Who steps the physics.** `adapter.step()` calls `bot.step()`. In attached
  mode the navigator has stopped, so the server is the only stepper — fine. But
  if D3 comes back "always on", the server thread and the navigator loop would
  both step the same `MjData` and you need the lock to span both, which is a
  much bigger change. That is why D3 sizes this phase.

`server.py:277` `main()` builds the adapter from `args.scene`. You will need a
way in that does not go through `argparse` — a `serve(adapter, host, port)` or
equivalent. `RemoteServer` is at `server.py:134`, a `ThreadingHTTPServer`.

---

## 3. Two things the plan asks for that may already be true

Check before doing work:

1. **Camera selector.** `robot_adapter.capabilities()` returns
   `"cameras": list(self.bot.camera_names)`, and `static/app.js:91` populates
   the dropdown from `capabilities.cameras`. So the UI already offers whatever
   the model has — `head_rgb`, `head_depth`, both stereo, both wrist cams,
   `chase`. Verify by loading the page; do not rewrite the selector.
2. **`on_give_up` already fires.** `VisualNavigator.on_give_up` is wired and
   tested (`comp_vision_sim/test_command_search.py` asserts the hook fires with
   its reason). The work really is replacing two `print()`s, not building a
   give-up mechanism. plan.md §3.1 says this explicitly.

---

## 4. Where things are

```
comp_vision_sim/run_navigation.py:230   hand_off_to_remote_control()  <- the stub
comp_vision_sim/vision_sim/navigation.py  VisualNavigator.on_give_up
remote_control/robot_adapter.py:29      SimulationRobotAdapter
remote_control/robot_adapter.py:141     close()  <- ownership problem
remote_control/server.py:134            RemoteServer (ThreadingHTTPServer)
remote_control/server.py:277            main()   <- argparse entry
remote_control/static/app.js:91         camera dropdown
integration/pick_adapter.py             Seam C, as a model for Seam D's shape
```

`integration/` is the agreed home for seam glue. Its `__init__.py` states the
rule: nothing in the four subsystems imports it, so the dependency runs one
way and either side still works alone. Seam D belongs there too — something
like `integration/remote_adapter.py` — rather than in `run_navigation.py`.

---

## 5. How to run things

```bash
git clone https://github.com/EliteAtlantico/Team-14-Battle-of-The-Schools.git
cd Team-14-Battle-of-The-Schools
git config core.longpaths true          # REQUIRED on Windows, see §7
git checkout main-merge-6pm

python comp_vision_sim/run_navigation.py --scene comp_vision_sim/home_search.xml \
    --detector geometric --goal 9.35,-2.2 --target keys --pick      # Seam C end to end
python -m remote_control.server --help                              # the server as it stands
python Hand_and_Wrists/run_pick.py --object remote                  # arm skills alone
```

To force a give-up (which is what Seam D is triggered by), ask for something
that is not there, with few rounds:

```bash
python comp_vision_sim/run_navigation.py --scene comp_vision_sim/home_search.xml \
    --command "find my sunglasses" --max-explore-steps 2
```

---

## 6. Verification bar

Phase 4 set the bar at measuring rather than asserting; match it.

- The acceptance criterion names a **one-instance test**: assert exactly one
  `BracketBot` exists after handover. Count instances, do not eyeball the UI.
  Phase 6 will want this test anyway, so write it now.
- `eval_pick.py` reproduces `6a4c6a0` **bit for bit** (84/84 trials identical,
  report byte-identical, max `sim_time` delta 0.000 s). If your change moves
  those numbers, something is wrong — Seam D should not touch the arm at all.
- `eval_tasks.py` was re-running at handoff time and had reached 28/30 all
  passing and matching baseline before the box ran out of memory. **Finish
  that check.** Use `--workers 1` or 2; this machine has ~4 GB free and
  MuJoCo workers OOM at 3+.

---

## 7. Environment traps, all hit the hard way

- **`git config core.longpaths true` is required.** The longest mesh path is
  258 chars against Windows' 260 limit; a plain clone into a deep directory
  fails checkout. Clone somewhere short (`C:\bbm`, not under `OneDrive\Desktop\...`).
- **`MUJOCO_GL`**: `wgl` for offscreen on Windows, `glfw` for the viewer,
  `egl` on Linux, `osmesa` in a headless container without a GPU. `egl` does
  not exist on Windows and `import mujoco` itself fails under it.
- **Never pipe a long run through `tail`** — it buffers until the process
  exits and you see nothing. Redirect to a file and read that.
- **MuJoCo eval workers OOM** on this box above 2 workers.
- The repo has ~18 `sys.path.insert` hacks. plan.md §7 says leave them.

---

## 8. Open risk worth knowing

`--pick --viewer` reports `"still in <phase> when the window closed"` if you
close the window mid-pick, because the outcome is read after the loop. That is
honest but means viewer runs are not a clean pass/fail signal — use headless
when you need a verdict. Seam D will have the same shape of problem if the
server blocks the main loop: decide what "done" means before you write it.

## 9. Suggested first message to the team

D3 is the gate. Something like:

> Phase 5 needs D3 settled before I write it. On give-up, should the remote
> server (a) start on demand and block until the operator releases — simplest,
> and the navigator has already stopped so only the server steps the physics;
> or (b) run for the whole session with the navigator handing over — matches
> "see through its main camera to look for their object" better, but the
> server thread and the navigator loop would then both step the same MjData
> and the locking gets substantially harder. I would pick (a).
