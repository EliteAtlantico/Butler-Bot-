# HANDOFF — LLM scene reasoning and the 8-photo survey

Branch: `feature/llm-scene-reasoning`. It is up to date with `main` and with
`tianjun-computer-vision` (both merged in) and has NOT been merged into
`main`; that is deliberately left for later.

Goal: the BracketBot looks around with its head camera, the LOCAL LLM
reasons about where the goal (red column) is, that becomes a world position,
and the robot drives there.

**Status: working end to end, tested, verified live.** Two ways to use the
model, both on the same `run_navigation.py`:

| Mode | What happens | Live result (obstacle course) |
|---|---|---|
| `--detector llm` | the robot spins; the model is asked about one frame at a time | arrived 1.08 m from the column, 8 queries, 100.7 s wall |
| `--detector llm --survey` | the robot photographs 8 headings, the model is asked ONCE about all 8, then drives | arrived 1.12 m from the column, 1 survey query (30.1 s) + 5 frame queries, 90.7 s wall |

## How to run

```
cd comp_vision_sim
../.venv/bin/python -u run_navigation.py --detector llm --survey -d 120    # survey, then drive
../.venv/bin/python -u run_navigation.py --detector llm -d 90             # per-frame only
../.venv/bin/python -m unittest -v test_llm_scene_reasoning test_llm_survey
```

venv = `/home/kc/Projects/Team-14-Battle-of-The-Schools/.venv` (Python 3.14,
mujoco 3.13, numpy, PIL, requests; matplotlib + ultralytics for the figure and
YOLO). pytest is NOT installed; the tests are stdlib `unittest` and pytest
collects them if added. Headless uses EGL; add `--viewer` for the window.

The LLM server: `http://localhost:8080` (llama-server, `Qwen/Qwen3.8-27B`
GGUF Q4_K_M + mmproj, multimodal, 131k ctx, 4 slots) on the RTX 5090. It is
already running — do NOT restart it; the user controls its lifecycle.

## The 8-photo survey (`--survey`)

`vision_sim/llm_survey.py`, wired into `VisualNavigator` as a SURVEY state that
replaces the spin scan.

1. The robot turns in place to N evenly spaced headings (8 → every 45°),
   waits for the balancing base to settle, and takes a registered RGB-D photo
   at each. It records the heading it ACTUALLY faced, maps every photo's depth
   into the occupancy grid, and skips the per-frame detector while surveying.
2. ONE request goes to the model with all photos. **Every photo is sent
   immediately after its own label** stating that photo's direction:

   > Photo 1: robot at (2.00, -2.50); this photo faces world heading 62 deg;
   > that is 45 deg counter-clockwise from photo 0; its left edge looks along
   > 99 deg and its right edge along 26 deg

   The model answers which photo holds the goal, the goal pixel in that photo,
   and the world heading toward it.
3. The answer is used by how much it can be trusted:
   photo + pixel → ranged against THAT photo's depth (full position) →
   else the pixel's exact bearing → else the model's heading → else not found.
4. Then: ranged goal → plan and drive. Heading only → a *provisional* goal the
   first real sighting replaces; reaching it unsighted falls back to a scan, it
   is never reported as ARRIVED. Not found or error → the ordinary spin scan.
   A coordinate goal (`--goal X,Y`) skips the survey.

Flags: `--survey`, `--survey-shots` (8), `--survey-max-tokens` (6144),
`--survey-conf` (0.40). The report prints the survey's answer, latency and
tokens.

**Token budget matters.** On multi-photo prompts the model reasons anyway
(into `reasoning_content`) despite `/no_think`. At `max_tokens` 1024 a hard
survey ran out before writing a single character of the answer (`finish=
length`, empty content). At 6144, surveys used 999–3127 tokens, 21–57 s.

Measured survey accuracy:
- start pose: photo 0, goal ranged to (6.05, 0.01) vs true (6.0, 0.0)
- off-axis pose (2.0, -2.5), goal straddling two photos: heading error 0.1°,
  position error 0.17 m (the case that failed at 1024 tokens)

## The per-frame detector (`--detector llm`)

`vision_sim/llm_reasoner.py`
- `LlmGoalDetector` — callable detector `obs -> list[Detection]` on the same
  contract as `perception.detect` / `YoloDetector`. Sends the head RGB frame,
  gets JSON (scene, goal_found, goal pixel, confidence, movement), ranges the
  pixel with the registered depth image (median of the goal's visible
  surface, as in `yolo_detector.py`), falls back to ray-to-ground when there
  is no depth under the pixel. Queries are gated on SIM time `t` (wall clock
  when `t` is None) and the last detection is cached between queries.
- `LLMClient` — `ask_image` (one frame) and `ask_images` (several, each with a
  text label). Quirks that MUST stay, enforced by tests: `\n\n/no_think`
  appended AND `"enable_thinking": false` sent. Records `finish_reason`, token
  usage and reasoning length.

Flags: `--llm-url` (`http://localhost:8080/v1`), `--llm-model`
(`Qwen/Qwen3.8-27B`), `--llm-period` (3.0 sim s), `--llm-conf` (0.40),
`--llm-max-tokens` (2048). The report prints query/error/truncation counts.

## Bugs found and fixed (each reproduced first, each has a regression test)

In `llm_reasoner.py`:
1. **A `}` inside a JSON string lost the whole query** — the extractor counted
   braces. Now uses `json.JSONDecoder.raw_decode`.
2. **`"goal_found": "false"` counted as found** — a non-empty string is truthy.
   Parsed as a real boolean now.
3. **Blind after a sim reset** — no query until `t` climbed back past the
   pre-reset query time (43 s in the repro). A backwards clock queries at once.
4. **Retry storm with the server down** — only successes stamped the query
   time, so every sense tick retried (10 attempts in 10 ticks), each able to
   block the sim for the 120 s timeout. Attempts are stamped now.

Brought in by merging `main` first: without the sim-clock fix from `main`, the
colour detector on this branch took 277 s of wall time for 30 s of sim and
never arrived.

## Merges

- `main` merged twice (latest brings `SceneInfo` / `VisualNavigator.for_bot`,
  `--goal`, `--camera`, the geometric detector, the YOLO channel-order fix).
  Conflicts in `run_navigation.py`, `navigation.py`, `perception.py` were
  resolved by keeping both sides.
- `tianjun-computer-vision` (`2f8649a`: apartment and moving-obstacle scenes,
  `demo_tour.py`, `--animate-movers`, real geom extents for bounds, obstacle
  ceiling from the robot's height). One conflict in `navigation.py`: the
  ceiling fix now lives inside `_integrate()`, so survey photos get the
  doorway fix too.
- After the CV merge, the apartment run (`--detector geometric --goal 9,-2`)
  and the moving-obstacle run printed exactly the same result as on the CV
  branch itself: apartment arrives 1.13 m from its goal; moving obstacle falls
  at t=18.7 s, the known failure that `demo_tour.py` shows on purpose.

## Tests

`test_llm_scene_reasoning.py` (44) and `test_llm_survey.py` (32), all offline
except one live test in each, which skip themselves when the server is down
(`SKIP_LIVE_LLM=1` forces the skip; `LLM_URL` retargets them).

| Area | Covered |
|---|---|
| Parsing | compact / fenced + nested / prose around it / braces inside strings / truncated or absent JSON |
| Coercion | `goal_found` as a `"false"` string; int / float parsing |
| Cadence | sim-time gating, cache, reset, failure back-off, bad JSON keeps last good, wall-clock fallback, clock switch, truncation counted |
| Requests | URL, model, token cap, timeout, `/no_think` + `enable_thinking:false`, PNG data URLs, HTTP errors; survey: 8 photos in one request, each label immediately before its own photo and naming its own heading and edges |
| Geometry | pixel → world within 0.5 m at 4 robot poses with correct bearing sign; label edge headings match pixel bearings within 0.5°; neighbouring photos overlap |
| Survey logic | ranged / pixel / model / photo-heading answers, rejections, dict pixel form |
| Navigation | per-frame LLM detector reaches ARRIVED with an oracle model; survey photographs 8 headings in place (<0.35 m drift) then arrives; not found → scan; heading-only provisional goal replaced then arrives; reaching a provisional goal unsighted is not ARRIVED; coordinate goals skip the survey; no per-frame queries while surveying; colour detector still arrives |
| CLI | every `--llm-*` and `--survey-*` flag reaches its object; defaults |
| Live | one real frame query; one real 8-photo survey from the off-axis pose |

Final results on the merged branch: **44/44 and 32/32 pass** (live tests
included: frame query 22.4 s, 0.15 m; survey 34.5 s, 0.1° heading error,
0.17 m position error). `test_llm_reasoner.py`, the original offline script,
also passes (0.145 m).

## Open / next steps

1. **Merge into `main`** when the user says so — intentionally not done.
2. **Model speed.** A survey query is 20–60 s and each frame query ~8–15 s,
   during which the sim is frozen. A faster VLM would shorten wall time; the
   survey already cuts queries (5 + 1 vs 8 per run here).
3. **Survey only at the start.** A lost goal or a STUCK state does not
   re-survey; that would be a small extension of the SURVEY state.
4. **The `movement` sentence** from the per-frame detector is logged, not
   used; driving uses pixel → world → A*, which also avoids obstacles.
5. **The prompts describe the obstacle course** (red column, orange barriers,
   blue pillars). Other scenes (e.g. `apartment.xml`) need the goal
   description changed before `--detector llm` / `--survey` make sense there.
6. Pose is ground truth (`bot.position`, `bot.yaw`), as for every detector here.

Line endings: `run_navigation.py` and `navigation.py` are CRLF; `perception.py`,
`llm_reasoner.py`, `llm_survey.py` and the tests are LF. Keep each file's
convention when editing.

## Key files

- `comp_vision_sim/vision_sim/llm_survey.py` — the survey
- `comp_vision_sim/vision_sim/llm_reasoner.py` — per-frame detector + client
- `comp_vision_sim/vision_sim/navigation.py` — SURVEY state
- `comp_vision_sim/run_navigation.py` — entry point and flags
- `comp_vision_sim/test_llm_scene_reasoning.py`, `test_llm_survey.py` — tests
- Remote: `git@github.com:EliteAtlantico/Team-14-Battle-of-The-Schools.git`
