# HANDOFF — LLM scene reasoning, survey, YOLO search with LLM waypoints, re-searching

Branch: `feature/llm-scene-reasoning`. It is up to date with `main` and with
`tianjun-computer-vision` (both merged in) and has NOT been merged into
`main`; that is deliberately left for later.

Goal: the BracketBot searches for an object of interest (the red column).
The trained YOLO net looks for it in every camera frame; while it has not been
seen, the LOCAL LLM looks at 8 labelled photos and picks where to go next; the
robot drives there, looks again, and once YOLO finds the object it navigates to
it and stops at the stand-off distance, ready for a pick (the manipulation
branch is not merged yet). If the goal is lost or moves, it searches again.

**Status: working end to end, tested, verified live.**

| Run (live model) | Wall time | LLM queries | Result |
|---|---|---|---|
| **`--scene search_course.xml --explore`** (target hidden behind a wall) | **14.8–19.8 s** | 2 waypoints, 0 for the goal | YOLO found it at (6.49, −2.53) vs (6.5, −2.5); arrived 1.08–1.14 m away |
| `--detector llm --survey` | **8.7 s** (was 90.7 s) | 1 survey (6.1 s) | arrived 1.12 m from the column |
| `--detector llm` | **13.9 s** (was 100.7 s) | 8 frames, mean 1.5 s | arrived 1.15 m |
| survey, then the goal is moved mid-drive | **11.8 s** | 2 surveys, 0 frames | re-surveyed, arrived 1.10 m from the new spot |

## How to run

```
cd comp_vision_sim
../.venv/bin/python -u run_navigation.py --scene search_course.xml --explore -d 240   # YOLO search, LLM waypoints
../.venv/bin/python -u run_navigation.py --detector llm --survey -d 120    # survey, then drive
../.venv/bin/python -u run_navigation.py --detector llm -d 90             # per-frame only
../.venv/bin/python -m unittest -v test_llm_scene_reasoning test_llm_survey test_goal_recovery test_explore
```

venv = `/home/kc/Projects/Team-14-Battle-of-The-Schools/.venv` (Python 3.14,
mujoco 3.13, numpy, PIL, requests; matplotlib + ultralytics for the figure and
YOLO). pytest is NOT installed; the tests are stdlib `unittest` and pytest
collects them if added. Headless uses EGL; add `--viewer` for the window.

The LLM server: `http://localhost:8080` (llama-server, `Qwen/Qwen3.8-27B`
GGUF Q4_K_M + mmproj, multimodal, 131k ctx, 4 slots, `--image-min-tokens
1024`) on the RTX 5090. It is already running — do NOT restart it; the user
controls its lifecycle.

## YOLO search with LLM-chosen waypoints (`--explore`)

`vision_sim/llm_explore.py` + an EXPLORE state in `navigation.py`.

About the branch: there is no branch called "navigation". The navigation stack
lives on `tianjun-computer-vision`, which was already fully merged into this
branch; nothing new was on it. `dev-remote-control` (phone remote control) and
`feature/manipulation` (picking) are NOT merged.

The loop:
1. **Survey:** 8 photos at 45° steps, each labelled with its own direction.
2. **YOLO on every survey photo.** If it sees the object, go to it — the LLM is
   not asked at all.
3. **Otherwise ask the LLM once** (~6 s, or ~2 s when the prompt prefix is
   cached): report the goal if it can see it, else put a box
   (`explore_bbox_2d`, 0–1000) around the most promising open floor to drive to
   next — a doorway, a gap, unexplored space. The prompt lists the places
   already explored.
4. **Box → waypoint by geometry:** the box's column gives an exact bearing, the
   depth image along that column gives how far is open, and the waypoint is
   placed a margin short of the nearest obstacle (at most `--explore-step`,
   3.5 m). A waypoint within 1 m of an explored place, or an unusable answer,
   uses the no-LLM fallback: the most open, least explored direction.
5. **Drive there with A*, with YOLO running every frame.** The moment YOLO puts
   the object on the map, switch to navigating to it.
6. **On reaching the waypoint, survey again** (back to 1). An unreachable
   waypoint is marked explored and triggers a survey from where the robot is.
7. Stop after `--max-explore-steps` (8) places → STUCK, never a false ARRIVED.
   A lost goal (see below) re-enters the loop.

`--explore` makes YOLO the default detector and depth tracking the default;
`--detector` still overrides. An LLM detector is never run on all 8 survey
photos (that would be 8 model queries).

Live, search course: start → LLM picked (2.16, 1.34) toward the doorway →
LLM picked (4.41, 2.62) through it → YOLO found the goal in the next survey →
arrived 1.14 m away; 19.8 s wall, 2 LLM queries.

`search_course.xml`: a full-height wall with one 1.2 m doorway splits the space;
the target is behind it and invisible from the start in all 8 directions
(checked geometrically and with YOLO). Materials match the obstacle course, so
YOLO sees the same classes; walls are neutral grey.

## Speed: what made it ~10x faster

Wall time is almost entirely model time (sim, rendering and planning are ~3 s
of a run), so the levers are faster queries and fewer queries.

1. **Thinking is actually off now.** llama-server reads `enable_thinking`
   only from `chat_template_kwargs` — that is where the chat template looks.
   The top-level key sent before was silently ignored, so every answer
   carried 1–4k characters of hidden reasoning. `LLMClient(thinking=False)`
   sends both. Frame 17.2 s → 1.8 s, survey 31–57 s → 6 s.
   `--llm-thinking` turns it back on.
2. **Answers are `bbox_2d` boxes normalised to 0–1000.** With thinking off,
   *pixel* answers became unreliable (range errors up to 4.8 m, pixels off
   the image). The box is this Qwen VL model's native output. Measured with
   thinking off across 6 frame poses and 4 survey poses:

   | Answer format | Frame wall | Frame range error | Survey wall | Survey range error |
   |---|---|---|---|---|
   | centre pixel, thinking on | 9.2 s | ≤0.18 m (3/3) | 50.1 s | ≤0.16 m (2/2) |
   | centre pixel, thinking off | 3.3 s | up to 2.52 m (3/6) | 6.4 s | up to 4.42 m (1/4) |
   | pixel box, thinking off | 1.5 s | up to 4.83 m (0/6) — it answers 0–1000 anyway | — | — |
   | **`bbox_2d` 0–1000, thinking off** | **1.5 s** | **≤0.18 m (6/6)** | **6.6 s** | **≤0.17 m (4/4)** |

   `_goal_pixel()` converts the box (clipped, ordered); legacy pixel answers
   are still accepted. The no-think model's own `heading_deg` can be ~30°
   off, but the survey prefers the heading derived from the ranged box.
3. **No per-frame queries while driving after a survey.** `--track depth`
   (the default with `--survey`) confirms the goal from the depth image — is
   something still standing where the goal is? — instead of asking the
   detector every few seconds.

Not a lever: shrinking images. The server's `--image-min-tokens 1024` gives a
160×120 frame the same 1,464 prompt tokens as 320×240.

## Losing the goal, and finding it again (`navigation.py`)

- **Moved goal:** a sighting more than `relocate_distance` (1 m) from the
  current goal moves the goal there at once instead of averaging toward it.
- **Lost goal:** if the goal *should* be in view — in frame, within
  `sighting_range` (7 m), not behind something nearer — but is not seen (by
  the detector, or by depth with `--track depth`) for `lost_after` (4 s), the
  search re-runs from where the robot is: a new 8-photo survey if one is
  configured, otherwise a spin scan. The detector's cached sighting is
  dropped (`LlmGoalDetector.forget()`). Out of view never counts as missing.
- **No route:** 8 failed plans re-search too, instead of going STUCK
  (fixed `--goal X,Y` coordinates still go STUCK).
- **Limits:** at most `max_researches` (3), with `research_cooldown` (3 s)
  after each search. A re-search scan that finds nothing in
  `research_scan_turns` (2) turns counts as a failed search. Running out ends
  in STUCK — never a false ARRIVED. The very first search still spins until
  it finds something.

Flags: `--track {detector,depth}`, `--lost-after`, `--max-researches`,
`--llm-thinking`. The report prints how many times it re-searched.

## The 8-photo survey (`--survey`)

`vision_sim/llm_survey.py`, a SURVEY state in `VisualNavigator` that replaces
the spin scan.

1. The robot turns in place to N evenly spaced headings (8 → every 45°),
   waits for the balancing base to settle, and takes a registered RGB-D photo
   at each, recording the heading it ACTUALLY faced and mapping every photo's
   depth. The per-frame detector is not queried while surveying.
2. ONE request with all photos. **Every photo is sent immediately after its
   own label** stating that photo's direction:

   > Photo 1: robot at (2.00, -2.50); this photo faces world heading 62 deg;
   > that is 45 deg counter-clockwise from photo 0; its left edge looks along
   > 99 deg and its right edge along 26 deg

3. The answer is used by trust: photo + box → ranged against THAT photo's
   depth → else the box's exact bearing → else the model's heading → else not
   found (spin scan). A heading-only answer is a *provisional* goal that the
   first real sighting replaces; reaching it unsighted scans instead of
   reporting ARRIVED. A coordinate goal (`--goal X,Y`) skips the survey.

Flags: `--survey`, `--survey-shots` (8), `--survey-max-tokens` (6144, only
needed with thinking on), `--survey-conf` (0.40).

## The per-frame detector (`--detector llm`)

`vision_sim/llm_reasoner.py`
- `LlmGoalDetector` — callable detector `obs -> list[Detection]` on the same
  contract as `perception.detect` / `YoloDetector`. Sends the head frame, gets
  JSON (`goal_found`, `bbox_2d`, `goal_confidence`, a short `movement`),
  ranges the box centre with the registered depth image (median of the goal's
  visible surface, as in `yolo_detector.py`), ray-to-ground when there is no
  depth. Gated on SIM time `t`; caches the last detection between queries.
- `LLMClient` — `ask_image` and `ask_images` (each image with its label).
  `thinking=False` sends `/no_think` plus `chat_template_kwargs:
  {enable_thinking: false}`. Records `finish_reason`, token usage and
  reasoning length.

Flags: `--llm-url`, `--llm-model`, `--llm-period` (3.0 sim s), `--llm-conf`
(0.40), `--llm-max-tokens` (2048).

## Bugs found and fixed (each reproduced first, each has a regression test)

1. A `}` inside a JSON string lost the whole query (brace counter) →
   `json.JSONDecoder.raw_decode`.
2. `"goal_found": "false"` counted as found → real boolean parsing.
3. Blind after a sim reset (no query until `t` passed the pre-reset time) →
   a backwards clock queries at once.
4. Retry storm with the server down (only successes stamped the query time)
   → attempts are stamped.
5. Thinking was never off (top-level `enable_thinking` ignored by the server)
   → sent in `chat_template_kwargs`.
6. A vanished goal left a re-search scan spinning for ever → re-search scans
   give up after 2 empty turns, ending in STUCK after the cap.

## Merges

- `main` merged twice (SceneInfo / `VisualNavigator.for_bot`, `--goal`,
  `--camera`, geometric detector, YOLO channel-order fix); conflicts resolved
  by keeping both sides.
- `tianjun-computer-vision` (`2f8649a`: apartment and moving-obstacle scenes,
  `demo_tour.py`, `--animate-movers`, real geom extents, obstacle ceiling from
  the robot's height). The ceiling fix lives in `_integrate()`, so survey
  photos get it too. After the merge, the apartment and moving-obstacle runs
  printed exactly the same results as on the CV branch itself.

## Tests

| File | Tests | Covers |
|---|---|---|
| `test_llm_scene_reasoning.py` | 53 | JSON parsing, coercion, cadence, request payload, `bbox_2d` maths, pixel → world at 4 poses, oracle-model run to ARRIVED, colour/YOLO still work, CLI, 1 live |
| `test_llm_survey.py` | 35 | plan, per-photo direction labels vs camera geometry, interpretation incl. boxes, request/error paths, navigator survey in place → ARRIVED, fallbacks, CLI, 1 live |
| `test_goal_recovery.py` | 27 | thinking switch, `forget()`, visibility rules, cooldown/cap/no-route/reset, moved goal followed, moved goal re-surveyed with 0 frame queries, vanished goal → STUCK without arriving, normal runs never re-search, CLI, 2 live speed guards |
| `test_explore.py` | 24 | open-floor distance, doorway pixel → doorway waypoint, no waypoints into near walls, fallback avoids explored places, explore/goal/unusable/explored answers, prompt lists explored places, server down or bad answer still yields a waypoint; hidden goal found via LLM waypoints + YOLO and reached, YOLO seeing the goal in a survey skips the model, waypoint reached → survey again, unreachable waypoint → survey again, step cap → STUCK, lost goal → explore again, CLI; 2 live (waypoint pick, full search) |

Offline everything passes; live tests skip themselves when the server is down
(`SKIP_LIVE_LLM=1` forces it, `LLM_URL` retargets). Latest live results (all 6
pass): frame 1.7 s / 0.15 m; survey 6.0 s / 0.2° / 0.17 m; speed guards frame
1.7 s and survey 6.0 s with no reasoning; explore waypoint 2.1 s, 0.99 m from
the doorway; full live search 14.8 s wall, 2 LLM waypoints, arrived 1.08 m. `test_llm_reasoner.py` (original offline
script, legacy pixel answers) also passes.

## Open / next steps

1. **Merge into `main`** when the user says so — intentionally not done.
   `feature/manipulation` (picking) and `dev-remote-control` are not merged.
2. **Picking the object up** is left for the manipulation branch; the robot
   stops at the stand-off distance (1.0 m) from the object.
3. **The no-LLM fallback is not doorway-aware:** it picks the longest open run
   plus novelty, so without the model the search wanders the first room before
   finding the doorway. The LLM is what makes the search efficient.
4. **The YOLO net knows three classes** (target, barrier, pillar) from the
   obstacle-course renders; a different object of interest needs retraining.
5. **Re-search triggers only on things the robot can check:** it needs the
   goal's spot to be in view to notice it is gone. A goal that disappears
   while the robot faces away is noticed when the robot next looks there.
6. **Depth tracking confirms "something is there", not "the goal is
   there".** If the goal is swapped for another object in the same spot,
   `--track depth` will not notice; use `--track detector` for that.
7. **The prompts describe the obstacle course** (red column, orange barriers,
   blue pillars). Other scenes need the goal description changed.
8. **The first (non-explore) search never gives up** (spins until it finds something), by
   design; only re-searches are capped.
9. Pose is ground truth (`bot.position`, `bot.yaw`), as for every detector.

Line endings: `run_navigation.py` and `navigation.py` are CRLF;
`perception.py`, `llm_reasoner.py`, `llm_survey.py`, `llm_explore.py`,
`search_course.xml` and the tests are LF.
Keep each file's convention when editing.

## Key files

- `comp_vision_sim/vision_sim/llm_explore.py` — LLM waypoints for the search
- `comp_vision_sim/search_course.xml` — hidden-target scene
- `comp_vision_sim/vision_sim/llm_survey.py` — the survey
- `comp_vision_sim/vision_sim/llm_reasoner.py` — per-frame detector + client
- `comp_vision_sim/vision_sim/navigation.py` — SURVEY/EXPLORE states, tracking, re-search
- `comp_vision_sim/run_navigation.py` — entry point and flags
- `comp_vision_sim/test_llm_scene_reasoning.py`, `test_llm_survey.py`,
  `test_goal_recovery.py`, `test_explore.py` — tests
- Remote: `git@github.com:EliteAtlantico/Team-14-Battle-of-The-Schools.git`
