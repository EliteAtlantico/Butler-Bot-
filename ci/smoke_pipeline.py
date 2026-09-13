#!/usr/bin/env python3
"""End-to-end smoke of the merged pipeline, deterministic and offline.

    python ci/smoke_pipeline.py give-up    # command -> explore -> give up -> phone remote
    python ci/smoke_pipeline.py pick       # drive to the sideboard -> Seam C pick of the remote
    python ci/smoke_pipeline.py            # both

Runs the real `run_navigation.main()` on the D2 demo scene. Only what CI
cannot have is stubbed: the LLM (no llama-server) in `give-up`, and nothing
at all in `pick`, which needs no model. The unit suites cover each stack;
this is what catches a seam coming apart.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
VISION = ROOT / "comp_vision_sim"
SCENE = VISION / "home_search.xml"
for _p in (VISION, ROOT / "main_mujoco", ROOT / "Hand_and_Wrists", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
# Before anything imports mujoco: some modules default MUJOCO_GL to egl,
# which does not exist on Windows.
os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")


def _check(ok: bool, what: str, failures: list[str]):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def _counting_bots():
    from bracketbot_sim.robot import BracketBot

    built = []
    real = BracketBot.__init__

    def init(self, *args, **kwargs):
        built.append(self)
        real(self, *args, **kwargs)
    return built, mock.patch.object(BracketBot, "__init__", init)


def give_up() -> list[str]:
    """A search for something that is not there ends at the phone remote,
    driving the same robot the navigator was holding."""
    import numpy as np

    import integration.remote_adapter as remote
    import run_navigation as rn
    from vision_sim.llm_command import SearchTask
    from vision_sim.llm_explore import ExploreResult, LlmExplorer

    task = SearchTask(command="find my sunglasses", target="sunglasses", description="my sunglasses",
                      likely_places=["coffee table", "sideboard"], reply="I'll look for your sunglasses.")

    def nowhere(self, shots, visited=()):
        self.queries += 1
        return self._finish(ExploreResult(notes=["every direction is blocked"]))

    seen, failures = {}, []
    real_hand_off = remote.hand_off

    def hand_off(bot, **kwargs):
        def phone(server):
            base = f"http://127.0.0.1:{server.server_address[1]}"

            def get(path):
                with urllib.request.urlopen(base + path, timeout=30) as reply:
                    return reply.read()

            def post(path, body):
                request = urllib.request.Request(base + path, json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=30) as reply:
                    return json.loads(reply.read())
            try:
                status = json.loads(get("/api/status"))
                seen["cameras"] = status["capabilities"]["cameras"]
                seen["start"] = np.array(bot.position[:2])
                seen["remote_start"] = np.array(status["telemetry"]["position"][:2])
                seen["frame"] = get("/api/camera?name=head_depth")[:2]
                for _ in range(20):                       # hold the stick; watchdog 0.35 s
                    post("/api/drive", {"linear": 1.0, "angular": 0.0})
                    time.sleep(0.1)
                post("/api/stop", {})
                time.sleep(0.5)
                seen["end"] = np.array(bot.position[:2])
                seen["remote_end"] = np.array(json.loads(get("/api/status"))["telemetry"]["position"][:2])
            except Exception as error:                     # reported below, not swallowed
                seen["error"] = repr(error)
            finally:
                server.shutdown()
        seen["bot"] = bot
        kwargs["on_ready"] = lambda server: threading.Thread(target=phone, args=(server,), daemon=True).start()
        return real_hand_off(bot, **kwargs)

    built, counting = _counting_bots()
    argv = ["run_navigation.py", "--scene", str(SCENE), "--command", task.command,
            "--max-explore-steps", "2", "--remote-port", "0"]
    with counting, mock.patch.object(sys, "argv", argv), \
            mock.patch("vision_sim.llm_command.CommandInterpreter.parse", return_value=task), \
            mock.patch.object(LlmExplorer, "query", nowhere), \
            mock.patch.object(rn, "build_detector", lambda args, info=None: (lambda obs, **_: [])), \
            mock.patch.object(remote, "hand_off", hand_off), \
            mock.patch.object(rn, "figure", lambda *a, **k: None):
        rn.main()

    print("give-up:")
    _check("error" not in seen, f"phone session ran ({seen.get('error', 'no error')})", failures)
    _check("bot" in seen, "the search gave up and handed off", failures)
    _check(len(built) == 1, f"exactly one BracketBot exists (built {len(built)})", failures)
    if "end" in seen:
        _check(seen["bot"] is built[0], "the remote drives the navigator's own robot", failures)
        _check(bool(np.allclose(seen["start"], seen["remote_start"], atol=0.02)),
               "the remote reports the pose the navigator left", failures)
        _check(float(np.linalg.norm(seen["end"] - seen["start"])) > 0.1, "driving from the remote moved it", failures)
        _check(bool(np.allclose(seen["end"], seen["remote_end"], atol=0.05)),
               "the remote reports where it went", failures)
        _check({"head_depth", "wrist_cam_left", "wrist_cam_right"} <= set(seen["cameras"]),
               "head_depth and both wrist cams are selectable", failures)
        _check(seen["frame"] == b"BM", "head_depth streams", failures)
    return failures


def pick() -> list[str]:
    """Drive to the sideboard by coordinate and pick up the remote (Seam C),
    no model.

    The arm is told where the remote is (the truth estimator). Seam C is the
    hand-over from navigator to arm, and that is what this checks; finding
    things with the camera is YOLO-World's job since Pick went object-first,
    and has its own tests. A smoke that failed on detection would say nothing
    about the seam."""
    import integration.pick_adapter as adapter
    import run_navigation as rn

    seen, failures = {}, []
    real_pick = adapter.pick_after_arrival

    def pick_after_arrival(bot, target, **kwargs):
        seen["outcome"] = real_pick(bot, target, **kwargs)
        return seen["outcome"]

    built, counting = _counting_bots()
    argv = ["run_navigation.py", "--scene", str(SCENE), "--detector", "geometric",
            "--goal", "9.35,-2.2", "--target", "remote", "--pick", "--pick-truth"]
    with counting, mock.patch.object(sys, "argv", argv), \
            mock.patch.object(adapter, "pick_after_arrival", pick_after_arrival), \
            mock.patch.object(rn, "figure", lambda *a, **k: None):
        rn.main()

    print("pick:")
    outcome = seen.get("outcome")
    _check(outcome is not None, "the navigator arrived and handed over to the arm", failures)
    if outcome is not None:
        _check(outcome.object_name == "remote", f"'remote' resolved to {outcome.object_name!r}", failures)
        _check(bool(outcome.succeeded), f"picked up the remote ({outcome.summary()})", failures)
    _check(len(built) == 1, f"exactly one BracketBot exists (built {len(built)})", failures)
    return failures


CASES = {"give-up": give_up, "pick": pick}


def main(argv=None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(CASES)
    failures = []
    for name in names:
        failures += [f"{name}: {f}" for f in CASES[name]()]
    print("\nSMOKE " + ("PASSED" if not failures else "FAILED:\n  " + "\n  ".join(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
