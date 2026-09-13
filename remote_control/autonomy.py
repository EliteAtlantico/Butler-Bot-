"""Glue the phone remote to the merged language and chore controllers.

This module owns no robot behaviour.  It asks the existing local-LLM client
what object the user means, maps that vocabulary through the integration seam,
and hands the resulting high-level action to ``handwrist.tasks.make_task``.
The returned task is the controller stepped by the remote's single MuJoCo loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import sys
import threading


PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _path in (
    PROJECT_ROOT,
    PROJECT_ROOT / "main_mujoco",
    PROJECT_ROOT / "comp_vision_sim",
    PROJECT_ROOT / "Hand_and_Wrists",
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


@dataclass(frozen=True)
class ChoreIntent:
    action: str
    arguments: dict[str, str]


def chore_intent(text: str, target: str | None) -> ChoreIntent:
    """Map an utterance to the high-level actions the merged chore API accepts."""
    words = " ".join(str(text).lower().split())
    if re.search(r"\b(tidy|clean up|clear)\b", words):
        surface = "floor" if "floor" in words else "coffee_table"
        into = "basket"
        return ChoreIntent("tidy", {"surface": surface, "into": into})

    from integration.pick_adapter import resolve_object

    item = resolve_object(target) or resolve_object(words)
    if item is None:
        raise ValueError(
            f"I understood the request, but {target or text!r} is not one of the "
            "objects the current grasp system knows how to hold."
        )

    if re.search(r"\b(bring|fetch|deliver|hand)\b", words):
        return ChoreIntent("fetch", {"item": item, "to": "person"})
    if re.search(r"\b(put|place|set|drop)\b", words):
        if "basket" in words or "bin" in words:
            destination = "basket"
        elif "side table" in words:
            destination = "side_table"
        elif "person" in words or "hand" in words or "me" in words:
            destination = "person"
        else:
            destination = "coffee_table"
        return ChoreIntent("put", {"item": item, "to": destination})
    if re.search(r"\b(pick|grab|take|lift|get)\b", words):
        return ChoreIntent("pick", {"item": item})
    raise ValueError(
        "The current chore system supports pick up, bring/fetch, put/place, and tidy."
    )


class AutonomousTaskManager:
    """Prepare and expose one existing chore controller at a time."""

    ACTIVE_STATES = {"processing", "executing"}

    def __init__(self, adapter, *, llm_url: str, llm_model: str):
        self.adapter = adapter
        self.llm_url = llm_url
        self.llm_model = llm_model
        self._lock = threading.RLock()
        self._generation = 0
        self._controller = None
        self._state = "idle"
        self._command = ""
        self._message = "Waiting for a command."
        self._reply = ""
        self._action: str | None = None
        self._source: str | None = None
        self._llm_error: str | None = None
        self._results: list = []

    def start(self, command: str) -> dict[str, object]:
        command = " ".join(str(command).split())
        if not command:
            raise ValueError("Say or type a task first.")
        with self._lock:
            if self._state in self.ACTIVE_STATES:
                raise RuntimeError("A task is already active. Cancel it before starting another.")
            self._generation += 1
            generation = self._generation
            self._controller = None
            self._state = "processing"
            self._command = command
            self._message = "Understanding your request..."
            self._reply = ""
            self._action = None
            self._source = None
            self._llm_error = None
            self._results = []
        threading.Thread(
            target=self._prepare,
            args=(generation, command),
            name="butlerbot-task-prepare",
            daemon=True,
        ).start()
        return self.snapshot()

    def _prepare(self, generation: int, command: str):
        try:
            search = self._interpret(command)
            intent = chore_intent(command, search.target)
            controller = self.adapter.make_chore(intent.action, **intent.arguments)
            with self._lock:
                if generation != self._generation:
                    return
                self._controller = controller
                self._state = "executing"
                self._action = intent.action
                self._source = search.source
                self._llm_error = search.error
                self._reply = search.reply
                self._message = controller.status
        except Exception as error:
            with self._lock:
                if generation != self._generation:
                    return
                self._controller = None
                self._state = "failed"
                self._message = str(error)

    def _interpret(self, command: str):
        from vision_sim.llm_command import CommandInterpreter, fallback_task

        try:
            return CommandInterpreter(
                base_url=self.llm_url,
                model=self.llm_model,
                thinking=False,
                verbose=False,
            ).parse(command)
        except Exception as error:
            # This is the teammate-provided fallback path.  Its error is kept
            # in status so the phone never implies the local model answered.
            return fallback_task(command, f"{type(error).__name__}: {error}")

    def controller(self):
        with self._lock:
            return self._controller if self._state == "executing" else None

    def after_step(self, controller):
        with self._lock:
            if controller is not self._controller or self._state != "executing":
                return
            self._message = controller.status
            if controller.done:
                self._results = list(controller.results)
                self._state = "complete" if controller.succeeded else "failed"
                self._message = controller.status
                self._controller = None

    def fail(self, error: Exception | str):
        with self._lock:
            if self._state not in self.ACTIVE_STATES:
                return
            self._generation += 1
            self._controller = None
            self._state = "failed"
            self._message = f"Task execution failed: {error}"

    def cancel(self, reason: str = "Cancelled by operator") -> bool:
        with self._lock:
            was_active = self._state in self.ACTIVE_STATES
            if not was_active:
                return False
            controller = self._controller
            self._generation += 1
            self._controller = None
            self._state = "cancelled"
            self._message = reason
        self.adapter.cancel_chore(controller)
        return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "state": self._state,
                "active": self._state in self.ACTIVE_STATES,
                "command": self._command,
                "message": self._message,
                "reply": self._reply,
                "action": self._action,
                "source": self._source,
                "llm_error": self._llm_error,
                "results": [list(item) for item in self._results],
            }
