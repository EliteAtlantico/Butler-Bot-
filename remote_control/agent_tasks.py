"""Hand typed or spoken requests to the tool-calling LLM agent (`robot_agent`).

`autonomy.AutonomousTaskManager` maps a request onto one of four fixed chores
for seven known items. This runs the same request through `robot_agent`
instead: the local LLM reasons step by step -- look around, go near, measure,
plan a grasp, pick up, put down -- with tools that drive the remote's own robot,
and can do things no chore was written for.

Only one thing may step the physics. While a request runs, the agent's tools
step the robot on the agent's thread, holding the adapter's lock for each
0.1 s chunk so camera renders and the viewer never see a half-stepped state,
paced to the wall clock so the phone and the viewer show it in real time. The
remote's control loop stands aside meanwhile (`owns_robot()`). Cancel, the
e-stop, the joystick and a fall interrupt it within one chunk; the robot holds
position and manual control resumes.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_agent.tools import Interrupted  # noqa: E402


def _llm_failure(error: BaseException) -> str | None:
    """The error text if `error` means the local LLM could not answer, else None."""
    text = f"{type(error).__name__}: {error}"
    try:
        import requests
        if isinstance(error, requests.RequestException):
            return text
    except ImportError:
        pass
    if isinstance(error, RuntimeError) and str(error).startswith("LLM HTTP"):
        return text
    return None


def _close_dangling_calls(agent, why: str):
    """Answer tool calls an interruption left unanswered.

    The next request goes into the same conversation, and an assistant turn
    whose tool calls have no results is rejected by the chat template.
    """
    messages = getattr(agent, "messages", None)
    if not messages:
        return
    answered = {m.get("tool_call_id") for m in messages if m.get("role") == "tool"}
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                if call.get("id") not in answered:
                    messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                     "name": call.get("function", {}).get("name", ""),
                                     "content": json.dumps({"ok": False, "error": why})})
            return


def _arguments_brief(arguments, limit: int = 80) -> str:
    try:
        value = json.loads(arguments) if isinstance(arguments, str) else dict(arguments or {})
        text = ", ".join(f"{k}={v}" for k, v in value.items())
    except (TypeError, ValueError):
        text = str(arguments)
    text = text if len(text) <= limit else text[:limit - 3] + "..."
    return f"({text})"


class _ObservedTools:
    """The agent's view of the tools: every call is reported, and refused once cancelled."""

    def __init__(self, tools, manager: "AgentTaskManager"):
        self.tools = tools
        self.manager = manager

    def world_summary(self) -> str:
        return self.tools.world_summary()

    def specs(self) -> list[dict]:
        return self.tools.specs()

    def call(self, name, arguments) -> dict:
        if self.manager._cancel.is_set():
            raise Interrupted("cancelled by the operator")
        self.manager._tool_started(name, arguments)
        result = self.tools.call(name, arguments)
        self.manager._tool_finished(name, result)
        return result


class AgentTaskManager:
    """One LLM-agent request at a time, in the shape the remote's runtime expects."""

    ACTIVE_STATES = {"processing", "executing"}

    def __init__(self, adapter, *, llm_url: str, llm_model: str, scene=None,
                 realtime: bool = True, max_steps: int = 40,
                 tools_factory=None, agent_factory=None):
        self.adapter = adapter
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.scene = scene
        self.realtime = realtime
        self.max_steps = max_steps
        self._tools_factory = tools_factory
        self._agent_factory = agent_factory
        self._lock = threading.RLock()
        self._generation = 0
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._agent = None
        self._pace = None
        self._state = "idle"
        self._command = ""
        self._message = "Waiting for a command."
        self._reply = ""
        self._action: str | None = None
        self._llm_error: str | None = None
        self._results: list = []

    # ------------------------------------------------ the runtime's interface
    def controller(self):
        """No controller for the control loop to step: the agent steps the robot."""
        return None

    def after_step(self, controller):
        return None

    def owns_robot(self) -> bool:
        with self._lock:
            return self._state in self.ACTIVE_STATES

    def start(self, command: str) -> dict[str, object]:
        command = " ".join(str(command).split())
        if not command:
            raise ValueError("Say or type a task first.")
        with self._lock:
            if self._state in self.ACTIVE_STATES:
                raise RuntimeError("A task is already active. Cancel it before starting another.")
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("The last request is still stopping; try again in a moment.")
            self._generation += 1
            generation = self._generation
            self._cancel = threading.Event()
            self._state = "processing"
            self._command = command
            self._message = "Thinking about your request..."
            self._reply = ""
            self._action = None
            self._llm_error = None
            self._results = []
            self._thread = threading.Thread(target=self._run, args=(generation, command),
                                            name="butlerbot-agent", daemon=True)
            thread = self._thread
        thread.start()
        return self.snapshot()

    def fail(self, error: Exception | str):
        with self._lock:
            if self._state not in self.ACTIVE_STATES:
                return
            self._generation += 1
            self._cancel.set()
            self._state = "failed"
            self._message = f"Task execution failed: {error}"

    def cancel(self, reason: str = "Cancelled by operator") -> bool:
        with self._lock:
            if self._state not in self.ACTIVE_STATES:
                return False
            self._generation += 1
            self._cancel.set()
            self._state = "cancelled"
            self._message = reason
        self.adapter.cancel_chore(None)
        return True

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "brain": "agent",
                "state": self._state,
                "active": self._state in self.ACTIVE_STATES,
                "command": self._command,
                "message": self._message,
                "reply": self._reply,
                "action": self._action,
                "source": "model" if self._results or self._reply else None,
                "llm_error": self._llm_error,
                "results": [list(item) for item in self._results],
            }

    # ----------------------------------------------------------- the agent
    def _build_agent(self):
        if self._tools_factory is not None:
            tools = self._tools_factory()
        else:
            from robot_agent.tools import RobotTools
            with self.adapter.lock:
                tools = RobotTools(bot=self.adapter.bot, scene=self.scene,
                                   llm_url=self.llm_url, llm_model=self.llm_model,
                                   verbose=False, step_lock=self.adapter.lock)
        tools.on_step = self._on_step
        observed = _ObservedTools(tools, self)
        if self._agent_factory is not None:
            return self._agent_factory(observed)
        from robot_agent.agent import RobotAgent
        return RobotAgent(observed, base_url=self.llm_url, model=self.llm_model,
                          max_steps=self.max_steps, verbose=False)

    def _run(self, generation: int, command: str):
        try:
            if self._agent is None:
                self._agent = self._build_agent()
            reply = self._agent.run(command)
            with self._lock:
                if generation != self._generation:
                    return
                self._state = "complete"
                self._reply = reply or ""
                self._message = reply or "Done."
        except Interrupted:
            _close_dangling_calls(self._agent, "cancelled by the operator")
            with self._lock:
                if generation == self._generation and self._state in self.ACTIVE_STATES:
                    self._state = "cancelled"
                    self._message = "Cancelled."
        except Exception as error:
            _close_dangling_calls(self._agent, f"{type(error).__name__}: {error}")
            llm = _llm_failure(error)
            with self._lock:
                if generation != self._generation:
                    return
                self._state = "failed"
                self._llm_error = llm
                self._message = (f"The local LLM could not be reached: {llm}" if llm
                                 else f"Task failed: {type(error).__name__}: {error}")
            self.adapter.hold_position()

    def _on_step(self, bot):
        """Called by the tools after every physics chunk: cancel point, and real-time pacing."""
        if self._cancel.is_set():
            raise Interrupted("cancelled by the operator")
        if not self.realtime:
            return
        now = time.monotonic()
        if self._pace is None:
            self._pace = (now, bot.time)
            return
        wall0, sim0 = self._pace
        lag = (bot.time - sim0) - (now - wall0)
        if lag > 0:
            if self._cancel.wait(min(lag, 0.5)):
                raise Interrupted("cancelled by the operator")
        elif lag < -0.5:                       # rendering fell behind: don't sprint to catch up
            self._pace = (now, bot.time)

    def _tool_started(self, name, arguments):
        self._pace = None                      # the model's thinking time is not sim time
        with self._lock:
            self._state = "executing"
            self._action = name
            self._message = f"{name}{_arguments_brief(arguments)}"

    def _tool_finished(self, name, result: dict):
        ok = result.get("ok") is not False
        why = None if ok else (result.get("failure") or result.get("error") or result.get("status"))
        with self._lock:
            self._results.append((name, ok, why))
            self._message = f"{name}: " + ("done" if ok else f"failed ({why})")
