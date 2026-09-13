"""The remote's LLM-agent brain and its Whisper voice path, without a sim or a model."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest

from remote_control.agent_tasks import AgentTaskManager
from remote_control.autonomy import AutonomousTaskManager
from remote_control.server import RobotRuntime, parse_args
from remote_control.speech import DEFAULT_MODEL, LocalSpeechToText
from robot_agent.tools import Interrupted

STATIC = Path(__file__).parents[1] / "static"


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class LoopAdapter:
    """The adapter surface the runtime and the agent manager use."""

    def __init__(self):
        self.steps = 0
        self.controller_steps = 0
        self.holds = 0
        self.lock = threading.RLock()
        self.scene = None

    def step(self, duration, linear, angular):
        self.steps += 1

    def step_controller(self, duration, controller):
        self.controller_steps += 1

    def telemetry(self):
        return {"fallen": False}

    def hold_position(self):
        self.holds += 1

    def stop(self):
        pass

    def cancel_chore(self, controller=None):
        self.hold_position()

    def close_control_renderers(self):
        pass

    def close(self):
        pass


class SteppingTools:
    """Steps like RobotTools._run: a physics chunk, then on_step, `chunks` times."""

    def __init__(self, chunks: int = 3, results=None, chunk_pause: float = 0.0):
        self.on_step = None
        self.calls = []
        self.chunks = chunks
        self.chunk_pause = chunk_pause
        self.results = results or {}
        self.bot = SimpleNamespace(time=0.0)

    def world_summary(self):
        return "Scene: test."

    def specs(self):
        return []

    def call(self, name, arguments):
        self.calls.append(name)
        for _ in range(self.chunks):
            self.bot.time += 0.1
            if self.chunk_pause:
                time.sleep(self.chunk_pause)
            self.on_step(self.bot)
        return dict(self.results.get(name, {"ok": True}))


class ScriptedAgent:
    """Calls the tools it is given, in order, as RobotAgent would, then replies."""

    def __init__(self, tools, script, reply="All done."):
        self.tools, self.script, self.reply = tools, script, reply
        self.messages = [{"role": "system", "content": tools.world_summary()}]
        self.requests = []

    def run(self, request):
        self.requests.append(request)
        self.messages.append({"role": "user", "content": request})
        for name in self.script:
            call_id = f"id-{name}"
            self.messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}]})
            result = self.tools.call(name, "{}")
            self.messages.append({"role": "tool", "tool_call_id": call_id, "name": name,
                                  "content": json.dumps(result)})
        return self.reply


def manager_with(adapter, tools, script, reply="All done.", run_error=None):
    agents = []

    def agent_factory(observed):
        agent = ScriptedAgent(observed, script, reply)
        if run_error is not None:
            def fail(_request):
                raise run_error
            agent.run = fail
        agents.append(agent)
        return agent

    manager = AgentTaskManager(adapter, llm_url="local", llm_model="model", realtime=False,
                               tools_factory=lambda: tools, agent_factory=agent_factory)
    return manager, agents


class AgentBrainTests(unittest.TestCase):
    def test_a_request_runs_through_the_agent_and_reports_every_tool(self):
        adapter = LoopAdapter()
        tools = SteppingTools(results={"pick_up": {"ok": False, "failure": "no reachable grasp"}})
        manager, agents = manager_with(adapter, tools, ["look_around", "pick_up"],
                                       reply="I couldn't reach the mug.")
        snapshot = manager.start("  pick   up the mug ")
        # the scripted agent can finish before start() takes its snapshot
        self.assertIn(snapshot["state"], {"processing", "executing", "complete"})
        self.assertTrue(wait_until(lambda: manager.snapshot()["state"] == "complete"))

        done = manager.snapshot()
        self.assertEqual(done["brain"], "agent")
        self.assertEqual(done["command"], "pick up the mug")
        self.assertEqual(done["reply"], "I couldn't reach the mug.")
        self.assertEqual(done["results"], [["look_around", True, None],
                                           ["pick_up", False, "no reachable grasp"]])
        self.assertEqual(done["action"], "pick_up")
        self.assertEqual(agents[0].requests, ["pick up the mug"])
        self.assertFalse(manager.owns_robot())
        self.assertIsNone(manager.controller())          # nothing for the control loop to step

    def test_conversation_continues_across_requests(self):
        manager, agents = manager_with(LoopAdapter(), SteppingTools(), ["get_status"])
        manager.start("pick up the mug")
        self.assertTrue(wait_until(lambda: manager.snapshot()["state"] == "complete"))
        manager.start("now put it in the basket")
        self.assertTrue(wait_until(lambda: manager.snapshot()["state"] == "complete"))
        self.assertEqual(len(agents), 1)
        self.assertEqual(agents[0].requests, ["pick up the mug", "now put it in the basket"])

    def test_cancel_stops_the_agent_between_physics_chunks(self):
        adapter = LoopAdapter()
        tools = SteppingTools(chunks=100000, chunk_pause=0.001)
        manager, agents = manager_with(adapter, tools, ["go_to", "pick_up"])
        manager.start("drive to the kitchen")
        self.assertTrue(wait_until(lambda: tools.calls == ["go_to"] and manager.owns_robot()))

        self.assertTrue(manager.cancel("Manual takeover"))
        self.assertFalse(manager.owns_robot())           # the control loop may step again at once
        self.assertTrue(wait_until(lambda: not manager._thread.is_alive()))
        snapshot = manager.snapshot()
        self.assertEqual(snapshot["state"], "cancelled")
        self.assertEqual(snapshot["message"], "Manual takeover")
        self.assertGreaterEqual(adapter.holds, 1)
        self.assertEqual(tools.calls, ["go_to"])         # pick_up never ran
        last = agents[0].messages[-1]                    # the cut-off call is answered
        self.assertEqual((last["role"], last["tool_call_id"]), ("tool", "id-go_to"))
        self.assertIn("cancelled", json.loads(last["content"])["error"])

    def test_one_request_at_a_time(self):
        tools = SteppingTools(chunks=100000, chunk_pause=0.001)
        manager, _ = manager_with(LoopAdapter(), tools, ["look_around"])
        manager.start("look around")
        self.assertTrue(wait_until(lambda: manager.owns_robot() and tools.calls))
        with self.assertRaisesRegex(RuntimeError, "already active"):
            manager.start("and again")
        manager.cancel()
        with self.assertRaisesRegex(ValueError, "task first"):
            manager.start("   ")

    def test_an_unreachable_llm_fails_the_request_and_says_so(self):
        adapter = LoopAdapter()
        manager, _ = manager_with(adapter, SteppingTools(), [],
                                  run_error=RuntimeError("LLM HTTP 503: model loading"))
        manager.start("tidy up")
        self.assertTrue(wait_until(lambda: manager.snapshot()["state"] == "failed"))
        snapshot = manager.snapshot()
        self.assertIn("503", snapshot["llm_error"])
        self.assertIn("local LLM could not be reached", snapshot["message"])
        self.assertEqual(adapter.holds, 1)

    def test_real_time_pacing_waits_for_the_wall_clock(self):
        manager = AgentTaskManager(LoopAdapter(), llm_url="local", llm_model="model")
        bot = SimpleNamespace(time=0.0)
        manager._on_step(bot)                            # starts the clock
        bot.time = 0.15
        started = time.monotonic()
        manager._on_step(bot)
        self.assertGreaterEqual(time.monotonic() - started, 0.12)
        manager._cancel.set()
        with self.assertRaises(Interrupted):
            manager._on_step(bot)


class RuntimeTests(unittest.TestCase):
    def test_the_control_loop_stands_aside_while_the_agent_drives(self):
        adapter = LoopAdapter()
        runtime = RobotRuntime(adapter, control_period=0.005)
        owns = [True]
        runtime.tasks = SimpleNamespace(owns_robot=lambda: owns[0], controller=lambda: None,
                                        cancel=lambda reason: False, snapshot=lambda: {},
                                        after_step=lambda c: None, fail=lambda e: None)
        runtime.start()
        try:
            time.sleep(0.1)
            self.assertEqual(adapter.steps, 0)
            owns[0] = False
            self.assertTrue(wait_until(lambda: adapter.steps > 0))
        finally:
            runtime.close()

    def test_agent_is_the_default_brain_and_the_chores_remain(self):
        self.assertIsInstance(RobotRuntime(LoopAdapter()).tasks, AgentTaskManager)
        self.assertIsInstance(RobotRuntime(LoopAdapter(), brain="chores").tasks,
                              AutonomousTaskManager)
        with self.assertRaisesRegex(ValueError, "brain"):
            RobotRuntime(LoopAdapter(), brain="psychic")

    def test_the_viewer_opens_and_syncs_only_while_holding_the_physics_lock(self):
        """Opening the viewer runs mj_forward on the shared data; doing it unlocked
        raced the control thread's mj_step and segfaulted in the constraint solver."""
        import sys
        from unittest import mock

        from remote_control import server

        class RecordingLock:
            held = False

            def __enter__(self):
                RecordingLock.held = True

            def __exit__(self, *exc):
                RecordingLock.held = False
                return False

        events = []

        class Handle:
            def __init__(self):
                self.frames = 0

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def is_running(self):
                self.frames += 1
                return self.frames <= 2

            def sync(self):
                events.append(("sync", RecordingLock.held))

        def launch_passive(model, data, **kwargs):
            events.append(("launch", RecordingLock.held))
            return Handle()

        fake_viewer = SimpleNamespace(launch_passive=launch_passive)
        adapter = SimpleNamespace(lock=RecordingLock(), bot=SimpleNamespace(model="m", data="d"))
        import mujoco
        with mock.patch.dict(sys.modules, {"mujoco.viewer": fake_viewer}), \
                mock.patch.object(mujoco, "viewer", fake_viewer, create=True):
            server.run_viewer(adapter, fps=1000.0)
        self.assertEqual(events, [("launch", True), ("sync", True), ("sync", True)])

    def test_server_flags(self):
        args = parse_args([])
        self.assertEqual((args.brain, args.stt_model, args.stt_device, args.viewer),
                         ("agent", "small.en", "auto", False))
        args = parse_args(["--viewer", "--brain", "chores", "--stt-device", "cuda"])
        self.assertEqual((args.brain, args.stt_device, args.viewer), ("chores", "cuda", True))


class WhisperVoiceTests(unittest.TestCase):
    def test_the_remote_uses_the_shared_whisper_wrapper(self):
        from vision_sim.speech import SpeechToText
        speech = LocalSpeechToText()
        self.assertIsInstance(speech.stt, SpeechToText)
        self.assertEqual((DEFAULT_MODEL, speech.model_name, speech.device),
                         ("small.en", "small.en", "auto"))

    def test_browser_audio_goes_to_whisper_as_a_file(self):
        class FakeWhisper:
            def transcribe(self, audio, **options):
                self.audio, self.options = audio, options
                return iter((SimpleNamespace(text=" Tidy up "), SimpleNamespace(text=" the table. "))), None

        speech = LocalSpeechToText("test-model", device="cpu")
        speech._model = FakeWhisper()
        self.assertEqual(speech.transcribe(b"x" * 300), "Tidy up the table.")
        self.assertIsInstance(speech._model.audio, BytesIO)
        self.assertEqual(speech._model.audio.getvalue(), b"x" * 300)
        self.assertEqual(speech._model.options["language"], "en")
        self.assertTrue(speech._model.options["vad_filter"])
        self.assertTrue(speech.status()["loaded"])

    def test_silence_and_undecodable_audio_fail_clearly(self):
        class Silent:
            def transcribe(self, audio, **options):
                return iter(()), None

        class Broken:
            def transcribe(self, audio, **options):
                raise OSError("Invalid data found when processing input")

        speech = LocalSpeechToText("test-model", device="cpu")
        speech._model = Silent()
        with self.assertRaisesRegex(ValueError, "No speech"):
            speech.transcribe(b"x" * 300)
        speech._model = Broken()
        with self.assertRaisesRegex(RuntimeError, "Local transcription failed"):
            speech.transcribe(b"x" * 300)
        self.assertIn("Invalid data", speech.status()["error"])
        with self.assertRaisesRegex(ValueError, "too short"):
            speech.transcribe(b"x")

    def test_the_page_records_for_whisper_instead_of_browser_speech_recognition(self):
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertNotIn("SpeechRecognition", script)
        self.assertNotIn("recognition.start()", script)
        self.assertIn("window.MediaRecorder", script)
        self.assertIn('fetch("/api/transcribe"', script)
        self.assertIn("submitTask(transcript)", script)
        self.assertIn("Whisper", script)


if __name__ == "__main__":
    unittest.main()
