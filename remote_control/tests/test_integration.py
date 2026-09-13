from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import threading
import time
import unittest
from urllib.request import Request, urlopen

from remote_control.autonomy import AutonomousTaskManager, ChoreIntent, chore_intent
from remote_control.robot_adapter import USER_CAMERAS
from remote_control.server import RemoteServer, RobotRuntime
from remote_control.speech import LocalSpeechToText


class FakeChore:
    def __init__(self):
        self.status = "pick up the bottle: starting"
        self.done = False
        self.succeeded = False
        self.results = []


class FakeAdapter:
    def __init__(self):
        self.made = []
        self.holds = 0
        self.stops = 0
        self.cancelled = []

    def make_chore(self, action, **arguments):
        self.made.append((action, arguments))
        return FakeChore()

    def hold_position(self):
        self.holds += 1

    def stop(self):
        self.stops += 1

    def cancel_chore(self, controller=None):
        self.cancelled.append(controller)
        self.hold_position()


class IntentTests(unittest.TestCase):
    def test_high_level_actions_use_existing_chore_shape(self):
        self.assertEqual(chore_intent("Pick up the red cup", "red cup"),
                         ChoreIntent(action="pick", arguments={"item": "mug"}))
        self.assertEqual(chore_intent("Bring me the water bottle", "water bottle"),
                         ChoreIntent(action="fetch",
                                    arguments={"item": "bottle", "to": "person"}))
        self.assertEqual(chore_intent("Put the bottle on the table", "bottle"),
                         ChoreIntent(action="put",
                                    arguments={"item": "bottle", "to": "coffee_table"}))
        self.assertEqual(chore_intent("Tidy up the table", None),
                         ChoreIntent(action="tidy",
                                    arguments={"surface": "coffee_table",
                                               "into": "basket"}))

    def test_unknown_grasp_object_fails_honestly(self):
        with self.assertRaisesRegex(ValueError, "grasp system"):
            chore_intent("Pick up the toys", "toys")


class AutonomousTaskTests(unittest.TestCase):
    def test_local_llm_result_becomes_existing_chore_controller(self):
        adapter = FakeAdapter()
        manager = AutonomousTaskManager(adapter, llm_url="local", llm_model="model")
        manager._interpret = lambda _text: SimpleNamespace(
            target="water bottle", source="model", error=None,
            reply="I'll bring the water bottle.")

        manager.start("Bring me the water bottle")
        deadline = time.monotonic() + 1.0
        while manager.snapshot()["state"] == "processing" and time.monotonic() < deadline:
            time.sleep(0.005)

        snapshot = manager.snapshot()
        self.assertEqual(snapshot["state"], "executing")
        self.assertEqual(snapshot["source"], "model")
        self.assertEqual(adapter.made,
                         [("fetch", {"item": "bottle", "to": "person"})])

        controller = manager.controller()
        controller.done = True
        controller.succeeded = True
        controller.status = "fetch the bottle: done"
        controller.results = [("pick up the bottle", True, None)]
        manager.after_step(controller)
        self.assertEqual(manager.snapshot()["state"], "complete")

    def test_cancel_invalidates_in_flight_preparation(self):
        adapter = FakeAdapter()
        manager = AutonomousTaskManager(adapter, llm_url="local", llm_model="model")
        manager._interpret = lambda _text: (time.sleep(0.05) or SimpleNamespace(
            target="mug", source="model", error=None, reply="Working."))
        manager.start("Pick up the mug")
        self.assertTrue(manager.cancel("Manual takeover"))
        time.sleep(0.08)
        self.assertEqual(manager.snapshot()["state"], "cancelled")
        self.assertIsNone(manager.controller())
        self.assertEqual(len(adapter.cancelled), 1)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FakeAdapter()
        self.runtime = RobotRuntime(self.adapter)
        self.runtime.tasks = SimpleNamespace(
            cancel=lambda reason: True,
            snapshot=lambda: {},
        )

    def test_manual_drive_cancels_autonomy_before_commanding_motion(self):
        self.runtime.drive(0.4, -0.2)
        self.assertGreaterEqual(self.adapter.holds, 1)
        motion = self.runtime.control.motion()
        self.assertGreater(motion.linear_mps, 0)

    def test_emergency_stop_cancels_and_freezes_all_actuators(self):
        self.runtime.emergency_stop()
        self.assertTrue(self.runtime.control.snapshot()["emergency_stop"])
        self.assertEqual(self.adapter.holds, 1)


class CameraAndUiContractTests(unittest.TestCase):
    def test_consumer_camera_contract_is_exactly_four_existing_model_poses(self):
        self.assertEqual([camera["id"] for camera in USER_CAMERAS],
                         ["scene", "head-left", "head-right", "wrist-left", "wrist-right"])
        self.assertEqual([camera["rgbd"] for camera in USER_CAMERAS],
                         [False, True, True, False, False])

    def test_mobile_page_has_four_camera_buttons_and_hides_manual_controls(self):
        page = (Path(__file__).parents[1] / "static" / "index.html").read_text(
            encoding="utf-8")
        self.assertEqual(page.count("data-camera-id="), 5)
        self.assertIn(
            '<details class="panel advanced-panel hidden" hidden aria-hidden="true">',
            page,
        )
        self.assertIn("id=\"voiceButton\"", page)
        self.assertIn("id=\"cancelTaskButton\"", page)

    def test_live_view_is_latest_frame_canvas_with_diagnostics(self):
        static = Path(__file__).parents[1] / "static"
        page = (static / "index.html").read_text(encoding="utf-8")
        script = (static / "app.js").read_text(encoding="utf-8")
        self.assertIn("id=\"cameraCanvas\"", page)
        for diagnostic in ("displayFps", "renderFps", "frameAge",
                           "activeCamera", "droppedFrames"):
            self.assertIn(f'id="{diagnostic}"', page)
        self.assertIn("cameraAbortController?.abort()", script)
        self.assertIn("/api/camera/stream?", script)
        self.assertIn("pendingCameraFrame = frame", script)
        self.assertIn("if (pendingCameraFrame) clientDroppedFrames += 1", script)
        self.assertIn('id="cameraNative"', page)
        self.assertIn("startNativeCameraFallback", script)
        self.assertIn("isIosWebKit", script)
        self.assertIn("Another viewer selected a different camera", script)
        self.assertNotIn("setInterval(refreshCamera", script)

    def test_push_to_talk_handles_mobile_pointer_lifecycle_before_speech_start(self):
        script = (Path(__file__).parents[1] / "static" / "app.js").read_text(
            encoding="utf-8")
        self.assertIn('voiceButton.addEventListener("pointerdown"', script)
        self.assertIn('voiceButton.addEventListener("pointerup"', script)
        self.assertIn('voiceButton.addEventListener("pointercancel"', script)
        self.assertIn('voiceButton.addEventListener("lostpointercapture"', script)
        # Whisper only: the button asks for the microphone before recording starts.
        self.assertLess(script.index('setVoiceUi("ALLOW MICROPHONE", true)'),
                        script.index("navigator.mediaDevices.getUserMedia"))
        self.assertIn("Microphone permission was denied", script)
        self.assertIn("submitTask(transcript)", script)
        self.assertIn("window.MediaRecorder", script)
        self.assertIn('fetch("/api/transcribe"', script)
        self.assertIn("iPhone microphone capture requires HTTPS", script)


class LocalSpeechTests(unittest.TestCase):
    def test_local_transcript_is_independent_of_task_and_llm(self):
        class FakeWhisper:
            def transcribe(self, audio, **kwargs):
                self.audio = audio.read()
                self.options = kwargs
                return iter((SimpleNamespace(text=" Pick up the red cup "),)), None

        speech = LocalSpeechToText("test-model")
        speech._model = FakeWhisper()
        self.assertEqual(speech.transcribe(b"a" * 256), "Pick up the red cup")
        self.assertEqual(speech._model.audio, b"a" * 256)
        self.assertEqual(speech._model.options["language"], "en")

    def test_audio_endpoint_returns_transcript_without_task_or_llm(self):
        class FakeSpeech:
            def transcribe(self, audio):
                self.audio = audio
                return "Turn left"

        speech = FakeSpeech()
        server = RemoteServer(("127.0.0.1", 0), SimpleNamespace(speech=speech))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}/api/transcribe",
                data=b"a" * 256,
                headers={"Content-Type": "audio/mp4"},
                method="POST",
            )
            with urlopen(request, timeout=2) as response:
                payload = json.load(response)
            self.assertEqual(payload, {"ok": True, "transcript": "Turn left"})
            self.assertEqual(speech.audio, b"a" * 256)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
