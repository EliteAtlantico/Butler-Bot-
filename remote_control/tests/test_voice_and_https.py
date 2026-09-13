"""The robot's spoken replies, the speech endpoint, and finding the HTTPS address phones need."""

from __future__ import annotations

import io
import json
from pathlib import Path
import shutil
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest import mock
from urllib.error import HTTPError
from urllib.request import urlopen
import wave

from remote_control import server as server_module
from remote_control.server import RemoteServer, tailscale_https_url
from remote_control.tts import RobotVoice

STATIC = Path(__file__).parents[1] / "static"


class FakePiper:
    def __init__(self):
        self.said = []

    def synthesize_wav(self, text, wav_file):
        self.said.append(text)
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(b"\x00\x00" * 2205)


class RobotVoiceTests(unittest.TestCase):
    def voice_with_fake_piper(self):
        voice = RobotVoice()
        voice.piper_ready = lambda: True
        voice._voice = FakePiper()
        return voice

    def test_piper_makes_a_wav_and_repeats_come_from_the_cache(self):
        voice = self.voice_with_fake_piper()
        audio = voice.synthesize("  I put the mug\\n in the basket. ")
        with wave.open(io.BytesIO(audio)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getframerate(), wav.getnframes()), (1, 22050, 2205))
        self.assertEqual(voice.synthesize("I put the mug\\n in the basket."), audio)
        self.assertEqual(len(voice._voice.said), 1)
        self.assertEqual(voice.status()["backend"], "piper")

    def test_nothing_to_say_is_refused(self):
        with self.assertRaises(ValueError):
            self.voice_with_fake_piper().synthesize("   ")

    @unittest.skipUnless(shutil.which("espeak-ng"), "espeak-ng is not installed")
    def test_without_a_piper_voice_espeak_speaks_instead(self):
        voice = RobotVoice(voice_dir="/nonexistent/voices")
        audio = voice.synthesize("Done.")
        self.assertEqual(audio[:4], b"RIFF")
        self.assertEqual(voice.backend, "espeak-ng")
        self.assertIn("no Piper voice", voice.status()["error"])

    def test_no_engine_at_all_says_so(self):
        voice = RobotVoice(voice_dir="/nonexistent/voices")
        with mock.patch("shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "No text-to-speech"):
                voice.synthesize("Done.")


class SpeechEndpointTests(unittest.TestCase):
    def serve(self, voice):
        server = RemoteServer(("127.0.0.1", 0), SimpleNamespace(voice=voice))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close(), thread.join(timeout=2)))
        return f"http://127.0.0.1:{server.server_address[1]}"

    def test_the_page_gets_the_spoken_line_as_wav(self):
        class Voice:
            def synthesize(self, text):
                self.text = text
                return b"RIFF-fake-wav"

        voice = Voice()
        base = self.serve(voice)
        with urlopen(f"{base}/api/speech?text=I%20picked%20up%20the%20mug.", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "audio/wav")
            self.assertEqual(response.read(), b"RIFF-fake-wav")
        self.assertEqual(voice.text, "I picked up the mug.")

    def test_empty_text_and_missing_engines_are_errors_not_audio(self):
        class Voice:
            def synthesize(self, text):
                if not text:
                    raise ValueError("Nothing to say.")
                raise RuntimeError("No text-to-speech is available.")

        base = self.serve(Voice())
        with self.assertRaises(HTTPError) as empty:
            urlopen(f"{base}/api/speech?text=", timeout=2)
        self.assertEqual(empty.exception.code, 400)
        with self.assertRaises(HTTPError) as missing:
            urlopen(f"{base}/api/speech?text=hi", timeout=2)
        self.assertEqual(missing.exception.code, 503)


class TailscaleHttpsTests(unittest.TestCase):
    SERVE_STATUS = {"TCP": {"443": {"HTTPS": True}, "10000": {"HTTPS": True}},
                    "Web": {"host.tail.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:9080"}}},
                            "host.tail.ts.net:10000": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8000"}}}}}

    def status_run(self, payload):
        return mock.patch.object(subprocess, "run", return_value=SimpleNamespace(stdout=json.dumps(payload)))

    def test_finds_the_https_address_that_forwards_to_the_remote(self):
        with mock.patch("shutil.which", return_value="/usr/bin/tailscale"), self.status_run(self.SERVE_STATUS):
            self.assertEqual(tailscale_https_url(8000), "https://host.tail.ts.net:10000")
            self.assertEqual(tailscale_https_url(9080), "https://host.tail.ts.net")
            self.assertIsNone(tailscale_https_url(1234))

    def test_no_tailscale_or_unreadable_status_means_no_address(self):
        with mock.patch("shutil.which", return_value=None):
            self.assertIsNone(tailscale_https_url(8000))
        with mock.patch("shutil.which", return_value="/usr/bin/tailscale"), \
                mock.patch.object(subprocess, "run", return_value=SimpleNamespace(stdout="not json")):
            self.assertIsNone(tailscale_https_url(8000))

    def test_status_reports_the_address_to_the_page(self):
        self.assertIn('"secure_url": self.secure_url', Path(server_module.__file__).read_text())


class PageContractTests(unittest.TestCase):
    def test_page_speaks_each_tasks_final_line_once(self):
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("function announceTaskEnd(task)", script)
        self.assertIn("/api/speech?text=", script)
        self.assertIn('["complete", "failed"].includes(state)', script)
        self.assertIn("announceTaskEnd(task);", script)

    def test_insecure_pages_explain_https_instead_of_blaming_the_browser(self):
        script = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn("NEEDS HTTPS", script)
        self.assertIn("secureUrl = status.secure_url", script)
        self.assertNotIn("Use Safari 14.1+, Chrome, Edge", script)


if __name__ == "__main__":
    unittest.main()
