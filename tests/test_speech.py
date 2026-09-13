"""Voice commands: recording, Whisper transcription and the CLI wiring, offline.

Neither faster-whisper nor a microphone is needed: both are replaced by fakes.
"""
from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import run_navigation as run_nav
from vision_sim import speech
from vision_sim.llm_command import SearchTask

KEYS_TASK = SearchTask(command="find my keys", target="keys", description="a set of keys")


class FakeWhisperModel:
    created = []
    fail_on = set()          # devices whose load raises
    fail_decode_on = set()   # devices whose decoding raises

    def __init__(self, name, device="auto", compute_type="default"):
        if device in self.fail_on:
            raise ValueError(f"no {device}")
        self.name, self.device, self.compute_type = name, device, compute_type
        self.calls = []
        FakeWhisperModel.created.append(self)

    def transcribe(self, audio, **kwargs):
        self.calls.append((audio, kwargs))

        def segments():
            if self.device in self.fail_decode_on:
                raise RuntimeError("CUDA failed with error no kernel image")
            yield SimpleNamespace(text=" Find my")
            yield SimpleNamespace(text=" keys. ")
        return segments(), SimpleNamespace(language="en")


@pytest.fixture
def whisper(monkeypatch):
    FakeWhisperModel.created.clear()
    FakeWhisperModel.fail_on, FakeWhisperModel.fail_decode_on = set(), set()
    module = ModuleType("faster_whisper")
    module.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", module)
    return FakeWhisperModel


def speech_audio(seconds=1.0):
    return np.zeros(int(speech.SAMPLE_RATE * seconds), np.float32)


def test_transcribe_joins_segments_with_english_and_vad(whisper):
    stt = speech.SpeechToText("base.en", device="cpu")
    assert stt.transcribe(speech_audio()) == "Find my keys."
    (model,) = whisper.created
    assert (model.name, model.device) == ("base.en", "cpu")
    _audio, kwargs = model.calls[0]
    assert kwargs["language"] == "en" and kwargs["vad_filter"] is True


def test_too_short_a_recording_is_not_sent_to_the_model(whisper):
    assert speech.SpeechToText().transcribe(speech_audio(0.1)) == ""
    assert whisper.created == []


def test_gpu_load_failure_falls_back_to_cpu(whisper):
    whisper.fail_on = {"auto"}
    stt = speech.SpeechToText()
    assert stt.transcribe(speech_audio()) == "Find my keys."
    assert (stt.device, stt.compute_type) == ("cpu", "int8")


def test_gpu_decode_failure_retries_on_cpu(whisper):
    whisper.fail_decode_on = {"cuda"}
    stt = speech.SpeechToText(device="cuda")
    assert stt.transcribe("clip.wav") == "Find my keys."
    assert [m.device for m in whisper.created] == ["cuda", "cpu"]


def test_cpu_failure_is_not_hidden(whisper):
    whisper.fail_decode_on = {"cpu"}
    with pytest.raises(RuntimeError):
        speech.SpeechToText(device="cpu").transcribe(speech_audio())


def test_missing_package_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    with pytest.raises(RuntimeError, match="requirements-voice.txt"):
        speech.SpeechToText().load()


def test_resample_changes_rate_and_keeps_shape_of_signal():
    t = np.arange(48000) / 48000
    out = speech.resample(np.sin(2 * np.pi * 5 * t), 48000)
    assert out.dtype == np.float32 and out.size == 16000
    assert np.allclose(out, np.sin(2 * np.pi * 5 * np.linspace(0, t[-1], 16000)), atol=1e-3)
    same = np.linspace(-1, 1, speech.SAMPLE_RATE, dtype=np.float32)
    assert np.array_equal(speech.resample(same, speech.SAMPLE_RATE), same)


DEVICES = [  # what a PipeWire desktop looks like to PortAudio
    {"index": 0, "name": "HDA Intel PCH: ALC1220 Analog (hw:1,0)", "hostapi": 0,
     "max_input_channels": 2, "default_samplerate": 44100.0},
    {"index": 1, "name": "default", "hostapi": 0, "max_input_channels": 64,
     "default_samplerate": 44100.0},
    {"index": 2, "name": "Speakers Monitor", "hostapi": 1, "max_input_channels": 2,
     "default_samplerate": 48000.0},
    {"index": 3, "name": "ACE Analog Stereo", "hostapi": 1, "max_input_channels": 2,
     "default_samplerate": 48000.0},
    {"index": 4, "name": "HDMI", "hostapi": 0, "max_input_channels": 0,
     "default_samplerate": 48000.0},
]


def fake_sounddevice(monkeypatch, broken=()):
    """A sounddevice whose default input is index 1 and whose `broken` devices fail to open."""
    streams = []

    class PortAudioError(Exception):
        pass

    class Stream:
        def __init__(self, **kwargs):
            if kwargs["device"] in broken:
                raise PortAudioError("Error opening InputStream")
            self.kwargs = kwargs
            streams.append(self)

        def __enter__(self):
            block = np.ones((4800, 1), np.float32)
            self.kwargs["callback"](block, 4800, None, None)
            self.kwargs["callback"](block, 4800, None, None)
            return self

        def __exit__(self, *exc):
            return False

    def query_devices(device=-1, kind=None):
        if device == -1:
            return DEVICES
        return DEVICES[1 if device is None else device]

    sd = ModuleType("sounddevice")
    sd.PortAudioError = PortAudioError
    sd.InputStream = Stream
    sd.query_devices = query_devices
    sd.query_hostapis = lambda: [{"name": "ALSA"}, {"name": "JACK Audio Connection Kit"}]
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    return sd, streams


def test_record_reads_the_device_rate_until_stopped(monkeypatch):
    _sd, streams = fake_sounddevice(monkeypatch)
    waited = []
    audio = speech.record(3.0, until=waited.append, device=3)
    assert waited == [3.0]
    assert streams[0].kwargs["samplerate"] == 48000 and streams[0].kwargs["device"] == 3
    assert audio.size == 3200 and np.allclose(audio, 1.0)


def test_default_input_first_then_sound_server_then_hardware_never_monitors(monkeypatch):
    sd, _streams = fake_sounddevice(monkeypatch)
    assert speech.input_devices(sd) == [None, 3, 1, 0]
    assert speech.input_devices(sd, "usb") == ["usb"]


def test_record_falls_back_when_the_default_input_will_not_open(monkeypatch, capsys):
    _sd, streams = fake_sounddevice(monkeypatch, broken={None})
    audio = speech.record(1.0, until=lambda s: None)
    assert [s.kwargs["device"] for s in streams] == [3]
    assert audio.size == 3200 and "ACE Analog Stereo" in capsys.readouterr().out


def test_record_reports_when_no_microphone_opens(monkeypatch):
    fake_sounddevice(monkeypatch, broken={None, 0, 1, 3})
    with pytest.raises(RuntimeError, match="no microphone"):
        speech.record(1.0, until=lambda s: None)


def test_listen_loads_first_then_records_and_transcribes(whisper, capsys):
    stt = speech.SpeechToText(device="cpu")
    order = []
    ask = lambda prompt: order.append(("ask", len(whisper.created)))
    recorder = lambda seconds, device=None: order.append(("rec", seconds, device)) or speech_audio()
    assert speech.listen(stt, 4.0, ask=ask, device="usb", recorder=recorder) == "Find my keys."
    assert order == [("ask", 1), ("rec", 4.0, "usb")]      # model loaded before the prompt


# ======================================================================= CLI
def test_voice_flags_make_a_spoken_search():
    a = run_nav.parse_args(["--voice"])
    assert (a.voice, a.command, a.explore) == (True, "", True)
    f = run_nav.parse_args(["--voice-file", "clip.wav"])
    assert (f.voice, f.command) == (True, "")
    assert run_nav.parse_args(["--command", "find a mug", "--voice"]).command == "find a mug"
    plain = run_nav.parse_args([])
    assert (plain.voice, plain.command, plain.whisper_model) == (False, None, "small.en")


def test_spoken_text_is_what_the_llm_reads(monkeypatch):
    a = run_nav.parse_args(["--voice", "--mic", "3"])
    heard = {}

    def fake_listen(stt, seconds, ask=input, device=None):
        heard.update(model=stt.model_name, seconds=seconds, device=device)
        return "find my keys"

    monkeypatch.setattr(speech, "listen", fake_listen)
    parsed = []
    monkeypatch.setattr("vision_sim.llm_command.CommandInterpreter.parse",
                        lambda self, text: parsed.append(text) or KEYS_TASK)
    asked = []
    task = run_nav.build_command(a, ask=asked.append)
    assert parsed == ["find my keys"] and asked == []
    assert heard == {"model": "small.en", "seconds": 10.0, "device": 3}
    assert task is KEYS_TASK and a.target == "keys"


def test_voice_file_is_transcribed_instead_of_the_mic(monkeypatch):
    a = run_nav.parse_args(["--voice-file", "clip.wav", "--whisper-device", "cpu"])
    files = []
    monkeypatch.setattr(speech.SpeechToText, "transcribe",
                        lambda self, audio: files.append((audio, self.device)) or "get the remote")
    monkeypatch.setattr(speech, "listen", lambda *a, **k: pytest.fail("used the microphone"))
    assert run_nav.hear_command(a) == "get the remote"
    assert files == [("clip.wav", "cpu")]


@pytest.mark.parametrize("failure", ["silence", "no microphone"])
def test_voice_failure_falls_back_to_typing(monkeypatch, failure):
    a = run_nav.parse_args(["--voice"])

    def fake_listen(*_a, **_k):
        if failure == "silence":
            return ""
        raise RuntimeError("voice input needs sounddevice")

    monkeypatch.setattr(speech, "listen", fake_listen)
    parsed = []
    monkeypatch.setattr("vision_sim.llm_command.CommandInterpreter.parse",
                        lambda self, text: parsed.append(text) or KEYS_TASK)
    run_nav.build_command(a, ask=lambda prompt: "find my keys")
    assert parsed == ["find my keys"]
