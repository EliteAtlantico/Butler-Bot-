"""Spoken commands: microphone -> Whisper -> the text the robot's LLM reads.

    stt = SpeechToText()                  # small.en, GPU if it works, else CPU
    text = listen(stt)                    # press Enter, speak, press Enter
    text = stt.transcribe("find_key.wav") # or any audio file

Transcription runs locally with faster-whisper, the CTranslate2 port of
OpenAI's open-source Whisper models. The weights are fetched from Hugging Face
on first use and cached, so after that it works offline. The microphone is read
through sounddevice (PortAudio). Both are optional installs
(`requirements-voice.txt`) and are imported only when voice is actually used,
so the rest of the package never needs them.

The transcript is handed to `llm_command.CommandInterpreter` exactly as typed
text would be: voice is only another way to produce the command string.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

SAMPLE_RATE = 16000                       # what Whisper expects
DEFAULT_MODEL = "small.en"                # ~0.5 GB; "base.en" is faster, "medium.en" better
MIN_SECONDS = 0.3                         # shorter than this is a key bounce, not speech

INSTALL_HINT = "pip install -r comp_vision_sim/requirements-voice.txt"


class SpeechToText:
    """A lazily loaded faster-whisper model.

    device="auto" tries CUDA and falls back to the CPU if the GPU libraries are
    missing or the card is busy -- the LLM server usually has most of the VRAM.
    """

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 compute_type: str = "default", language: str | None = "en",
                 beam_size: int = 5, verbose: bool = False):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        # The ".en" checkpoints only speak English; a multilingual model detects
        # the language itself when this is None.
        self.language = language
        self.beam_size = beam_size
        self.verbose = verbose
        self._model = None

    def load(self):
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as e:
                raise RuntimeError(f"voice input needs faster-whisper: {INSTALL_HINT}") from e
            try:
                self._model = WhisperModel(self.model_name, device=self.device,
                                           compute_type=self.compute_type)
            except Exception as e:  # no CUDA runtime, out of memory, ...
                if self.device == "cpu":
                    raise
                self._use_cpu(e)
            if self.verbose:
                print(f"    [voice] whisper {self.model_name} on {self.device}")
        return self._model

    def _use_cpu(self, error):
        from faster_whisper import WhisperModel
        if self.verbose:
            print(f"    [voice] {self.device} unavailable ({type(error).__name__}: {error}); "
                  "using the CPU")
        self.device, self.compute_type = "cpu", "int8"
        self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio) -> str:
        """Text of a float32 16 kHz mono array, or of an audio file path."""
        if isinstance(audio, np.ndarray) and audio.size < SAMPLE_RATE * MIN_SECONDS:
            return ""
        model = self.load()
        t0 = time.time()
        try:
            text = self._run(model, audio)
        except RuntimeError as e:
            # CUDA kernels are only exercised once decoding starts, so a GPU
            # that loaded fine can still fail here.
            if self.device == "cpu":
                raise
            self._use_cpu(e)
            text = self._run(self._model, audio)
        if self.verbose:
            print(f"    [voice] transcribed in {time.time() - t0:.1f}s: {text!r}")
        return text

    def _run(self, model, audio) -> str:
        segments, _info = model.transcribe(audio, language=self.language,
                                           beam_size=self.beam_size, vad_filter=True,
                                           condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segments).strip()

    def __repr__(self):
        return f"<SpeechToText {self.model_name} on {self.device}>"


def resample(audio: np.ndarray, rate: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """Linear resampling; plenty for speech, and needs no scipy."""
    audio = np.asarray(audio, np.float32).reshape(-1)
    if rate == target or audio.size == 0:
        return audio
    n = int(round(audio.size * target / rate))
    return np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size),
                     audio).astype(np.float32)


def wait_for_enter(timeout: float) -> bool:
    """Block until Enter is pressed or `timeout` passes; True if it was Enter.

    Polls instead of parking a thread on input(): a reader left behind after a
    timeout would swallow the next line typed at the "What should I find?" prompt.
    """
    deadline = time.monotonic() + timeout
    if os.name == "nt":
        import msvcrt
        while time.monotonic() < deadline:
            if msvcrt.kbhit() and msvcrt.getwch() in "\r\n":
                return True
            time.sleep(0.02)
        return False
    import select
    while True:
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        ready, _, _ = select.select([sys.stdin], [], [], left)
        if ready:
            if sys.stdin.readline():
                return True
            time.sleep(max(0.0, deadline - time.monotonic()))   # EOF: just time out
            return False


def input_devices(sd, device=None) -> list:
    """Microphones to try, in order.

    An explicit choice is the only one tried. Otherwise the system default comes
    first, then inputs behind a sound server (JACK / PulseAudio, both of which
    PipeWire provides), then ALSA's virtual devices, and raw hardware last since
    the sound server usually holds it open. On PipeWire desktops the ALSA default
    often fails to open while the JACK route works fine.
    """
    if device is not None:
        return [device]
    apis = [a["name"] for a in sd.query_hostapis()]

    def rank(d):
        api = apis[d["hostapi"]]
        if "JACK" in api:
            return 0
        if "PulseAudio" in api:
            return 1
        return 3 if "(hw:" in d["name"] else 2

    found = sorted((rank(d), d["index"]) for d in sd.query_devices()
                   if d["max_input_channels"] > 0
                   and not any(w in d["name"].lower() for w in ("monitor", "sink-")))
    return [None] + [index for _, index in found]


def record(max_seconds: float = 10.0, until=wait_for_enter, device=None) -> np.ndarray:
    """Record the microphone until `until(max_seconds)` returns; 16 kHz mono float32."""
    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:   # OSError: PortAudio itself is missing
        raise RuntimeError(f"voice input needs sounddevice and PortAudio: {INSTALL_HINT}") from e
    chunks = []

    def callback(indata, _frames, _time, _status):
        chunks.append(indata[:, 0].copy())

    error = None
    for n, dev in enumerate(input_devices(sd, device)):
        try:
            info = sd.query_devices(dev, "input")
            # Record at the device's own rate: many mics refuse 16 kHz outright.
            rate = int(info["default_samplerate"])
            stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32",
                                    device=dev, callback=callback)
        except (sd.PortAudioError, ValueError) as e:
            error = error or e
            continue
        if n:
            print(f"  microphone: {info['name']} (the default input would not open)")
        with stream:
            until(max_seconds)
        break
    else:
        raise RuntimeError(f"no microphone could be opened: {error}")
    audio = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
    return resample(audio, rate)


def listen(stt: SpeechToText, max_seconds: float = 10.0, ask=input, device=None,
           recorder=record) -> str:
    """Push-to-talk: Enter starts, Enter (or `max_seconds`) stops; returns the text."""
    stt.load()                            # before speaking, not after: first load can download
    ask("Press Enter, then say what to find (Enter again to stop) ")
    print(f"  listening... (up to {max_seconds:g} s)")
    audio = recorder(max_seconds, device=device)
    return stt.transcribe(audio)
