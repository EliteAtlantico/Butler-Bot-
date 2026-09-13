"""Local speech-to-text for audio recorded in the browser: Whisper, on this computer.

The TALK button always records in the browser and posts the audio to
`/api/transcribe`; nothing is sent to a cloud speech service. Transcription is
the repository's shared Whisper wrapper (`comp_vision_sim/vision_sim/speech.py`,
faster-whisper) -- the same model, GPU-with-CPU-fallback and voice-activity
filtering that `run_navigation.py --voice` and `python -m robot_agent --voice`
use -- so a command is heard the same way however it is spoken.
"""

from __future__ import annotations

from io import BytesIO
import importlib.util
from pathlib import Path
import sys
import threading

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "comp_vision_sim") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "comp_vision_sim"))

from vision_sim.speech import SpeechToText  # noqa: E402

DEFAULT_MODEL = "small.en"


class LocalSpeechToText:
    """Whisper for the remote: lazily loaded, one transcription at a time."""

    def __init__(self, model: str = DEFAULT_MODEL, *, device: str = "auto",
                 compute_type: str = "default"):
        self.stt = SpeechToText(model, device=device, compute_type=compute_type,
                                language="en")
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def model_name(self) -> str:
        return self.stt.model_name

    @property
    def device(self) -> str:
        return self.stt.device

    @property
    def _model(self):
        return self.stt._model

    @_model.setter
    def _model(self, model):
        self.stt._model = model

    def preload(self):
        """Load the model ahead of the first TALK (downloads it on first use)."""
        with self._lock:
            try:
                self.stt.load()
                self._error = None
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"

    def transcribe(self, audio: bytes) -> str:
        if len(audio) < 128:
            raise ValueError("The recording was empty or too short to transcribe.")
        with self._lock:
            try:
                # faster-whisper decodes the browser's webm/ogg/mp4 itself (PyAV).
                text = self.stt.transcribe(BytesIO(audio))
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                if "faster-whisper" in str(error):
                    raise RuntimeError(str(error)) from error
                raise RuntimeError(f"Local transcription failed: {error}") from error
            if not text:
                raise ValueError("No speech was detected in the recording.")
            self._error = None
            return text

    def status(self) -> dict[str, object]:
        return {
            "backend": "faster-whisper",
            "installed": importlib.util.find_spec("faster_whisper") is not None,
            "loaded": self._model is not None,
            "model": self.model_name,
            "device": self.device,
            "error": self._error,
        }
