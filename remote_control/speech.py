"""Local speech-to-text adapter for browser-recorded audio."""

from __future__ import annotations

from io import BytesIO
import importlib.util
import threading


class LocalSpeechToText:
    """Lazily run faster-whisper locally without touching the task/LLM layer."""

    def __init__(self, model: str = "tiny.en", *, device: str = "cpu",
                 compute_type: str = "int8"):
        self.model_name = model
        self.device = device
        self.compute_type = compute_type
        self._model = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def _load(self):
        if self._model is not None:
            return self._model
        if importlib.util.find_spec("faster_whisper") is None:
            raise RuntimeError(
                "Local speech recognition is not installed. Run: "
                ".\\.venv\\Scripts\\python.exe -m pip install -r "
                "remote_control\\requirements.txt"
            )
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_name, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, audio: bytes) -> str:
        if len(audio) < 128:
            raise ValueError("The recording was empty or too short to transcribe.")
        with self._lock:
            try:
                model = self._load()
                segments, _ = model.transcribe(
                    BytesIO(audio), language="en", beam_size=1,
                    vad_filter=True, condition_on_previous_text=False)
                text = " ".join(segment.text.strip() for segment in segments).strip()
                if not text:
                    raise ValueError("No speech was detected in the recording.")
                self._error = None
                return text
            except (ValueError, RuntimeError):
                raise
            except Exception as error:
                self._error = f"{type(error).__name__}: {error}"
                raise RuntimeError(f"Local transcription failed: {error}") from error

    def status(self) -> dict[str, object]:
        installed = importlib.util.find_spec("faster_whisper") is not None
        return {
            "backend": "faster-whisper",
            "installed": installed,
            "loaded": self._model is not None,
            "model": self.model_name,
            "device": self.device,
            "error": self._error,
        }
