"""Text to speech for the robot's replies: Piper, an open-source neural voice, run locally.

When a task ends the page speaks the robot's last line -- the model's own account
of what it did. The audio is synthesized on this computer and served as WAV from
`/api/speech`, so it plays on whichever device the remote is open on.

Voices live in ~/.local/share/piper-voices. Download the default once with:

    python -m piper.download_voices en_US-lessac-medium --data-dir ~/.local/share/piper-voices

Without Piper or its voice, espeak-ng is used; without either, the endpoint says so.
"""

from __future__ import annotations

from collections import OrderedDict
import importlib.util
import io
from pathlib import Path
import shutil
import subprocess
import threading
import wave

DEFAULT_VOICE = "en_US-lessac-medium"
VOICE_DIR = Path.home() / ".local" / "share" / "piper-voices"
MAX_CHARS = 600


class RobotVoice:
    """Synthesize short replies to WAV; the last few are cached (a reply is spoken once per viewer)."""

    def __init__(self, voice: str = DEFAULT_VOICE, voice_dir: str | Path = VOICE_DIR,
                 cache_size: int = 16):
        self.voice_name = voice
        self.voice_dir = Path(voice_dir)
        self.cache_size = cache_size
        self.backend: str | None = None
        self._voice = None
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._lock = threading.Lock()
        self._error: str | None = None

    @property
    def model_path(self) -> Path:
        return self.voice_dir / f"{self.voice_name}.onnx"

    def piper_ready(self) -> bool:
        return importlib.util.find_spec("piper") is not None and self.model_path.exists()

    def synthesize(self, text: str) -> bytes:
        text = " ".join(str(text).split())[:MAX_CHARS]
        if not text:
            raise ValueError("Nothing to say.")
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]
            audio = self._synthesize(text)
            self._cache[text] = audio
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return audio

    def _piper(self):
        if self._voice is None:
            from piper import PiperVoice
            self._voice = PiperVoice.load(self.model_path)
        return self._voice

    def _synthesize(self, text: str) -> bytes:
        errors = []
        if self.piper_ready():
            try:
                buffer = io.BytesIO()
                with wave.open(buffer, "wb") as wav:
                    self._piper().synthesize_wav(text, wav)
                self.backend, self._error = "piper", None
                return buffer.getvalue()
            except Exception as error:
                errors.append(f"piper: {type(error).__name__}: {error}")
        elif importlib.util.find_spec("piper") is None:
            errors.append("piper-tts is not installed")
        else:
            errors.append(f"no Piper voice at {self.model_path}")
        espeak = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak:
            try:
                audio = subprocess.run([espeak, "--stdout", text], capture_output=True,
                                       check=True, timeout=30).stdout
                self.backend, self._error = "espeak-ng", "; ".join(errors) or None
                return audio
            except (OSError, subprocess.SubprocessError) as error:
                errors.append(f"espeak-ng: {error}")
        self._error = "; ".join(errors)
        raise RuntimeError(f"No text-to-speech is available ({self._error}).")

    def status(self) -> dict[str, object]:
        return {
            "backend": self.backend or ("piper" if self.piper_ready() else
                                        "espeak-ng" if shutil.which("espeak-ng") else None),
            "voice": self.voice_name,
            "piper_ready": self.piper_ready(),
            "error": self._error,
        }
