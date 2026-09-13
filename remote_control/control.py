"""Thread-safe command state and fail-safe motion watchdog."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


@dataclass(frozen=True)
class MotionCommand:
    linear_mps: float
    angular_rads: float
    reason: str


class ControlState:
    def __init__(self, *, watchdog_seconds: float = 0.35,
                 max_linear_mps: float = 0.42,
                 max_angular_rads: float = 1.0):
        self.watchdog_seconds = float(watchdog_seconds)
        self.max_linear_mps = float(max_linear_mps)
        self.max_angular_rads = float(max_angular_rads)
        self._lock = threading.RLock()
        self._linear = 0.0
        self._angular = 0.0
        self._last_drive = 0.0
        self._estopped = False
        self._reason = "Ready"
        self._fault: str | None = None

    def drive(self, linear: float, angular: float, *, now: float | None = None):
        with self._lock:
            if self._estopped:
                raise RuntimeError("Emergency stop is engaged.")
            if self._fault:
                raise RuntimeError(self._fault)
            self._linear = _clamp(linear, -1.0, 1.0) * self.max_linear_mps
            self._angular = _clamp(angular, -1.0, 1.0) * self.max_angular_rads
            self._last_drive = time.monotonic() if now is None else float(now)
            self._reason = "Manual drive"

    def stop(self, reason: str = "Stopped"):
        with self._lock:
            self._linear = 0.0
            self._angular = 0.0
            self._last_drive = 0.0
            self._reason = reason

    def emergency_stop(self):
        with self._lock:
            self._estopped = True
            self._linear = 0.0
            self._angular = 0.0
            self._last_drive = 0.0
            self._reason = "EMERGENCY STOP"

    def reset_emergency_stop(self):
        with self._lock:
            self._estopped = False
            self._linear = 0.0
            self._angular = 0.0
            self._last_drive = 0.0
            self._reason = "Emergency stop reset; stopped"

    def fault(self, message: str):
        with self._lock:
            self._fault = str(message)
            self._estopped = True
            self._linear = 0.0
            self._angular = 0.0
            self._reason = "FAULT"

    def motion(self, *, now: float | None = None) -> MotionCommand:
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._estopped or self._fault:
                return MotionCommand(0.0, 0.0, self._reason)
            if self._last_drive and current - self._last_drive > self.watchdog_seconds:
                self._linear = 0.0
                self._angular = 0.0
                self._last_drive = 0.0
                self._reason = "Stopped by command watchdog"
            return MotionCommand(self._linear, self._angular, self._reason)

    def snapshot(self, *, now: float | None = None) -> dict[str, object]:
        motion = self.motion(now=now)
        with self._lock:
            return {
                "linear_mps": motion.linear_mps,
                "angular_rads": motion.angular_rads,
                "reason": motion.reason,
                "emergency_stop": self._estopped,
                "fault": self._fault,
                "watchdog_ms": int(self.watchdog_seconds * 1000),
            }

