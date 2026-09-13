"""Deterministic natural-language commands for the remote-control MVP."""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ParsedCommand:
    action: str
    payload: dict[str, object]
    message: str


def parse_command(text: str) -> ParsedCommand:
    command = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if not command:
        raise ValueError("Enter a command first.")

    if "reset" in command and ("emergency" in command or "e stop" in command):
        return ParsedCommand("reset_estop", {}, "Emergency stop reset; robot remains stopped.")
    if "emergency" in command or "e stop" in command or "estop" in command:
        return ParsedCommand("estop", {}, "Emergency stop engaged.")
    if command in {"stop", "halt", "freeze"} or "stop moving" in command:
        return ParsedCommand("stop", {}, "Stopped.")

    side = "both"
    if "left" in command:
        side = "left"
    elif "right" in command:
        side = "right"

    if "gripper" in command or "hand" in command:
        if "open" in command or "release" in command:
            return ParsedCommand("gripper", {"side": side, "action": "open"},
                                 f"Opening {side} gripper.")
        if "close" in command or "grab" in command or "grip" in command:
            return ParsedCommand("gripper", {"side": side, "action": "close"},
                                 f"Closing {side} gripper.")
        if "hold" in command or "neutral" in command or "stop" in command:
            return ParsedCommand("gripper", {"side": side, "action": "hold"},
                                 f"Holding {side} gripper.")

    if "forward" in command or "ahead" in command:
        return ParsedCommand("drive", {"linear": 0.55, "angular": 0.0},
                             "Moving forward.")
    if "backward" in command or "backwards" in command or "reverse" in command:
        return ParsedCommand("drive", {"linear": -0.45, "angular": 0.0},
                             "Reversing.")
    if "turn left" in command or command == "left":
        return ParsedCommand("drive", {"linear": 0.0, "angular": 0.6},
                             "Turning left.")
    if "turn right" in command or command == "right":
        return ParsedCommand("drive", {"linear": 0.0, "angular": -0.6},
                             "Turning right.")

    raise ValueError(
        "Command not recognized. Try move forward, turn left, stop, or open gripper."
    )

