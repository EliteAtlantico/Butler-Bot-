#!/usr/bin/env python3
"""Serve the BracketBot mobile remote on the local network."""

from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

from .command_parser import ParsedCommand, parse_command
from .control import ControlState
from .robot_adapter import DEFAULT_SCENE, SimulationRobotAdapter


STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {
    "/": "index.html",
    "/index.html": "index.html",
    "/styles.css": "styles.css",
    "/app.js": "app.js",
}


class RobotRuntime:
    def __init__(self, adapter: SimulationRobotAdapter,
                 *, watchdog_seconds: float = 0.35,
                 control_period: float = 0.02):
        self.adapter = adapter
        self.control = ControlState(watchdog_seconds=watchdog_seconds)
        self.control_period = float(control_period)
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bracketbot-control",
                                        daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        next_tick = time.monotonic()
        try:
            while not self._closing.is_set():
                command = self.control.motion()
                self.adapter.step(self.control_period, command.linear_mps,
                                  command.angular_rads)
                telemetry = self.adapter.telemetry()
                if telemetry.get("fallen"):
                    self.control.fault("Robot exceeded its safe pitch angle.")
                    self.adapter.stop()
                next_tick += self.control_period
                self._closing.wait(max(0.0, next_tick - time.monotonic()))
        except Exception as error:
            self.control.fault(f"Robot control loop failed: {error}")
            try:
                self.adapter.stop()
            except Exception:
                pass

    def drive(self, linear: float, angular: float):
        self.control.drive(linear, angular)

    def stop(self, reason: str = "Stopped"):
        self.control.stop(reason)
        self.adapter.stop()

    def emergency_stop(self):
        self.control.emergency_stop()
        self.adapter.stop()

    def reset_emergency_stop(self):
        self.control.reset_emergency_stop()
        self.adapter.stop()

    def _ensure_actuators_enabled(self):
        snapshot = self.control.snapshot()
        if snapshot["emergency_stop"]:
            raise RuntimeError("Emergency stop is engaged.")
        if snapshot["fault"]:
            raise RuntimeError(str(snapshot["fault"]))

    def set_arm_target(self, joint: str, value: float) -> float:
        self._ensure_actuators_enabled()
        return self.adapter.set_arm_target(joint, value)

    def set_gripper(self, side: str, action: str):
        self._ensure_actuators_enabled()
        self.adapter.set_gripper(side, action)

    def run_text_command(self, text: str) -> dict[str, object]:
        command = parse_command(text)
        self._execute(command)
        return {"ok": True, "message": command.message,
                "action": command.action, "payload": command.payload}

    def _execute(self, command: ParsedCommand):
        if command.action == "drive":
            self.drive(float(command.payload["linear"]),
                       float(command.payload["angular"]))
        elif command.action == "stop":
            self.stop("Stopped by text command")
        elif command.action == "estop":
            self.emergency_stop()
        elif command.action == "reset_estop":
            self.reset_emergency_stop()
        elif command.action == "gripper":
            self.set_gripper(str(command.payload["side"]),
                             str(command.payload["action"]))
        else:
            raise ValueError(f"Unsupported command action: {command.action}")

    def status(self) -> dict[str, object]:
        status = self.control.snapshot()
        status.update({
            "connected": self._thread.is_alive() and not status["fault"],
            "capabilities": self.adapter.capabilities,
            "telemetry": self.adapter.telemetry(),
        })
        return status

    def close(self):
        self.emergency_stop()
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.adapter.close()


class RemoteServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, runtime: RobotRuntime):
        super().__init__(address, RemoteHandler)
        self.runtime = runtime


class RemoteHandler(BaseHTTPRequestHandler):
    server: RemoteServer
    protocol_version = "HTTP/1.1"

    def log_message(self, message, *args):
        print(f"{self.client_address[0]} - {message % args}")

    def _headers(self, content_type: str, length: int,
                 status: HTTPStatus = HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' blob: data:; "
                         "connect-src 'self'; style-src 'self'; script-src 'self'")
        self.end_headers()

    def _json(self, payload: dict[str, object],
              status: HTTPStatus = HTTPStatus.OK):
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self._headers("application/json; charset=utf-8", len(body), status)
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise ValueError("Invalid Content-Length.") from error
        if length <= 0 or length > 8192:
            raise ValueError("JSON body must be between 1 and 8192 bytes.")
        try:
            value = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Request body must be valid JSON.") from error
        if not isinstance(value, dict):
            raise ValueError("Request body must be a JSON object.")
        return value

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._json(self.server.runtime.status())
            return
        if parsed.path == "/api/camera":
            try:
                name = parse_qs(parsed.query).get("name", ["head_rgb"])[0]
                image = self.server.runtime.adapter.camera_bmp(name)
                self._headers("image/bmp", len(image))
                self.wfile.write(image)
            except ValueError as error:
                self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            except Exception as error:
                self._json({"ok": False, "error": f"Camera unavailable: {error}"},
                           HTTPStatus.SERVICE_UNAVAILABLE)
            return

        filename = STATIC_FILES.get(parsed.path)
        if filename is None:
            self._json({"ok": False, "error": "Not found."}, HTTPStatus.NOT_FOUND)
            return
        body = (STATIC_DIR / filename).read_bytes()
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        if filename.endswith((".html", ".css", ".js")):
            content_type += "; charset=utf-8"
        self._headers(content_type, len(body))
        self.wfile.write(body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            payload = self._read_json()
            if path == "/api/drive":
                self.server.runtime.drive(float(payload.get("linear", 0.0)),
                                          float(payload.get("angular", 0.0)))
                result = {"ok": True, "message": "Drive command accepted."}
            elif path == "/api/stop":
                self.server.runtime.stop("Stopped by operator")
                result = {"ok": True, "message": "Stopped."}
            elif path == "/api/emergency-stop":
                self.server.runtime.emergency_stop()
                result = {"ok": True, "message": "Emergency stop engaged."}
            elif path == "/api/emergency-stop/reset":
                self.server.runtime.reset_emergency_stop()
                result = {"ok": True,
                          "message": "Emergency stop reset; robot remains stopped."}
            elif path == "/api/arm":
                target = self.server.runtime.set_arm_target(
                    str(payload.get("joint", "")), float(payload.get("value", 0.0)))
                result = {"ok": True, "message": "Arm target accepted.",
                          "target": target}
            elif path == "/api/gripper":
                side = str(payload.get("side", "both"))
                action = str(payload.get("action", "hold"))
                self.server.runtime.set_gripper(side, action)
                result = {"ok": True,
                          "message": f"Gripper command accepted: {side} {action}."}
            elif path == "/api/command":
                result = self.server.runtime.run_text_command(str(payload.get("text", "")))
            else:
                self._json({"ok": False, "error": "Not found."}, HTTPStatus.NOT_FOUND)
                return
            self._json(result)
        except ValueError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.CONFLICT)
        except Exception as error:
            self.server.runtime.emergency_stop()
            self._json({"ok": False, "error": f"Command failed safely: {error}"},
                       HTTPStatus.INTERNAL_SERVER_ERROR)


def local_ip() -> str:
    candidates = []
    try:
        candidates = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    return next((address for address in candidates
                 if not address.startswith(("127.", "169.254."))), "127.0.0.1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0",
                        help="listen address (default: all local interfaces)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--watchdog-ms", type=int, default=350)
    return parser.parse_args()


def main():
    args = parse_args()
    adapter = SimulationRobotAdapter(args.scene)
    runtime = RobotRuntime(adapter, watchdog_seconds=args.watchdog_ms / 1000.0)
    server = RemoteServer((args.host, args.port), runtime)
    runtime.start()
    print("BracketBot Remote is ready")
    print(f"Computer: http://127.0.0.1:{args.port}")
    print(f"Phone:    http://{local_ip()}:{args.port}")
    print("Press Ctrl+C to stop the server and robot.")
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping BracketBot safely...")
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


if __name__ == "__main__":
    main()

