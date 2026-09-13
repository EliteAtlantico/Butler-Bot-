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
import ssl
import threading
import time
from urllib.parse import parse_qs, urlparse

from .agent_tasks import AgentTaskManager
from .autonomy import AutonomousTaskManager
from .command_parser import ParsedCommand, parse_command
from .control import ControlState
from .robot_adapter import DEFAULT_SCENE, SimulationRobotAdapter
from .speech import DEFAULT_MODEL as DEFAULT_STT_MODEL
from .speech import LocalSpeechToText
from .tts import RobotVoice


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
                 control_period: float = 0.02,
                 llm_url: str = "http://localhost:8080/v1",
                 llm_model: str = "Qwen/Qwen3.8-27B",
                 stt_model: str = DEFAULT_STT_MODEL,
                 stt_device: str = "auto",
                 brain: str = "agent",
                 scene=None,
                 preload_speech: bool = False):
        self.adapter = adapter
        self.control = ControlState(watchdog_seconds=watchdog_seconds)
        # "agent": the tool-calling LLM reasons its way through any request
        # (robot_agent). "chores": the four fixed chores for the seven known items.
        if brain == "agent":
            self.tasks = AgentTaskManager(adapter, llm_url=llm_url, llm_model=llm_model,
                                          scene=scene)
        elif brain == "chores":
            self.tasks = AutonomousTaskManager(adapter, llm_url=llm_url, llm_model=llm_model)
        else:
            raise ValueError(f"brain must be agent or chores, not {brain!r}")
        self.brain = brain
        self.speech = LocalSpeechToText(stt_model, device=stt_device)
        self.voice = RobotVoice()
        self.preload_speech = preload_speech
        # A secure (HTTPS) address for this remote, when there is one: browsers only give
        # a page the microphone on HTTPS (or localhost), so the page links phones to it.
        self.secure_url: str | None = None
        self.control_period = float(control_period)
        self._closing = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bracketbot-control",
                                        daemon=True)

    def start(self):
        if self.preload_speech:
            threading.Thread(target=self.speech.preload, name="butlerbot-whisper-load",
                             daemon=True).start()
        self._thread.start()

    def _run(self):
        next_tick = time.monotonic()
        try:
            while not self._closing.is_set():
                safety = self.control.snapshot()
                owns_robot = getattr(self.tasks, "owns_robot", None)
                task_controller = (None if safety["emergency_stop"] or safety["fault"]
                                   else self.tasks.controller())
                if callable(owns_robot) and owns_robot():
                    # The LLM agent's tools are stepping this same robot on their own
                    # thread; stepping it here as well would double the physics rate.
                    pass
                elif task_controller is not None:
                    try:
                        self.adapter.step_controller(self.control_period, task_controller)
                        self.tasks.after_step(task_controller)
                    except Exception as error:
                        self.tasks.fail(error)
                        self.adapter.hold_position()
                else:
                    command = self.control.motion()
                    self.adapter.step(self.control_period, command.linear_mps,
                                      command.angular_rads)
                telemetry = self.adapter.telemetry()
                if telemetry.get("fallen"):
                    self.tasks.cancel("Cancelled because the robot fell")
                    self.control.fault("Robot exceeded its safe pitch angle.")
                    self.adapter.hold_position()
                next_tick += self.control_period
                self._closing.wait(max(0.0, next_tick - time.monotonic()))
        except Exception as error:
            self.tasks.cancel(f"Cancelled because the control loop failed: {error}")
            self.control.fault(f"Robot control loop failed: {error}")
            try:
                self.adapter.stop()
            except Exception:
                pass
        finally:
            self.adapter.close_control_renderers()

    def drive(self, linear: float, angular: float):
        if abs(float(linear)) > 1e-6 or abs(float(angular)) > 1e-6:
            if self.tasks.cancel("Cancelled for manual control"):
                self.adapter.hold_position()
        self.control.drive(linear, angular)

    def stop(self, reason: str = "Stopped"):
        self.tasks.cancel(reason)
        self.control.stop(reason)
        self.adapter.hold_position()

    def emergency_stop(self):
        self.tasks.cancel("Cancelled by emergency stop")
        self.control.emergency_stop()
        self.adapter.hold_position()

    def reset_emergency_stop(self):
        self.control.reset_emergency_stop()
        self.adapter.stop()

    def start_task(self, text: str) -> dict[str, object]:
        self._ensure_actuators_enabled()
        self.control.stop("Autonomous task")
        self.adapter.stop()
        return self.tasks.start(text)

    def cancel_task(self) -> bool:
        cancelled = self.tasks.cancel("Cancelled by operator")
        self.control.stop("Task cancelled; manual control ready")
        self.adapter.hold_position()
        return cancelled

    def _ensure_actuators_enabled(self):
        snapshot = self.control.snapshot()
        if snapshot["emergency_stop"]:
            raise RuntimeError("Emergency stop is engaged.")
        if snapshot["fault"]:
            raise RuntimeError(str(snapshot["fault"]))

    def set_arm_target(self, joint: str, value: float) -> float:
        self._ensure_actuators_enabled()
        self.tasks.cancel("Cancelled for manual arm control")
        return self.adapter.set_arm_target(joint, value)

    def set_gripper(self, side: str, action: str):
        self._ensure_actuators_enabled()
        self.tasks.cancel("Cancelled for manual gripper control")
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
            "camera": self.adapter.camera_diagnostics(),
            "speech": self.speech.status(),
            "tts": self.voice.status(),
            "secure_url": self.secure_url,
            "brain": self.brain,
            "task": self.tasks.snapshot(),
        })
        return status

    def close(self):
        self.emergency_stop()
        self._closing.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.adapter.close()


def _safe_print(text: str):
    """Log a line without letting logging break the request being served.

    Every request is logged. Started as `server ... | tee log`, a closed reader made
    each print raise BrokenPipeError inside the handler, so every request -- the page
    included -- was dropped with an empty reply (a 502 through tailscale serve).
    """
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


class RemoteServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, runtime: RobotRuntime):
        super().__init__(address, RemoteHandler)
        self.runtime = runtime

    def handle_error(self, request, client_address):
        # Phones routinely close an old camera request while switching views.
        # That is not a robot fault and should not produce a server traceback.
        import sys
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class RemoteHandler(BaseHTTPRequestHandler):
    server: RemoteServer
    protocol_version = "HTTP/1.1"

    def log_message(self, message, *args):
        # A malformed request (a phone speaking TLS to the plain-http port) is logged
        # before `path` is parsed, and the old `self.path` raised AttributeError there.
        if getattr(self, "path", "").startswith("/api/camera"):
            return
        _safe_print(f"{self.client_address[0]} - {message % args}")

    def _headers(self, content_type: str, length: int,
                 status: HTTPStatus = HTTPStatus.OK,
                 extra: dict[str, object] | None = None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy",
                         "default-src 'self'; img-src 'self' blob: data:; "
                         "connect-src 'self'; style-src 'self'; script-src 'self'")
        for name, value in (extra or {}).items():
            self.send_header(name, str(value))
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
            status = self.server.runtime.status()
            capabilities = status.get("capabilities", {})
            # Existing Python/API consumers use raw MuJoCo names in `cameras`.
            # The browser remote predates `mobile_cameras`, so negotiate its
            # structured view here without changing the static client contract.
            user_agent = self.headers.get("User-Agent", "")
            wants_mobile = (
                user_agent.startswith("Mozilla/")
                or parse_qs(parsed.query).get("client") == ["mobile"]
            )
            if wants_mobile and capabilities.get("mobile_cameras"):
                capabilities["cameras"] = capabilities["mobile_cameras"]
            self._json(status)
            return
        if parsed.path == "/api/speech":
            self._speech_audio(parsed)
            return
        if parsed.path == "/api/camera/stream":
            self._camera_stream(parsed)
            return
        if parsed.path == "/api/camera":
            try:
                query = parse_qs(parsed.query)
                name = query.get("name", ["head-left"])[0]
                if (name in self.server.runtime.adapter.bot.camera_names
                        and name not in {camera["id"] for camera in
                                        self.server.runtime.adapter.capabilities[
                                            "mobile_cameras"]}):
                    image = self.server.runtime.adapter.camera_bmp(name)
                    self._headers("image/bmp", len(image))
                    self.wfile.write(image)
                    return
                mode = query.get("mode", ["rgb"])[0]
                try:
                    after = max(0, int(query.get("after", ["0"])[0]))
                except ValueError as error:
                    raise ValueError("Camera frame sequence must be an integer.") from error
                frame = self.server.runtime.adapter.camera_frame(
                    name, mode=mode, after=after, timeout=1.25)
                image = frame["data"]
                age_ms = max(0.0, (time.monotonic()
                                   - frame["captured_monotonic"]) * 1000.0)
                self._headers(frame["content_type"], len(image), extra={
                    "X-Frame-Sequence": frame["sequence"],
                    "X-Frame-Age-Ms": f"{age_ms:.1f}",
                    "X-Server-Wait-Ms": f"{frame['server_wait_ms']:.1f}",
                    "X-Render-Ms": f"{frame['render_ms']:.1f}",
                    "X-Render-FPS": f"{frame['render_fps']:.1f}",
                    "X-Dropped-Frames": frame["dropped_frames"],
                    "X-Active-Camera": frame["camera"],
                    "X-Camera-Mode": frame["mode"],
                })
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

    def _speech_audio(self, parsed):
        """The robot saying `text`, as WAV: the page plays each task's final line."""
        text = parse_qs(parsed.query).get("text", [""])[0]
        try:
            audio = self.server.runtime.voice.synthesize(text)
        except ValueError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.SERVICE_UNAVAILABLE)
            return
        self._headers("audio/wav", len(audio))
        self.wfile.write(audio)

    def _camera_stream(self, parsed):
        query = parse_qs(parsed.query)
        name = query.get("name", ["head-left"])[0]
        mode = query.get("mode", ["rgb"])[0]
        stream_id = object()
        try:
            frame = self.server.runtime.adapter.camera_frame(
                name, mode=mode, after=0, timeout=1.25, stream_id=stream_id)
        except ValueError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as error:
            self._json({"ok": False, "error": f"Camera unavailable: {error}"},
                       HTTPStatus.SERVICE_UNAVAILABLE)
            return

        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True

        try:
            while True:
                image = frame["data"]
                age_ms = max(0.0, (time.monotonic()
                                   - frame["captured_monotonic"]) * 1000.0)
                part_headers = (
                    "--frame\r\n"
                    f"Content-Type: {frame['content_type']}\r\n"
                    f"Content-Length: {len(image)}\r\n"
                    f"X-Frame-Sequence: {frame['sequence']}\r\n"
                    f"X-Frame-Age-Ms: {age_ms:.1f}\r\n"
                    f"X-Render-Ms: {frame['render_ms']:.1f}\r\n"
                    f"X-Render-FPS: {frame['render_fps']:.1f}\r\n"
                    f"X-Dropped-Frames: {frame['dropped_frames']}\r\n"
                    f"X-Active-Camera: {frame['camera']}\r\n"
                    f"X-Camera-Mode: {frame['mode']}\r\n\r\n"
                ).encode("ascii")
                self.wfile.write(part_headers)
                self.wfile.write(image)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
                frame = self.server.runtime.adapter.camera_frame(
                    name, mode=mode, after=frame["sequence"], timeout=1.25,
                    activate=False, stream_id=stream_id)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError,
                TimeoutError):
            return
        except RuntimeError as error:
            if "superseded by a newer selection" in str(error):
                return
            _safe_print(f"Camera stream stopped: {error}")
        except Exception as error:
            _safe_print(f"Camera stream stopped: {error}")

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/transcribe":
            self._transcribe_audio()
            return
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
            elif path == "/api/task":
                task = self.server.runtime.start_task(str(payload.get("text", "")))
                result = {"ok": True, "message": "Task submitted.", "task": task}
            elif path == "/api/task/cancel":
                cancelled = self.server.runtime.cancel_task()
                result = {"ok": True, "message": (
                    "Task cancelled; manual control ready." if cancelled
                    else "No active task to cancel.")}
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

    def _transcribe_audio(self):
        try:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Invalid audio Content-Length.") from error
            if length < 128:
                raise ValueError("The recording was empty or too short to transcribe.")
            if length > 12 * 1024 * 1024:
                raise ValueError("The recording is too large; keep it under 12 MB.")
            content_type = self.headers.get("Content-Type", "").split(";", 1)[0].lower()
            allowed = {
                "audio/mp4", "audio/webm", "audio/ogg", "audio/wav",
                "audio/x-wav", "audio/mpeg", "audio/flac",
                "application/octet-stream",
            }
            if content_type not in allowed:
                raise ValueError(f"Unsupported recording format: {content_type or 'unknown'}.")
            transcript = self.server.runtime.speech.transcribe(self.rfile.read(length))
            self._json({"ok": True, "transcript": transcript})
        except ValueError as error:
            self._json({"ok": False, "error": str(error)}, HTTPStatus.BAD_REQUEST)
        except RuntimeError as error:
            self._json({"ok": False, "error": str(error)},
                       HTTPStatus.SERVICE_UNAVAILABLE)


def tailscale_https_url(port: int) -> str | None:
    """The tailnet HTTPS address that `tailscale serve` forwards to this port, if any.

    Phones reach the remote over Tailscale, and over plain http:// a browser gives
    the page no microphone at all. `tailscale serve --bg --https=10000
    http://127.0.0.1:8000` puts a trusted certificate in front of it.
    """
    import shutil
    import subprocess

    exe = shutil.which("tailscale")
    if not exe:
        return None
    try:
        result = subprocess.run([exe, "serve", "status", "--json"], capture_output=True,
                                text=True, timeout=5)
        config = json.loads(result.stdout or "{}")
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    for host, site in (config.get("Web") or {}).items():
        for handler in ((site or {}).get("Handlers") or {}).values():
            proxy = str((handler or {}).get("Proxy", "")).rstrip("/")
            if proxy.endswith(f":{port}"):
                return "https://" + (host[:-4] if host.endswith(":443") else host)
    return None


def local_ip() -> str:
    candidates = []
    try:
        candidates = socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        pass
    return next((address for address in candidates
                 if not address.startswith(("127.", "169.254."))), "127.0.0.1")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0",
                        help="listen address (default: all local interfaces)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--watchdog-ms", type=int, default=350)
    parser.add_argument("--llm-url", default="http://localhost:8080/v1",
                        help="OpenAI-compatible URL of the existing local llama-server")
    parser.add_argument("--llm-model", default="Qwen/Qwen3.8-27B",
                        help="model name exposed by the existing local llama-server")
    parser.add_argument("--brain", default="agent", choices=("agent", "chores"),
                        help="agent: the tool-calling LLM reasons through any request "
                             "(default); chores: the four fixed chores")
    parser.add_argument("--viewer", action="store_true",
                        help="also open the MuJoCo viewer on the same robot")
    parser.add_argument("--stt-model", default=DEFAULT_STT_MODEL,
                        help="Whisper (faster-whisper) model name or directory "
                             "(default: %(default)s)")
    parser.add_argument("--stt-device", default="auto", choices=("auto", "cpu", "cuda"),
                        help="where Whisper runs; auto tries the GPU, then the CPU")
    parser.add_argument("--cert-file", type=Path,
                        help="TLS certificate PEM for phone microphone access")
    parser.add_argument("--key-file", type=Path,
                        help="TLS private-key PEM for phone microphone access")
    return parser.parse_args(argv)


def run_viewer(adapter, fps: float = 60.0):
    """Show the robot in the MuJoCo viewer until its window is closed.

    GLFW windows belong on the main thread, so this blocks there while the web
    server runs on another. Syncing takes the adapter's lock, like every step.
    """
    import mujoco.viewer

    # Launching runs mj_forward on the shared MjData. Unlocked, it raced the control
    # thread's mj_step and segfaulted inside the constraint solver (mj_fwdConstraint).
    with adapter.lock:
        handle = mujoco.viewer.launch_passive(adapter.bot.model, adapter.bot.data,
                                              show_left_ui=False, show_right_ui=False)
    with handle as viewer:
        while viewer.is_running():
            with adapter.lock:
                viewer.sync()
            time.sleep(1.0 / fps)


def serve(adapter: SimulationRobotAdapter, host: str = "0.0.0.0", port: int = 8000,
          *, watchdog_seconds: float = 0.35, on_ready=None,
          llm_url: str = "http://localhost:8080/v1",
          llm_model: str = "Qwen/Qwen3.8-27B",
          stt_model: str = DEFAULT_STT_MODEL, stt_device: str = "auto",
          tls: ssl.SSLContext | None = None, brain: str = "agent",
          viewer: bool = False, preload_speech: bool = False):
    """Serve the remote for `adapter` until Ctrl+C (or `server.shutdown()`).

    `on_ready(server)` is called once the socket is bound, before serving; it
    is how a caller that is not a terminal learns the real port (port 0) and
    gets a handle to stop the server from another thread.
    """
    runtime = RobotRuntime(
        adapter, watchdog_seconds=watchdog_seconds,
        llm_url=llm_url, llm_model=llm_model,
        stt_model=stt_model, stt_device=stt_device,
        brain=brain, scene=getattr(adapter, "scene", None),
        preload_speech=preload_speech)
    server = RemoteServer((host, port), runtime)
    scheme = "http"
    if tls is not None:
        server.socket = tls.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    port = server.server_address[1]
    runtime.secure_url = (f"https://{local_ip()}:{port}" if tls is not None
                          else tailscale_https_url(port))
    runtime.start()
    print("BracketBot Remote is ready")
    print(f"Computer: {scheme}://127.0.0.1:{port}")
    print(f"Phone:    {scheme}://{local_ip()}:{port}")
    if runtime.secure_url and tls is None:
        print(f"Phone with voice (HTTPS): {runtime.secure_url}")
    elif tls is None:
        print("Phone voice needs HTTPS. Once: sudo tailscale set --operator=$USER; then:\n"
              f"  tailscale serve --bg --https=10000 http://127.0.0.1:{port}")
    print(f"Brain: {brain}. Voice: Whisper {stt_model} ({stt_device}), transcribed on this computer.")
    print("Press Ctrl+C to stop the server and robot."
          + (" Closing the viewer window also stops it." if viewer else ""))
    if on_ready is not None:
        on_ready(server)
    try:
        if viewer:
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2},
                             name="butlerbot-http", daemon=True).start()
            run_viewer(adapter)
        else:
            server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nStopping BracketBot safely...")
    finally:
        server.shutdown()
        server.server_close()
        runtime.close()


def main():
    args = parse_args()
    if bool(args.cert_file) != bool(args.key_file):
        raise SystemExit("--cert-file and --key-file must be provided together")
    tls = None
    if args.cert_file:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.load_cert_chain(args.cert_file, args.key_file)
    adapter = SimulationRobotAdapter(args.scene)
    serve(
        adapter, args.host, args.port,
        watchdog_seconds=args.watchdog_ms / 1000.0,
        llm_url=args.llm_url, llm_model=args.llm_model,
        stt_model=args.stt_model, stt_device=args.stt_device, tls=tls,
        brain=args.brain, viewer=args.viewer, preload_speech=True)


if __name__ == "__main__":
    main()
