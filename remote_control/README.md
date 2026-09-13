# Butler[bot] mobile remote

The remote is the phone-facing entry point for the merged MuJoCo, perception,
local-LLM, navigation, and manipulation work. It runs one BracketBot and steps
either manual drive commands or one existing high-level chore controller
against that same simulation state.

## What it reuses

- main_mujoco/bracketbot_sim/robot.py: drive, arm, gripper, RGB, depth, and the
  single MuJoCo stepping loop.
- main_mujoco/chopped_dynamic.xml: the existing left/right head and wrist
  camera poses.
- comp_vision_sim/vision_sim/llm_command.py: the existing OpenAI-compatible
  client for the local llama-server.
- comp_vision_sim/vision_sim/perception.py and
  Hand_and_Wrists/handwrist/vision.py: registered MuJoCo RGB-D perception.
- integration/pick_adapter.py: spoken object names to the manipulation
  catalogue.
- Hand_and_Wrists/handwrist/tasks.py: high-level pick, fetch, put, and tidy
  controllers, including the existing navigation, grasp, and place mechanics.

The browser never imports MuJoCo details. It asks the backend for capabilities,
camera frames, task status, and robot commands, leaving a hardware adapter
boundary for later.

## Install

From the repository root:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r remote_control\requirements.txt
~~~

The complete repository test dependencies are separate:

~~~powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-test.txt
~~~

## Local LLM

The merged intelligence stack expects an already-running OpenAI-compatible
llama-server at http://localhost:8080/v1, serving Qwen/Qwen3.8-27B. The
repository does not contain the GGUF/mmproj files or a machine-specific
llama-server launch script, so start the teammate-managed server with its
existing model paths before launching the remote.

Check it without sending a prompt:

~~~powershell
Invoke-RestMethod http://localhost:8080/v1/models
~~~

If that service is unavailable, the teammate-provided deterministic
fallback_task still extracts an object name. The phone reports the local LLM as
unavailable rather than claiming the fallback came from the model.

## Start the complete MuJoCo remote

The server creates the one shared MuJoCo world; do not start a separate
main_mujoco/run.py process for this workflow.

~~~powershell
cd C:\Users\devar\OneDrive\Desktop\bracketbot
.\.venv\Scripts\python.exe -m remote_control.server
~~~

The default scene is Hand_and_Wrists/scenes/scene_home.xml, the merged
household scene that supports pick, fetch, put, and tidy. Override it only when
testing another compatible scene:

~~~powershell
.\.venv\Scripts\python.exe -m remote_control.server --scene path\to\scene.xml
~~~

Open http://127.0.0.1:8000 on the laptop. The server prints the LAN URL for a
phone on the same network. If Windows asks, allow Python on private networks.

The phone receives a 320x240 multipart JPEG stream for only its selected
camera. The backend retains one newest frame, drops overwritten frames, and
stops live rendering shortly after the viewer disconnects. RGB-D perception
continues to use the original MuJoCo arrays rather than the compressed display
image. Camera FPS, approximate frame age, active view, and dropped-frame counts
are available under **Advanced**.

To list the computer's current LAN IPv4 addresses:

~~~powershell
Get-NetIPAddress -AddressFamily IPv4 |
  Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' }
~~~

Then open http://LAN-IP:8000 on the phone. Camera and typed control work over
plain HTTP. iPhone microphone capture requires a trusted HTTPS origin; see the
next section.

## Local voice recognition on a phone

Every browser records the command with `MediaRecorder` and posts the audio to
`/api/transcribe`; the robot computer transcribes it with Whisper
(`faster-whisper`, the repository's shared `comp_vision_sim/vision_sim/speech.py`
wrapper, `small.en` by default, GPU when there is room and the CPU otherwise).
The browser's own speech recognition, which sends audio to a cloud service, is
no longer used. The transcript enters the same `/api/task` endpoint as typed
commands. The server loads the model at start-up; the first run downloads it.

On the robot computer itself, `http://127.0.0.1:8000` is a secure context, so
the microphone works in any desktop browser without certificates.

To preload the default small English model before the demo:

~~~powershell
$env:HF_HOME = Join-Path $env:TEMP 'butlerbot-hf'
.\.venv\Scripts\python.exe -c "from faster_whisper import WhisperModel; WhisperModel('tiny.en', device='cpu', compute_type='int8'); print('Local STT ready')"
~~~

Safari on an iPhone permits microphone capture only from a secure context. A
LAN address such as `http://192.168.x.x:8000` is not a secure context. Supply a
certificate whose issuing CA is installed and explicitly trusted on the phone:

~~~powershell
$lanIp = .\.venv\Scripts\python.exe -c "from remote_control.server import local_ip; print(local_ip())"
$certDir = Join-Path $env:LOCALAPPDATA 'ButlerBot\certs'
$openssl = 'C:\msys64\ucrt64\bin\openssl.exe'
New-Item -ItemType Directory -Force $certDir | Out-Null
& $openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 3650 `
  -keyout "$certDir\ca.key" -out "$certDir\ca.crt" `
  -subj '/CN=ButlerBot Local CA' `
  -addext 'basicConstraints=critical,CA:TRUE' `
  -addext 'keyUsage=critical,keyCertSign,cRLSign'
& $openssl req -newkey rsa:2048 -sha256 -nodes `
  -keyout "$certDir\server.key" -out "$certDir\server.csr" `
  -subj "/CN=$lanIp"
@"
subjectAltName=IP:$lanIp
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
"@ | Set-Content "$certDir\server.ext" -Encoding ascii
& $openssl x509 -req -sha256 -days 365 `
  -in "$certDir\server.csr" -CA "$certDir\ca.crt" `
  -CAkey "$certDir\ca.key" -CAcreateserial `
  -out "$certDir\server.crt" -extfile "$certDir\server.ext"

.\.venv\Scripts\python.exe -m remote_control.server `
  --cert-file "$certDir\server.crt" `
  --key-file "$certDir\server.key"
~~~

Open the printed `https://LAN-IP:8000` address. On iOS, install the local CA
certificate as a profile, then enable it under **Settings > General > About >
Certificate Trust Settings**. A browser warning that is merely clicked through
may still leave the page outside a trusted secure context. The remote detects
that condition and reports it before recording.

## Voice on a phone: HTTPS through Tailscale

Browsers only give a page the microphone on HTTPS or `localhost`. Opened as
`http://100.x.y.z:8000` from a phone, TALK has no microphone to use, in any
browser. Tailscale can put a trusted certificate in front of the remote, reachable
only from your own tailnet:

~~~bash
sudo tailscale set --operator=$USER      # once, so tailscale serve works without sudo
tailscale serve --bg --https=10000 http://127.0.0.1:8000
~~~

The server detects this at start-up, prints `Phone with voice (HTTPS):
https://<machine>.<tailnet>.ts.net:10000`, and the page offers a NEEDS HTTPS
button that opens that address. `tailscale serve --https=10000 off` removes it.

## The robot's voice

When a task ends the page speaks the robot's final line, the model's own
confirmation of what it did (or why it could not). The audio is made on the robot
computer by Piper, an open-source neural voice, served from `/api/speech`, so it
plays on whichever device the remote is open on. Install once:

~~~bash
~/miniforge3/bin/python -m pip install piper-tts==1.8.0
~/miniforge3/bin/python -m piper.download_voices en_US-lessac-medium --data-dir ~/.local/share/piper-voices
~~~

Without Piper the server falls back to `espeak-ng`.

## The LLM agent brain and the viewer

By default (`--brain agent`) a request goes to `robot_agent`: the local LLM
calls tools -- look around, go near, inspect, plan a grasp, pick up, put down,
list surfaces -- one at a time against this same robot, reading each result
before the next, so it is not limited to four chores and seven items. The task
panel shows each tool call as it happens and the model's reply at the end. The
agent steps the robot itself, in real time; the control loop stands aside until
it finishes, and CANCEL TASK, E-STOP, the joystick or a fall interrupt it within
0.1 s of simulated time. `--brain chores` keeps the fixed-chore pipeline.

To watch the robot in 3-D while driving it from the page:

~~~bash
~/miniforge3/bin/python -m remote_control.server --viewer
~~~

Then open `http://127.0.0.1:8000`, hold TALK, and say a request. Closing the
viewer window stops the server.

## Demo

1. Confirm the phone says **CONNECTED** and the live Head L frame is moving.
2. Switch among **SCENE** (a third-person view that follows the robot around the
   room, shown first), **HEAD L**, **HEAD R**, **WRIST L**, and **WRIST R**. Head
   cameras also offer an RGB/depth display toggle. The head views are shown with
   the head depth sensor's 22 degree downward tilt; the model's stereo cameras
   look dead level, which put everything near the robot out of shot.
3. Move the joystick briefly. Releasing it sends an immediate stop; the
   independent 350 ms server watchdog stops stale commands.
4. Hold **TALK**, say a command, then release. Typed commands use the same task
   endpoint.
5. Try "Pick up the red cup", "Bring me the water bottle",
   "Put the bottle on the table", or "Tidy up the table".
6. Watch the honest task state produced by the existing controller.
7. Press **CANCEL TASK** to stop the chore, freeze the arms, and return to
   manual mode. Moving the joystick also performs this takeover.
8. Press **E-STOP** to cancel the chore and freeze base and arm commands.
   **RESET E-STOP** is explicit and leaves the robot stopped.

Hold TALK while speaking, then release. The button shows microphone permission,
recording, and local transcription states. The recognized text is displayed
before it is sent to the task pipeline. Local transcription can be tested while
the LLM is offline; only the subsequent autonomous task will report the LLM
state. Permission, microphone, secure-context, and audio decode failures are
shown explicitly. The typed task box remains available and exercises the
identical task pipeline.

## Tests

~~~powershell
.\.venv\Scripts\python.exe -m pytest remote_control\tests -q
.\.venv\Scripts\python.exe -m pytest
node --check remote_control\static\app.js
python -m compileall -q remote_control
~~~
