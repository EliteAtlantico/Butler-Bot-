# ButlerBot

**Your own voice-activated, AI-powered home assistant.** Tell ButlerBot what you
want ("bring me the TV remote", "put the mug in the basket") and it works out
where to look, drives there without hitting anything, recognises the object,
picks it up and delivers it. From your phone you can watch through its cameras,
give it spoken or typed commands, and take over with a joystick at any moment.

ButlerBot runs on the BracketBot, a self-balancing, two-wheeled robot with two
arms. Here it runs in a full MuJoCo physics
simulation of the real hardware, down to the wheels, motors and fingertip pads.
Everything runs on your own computer: the language model, speech recognition and
vision are all local.

Team 14, Battle of the Schools.

---

## What it does

| | |
|---|---|
| **Understands you** | A local LLM turns a sentence into a plan. As a tool-calling agent it looks around, walks up to things, measures them, plans grasps, picks up and puts down, reading each result before deciding the next step. Speak or type; speech is transcribed locally with Whisper. |
| **Sees** | Head RGB-D camera. Objects are found by *what they are* with YOLO-World, an open-vocabulary detector ("remote control", "mug"), paired with depth for their exact 3-D position. A colour in the request ("the red mug") only breaks ties. The vision LLM steps in when the detector is unsure. |
| **Navigates** | The depth camera builds a live occupancy map; A\* plans a route through free space and re-plans as the map fills in. It handles thin chair legs, tabletops at hip height and single doorways, while an LQR controller keeps the robot balanced. |
| **Picks and places** | A grasp planner tries every approach, arm and wrist angle, checks each with inverse kinematics, parks the base, reaches, closes, and confirms the hold by touch before lifting. Things can be set on tables, dropped in the basket, or handed to a person. |
| **Phone remote** | A web app on your phone: live camera views (a third-person scene view, both head cameras and both wrists), a joystick, voice or typed commands, spoken replies, cancel and E-stop. If a search comes up empty, ButlerBot hands control to the phone instead of guessing. |

The household it knows: a **mug**, **soda can**, **water bottle**, **TV remote** and
**tennis ball**; places **coffee table**, **side table**, **laundry basket** and a
**person's hand**. Two scenes: a living room, and a two-room apartment where the
remote is hidden behind a divider.

---

## How it fits together

```
 voice / text ─► robot_agent (local LLM, tool calls) ──┐
 phone app ────► remote_control (web server) ──────────┤
                                                       ▼
            comp_vision_sim:  search, navigation, perception, YOLO-World
            Hand_and_Wrists:  grasp planning, pick, place, chores
            main_mujoco:      the BracketBot model, balance controller, cameras, arm IK
            integration:      the seams: navigator → arm (pick), give-up → phone
```

| Folder | What's in it |
|---|---|
| `main_mujoco/` | The simulatable BracketBot: model, LQR balance controller, drive, seven cameras, arm IK. [README](main_mujoco/README.md) |
| `comp_vision_sim/` | Perception, the occupancy grid and A\* navigation, natural-language search, voice input, the scenes and their realistic object models (`tools/make_assets.py`). |
| `Hand_and_Wrists/` | Arms, wrists and grippers: object estimates, grasp planning, `Pick`, `Place`, and chores (fetch, put, tidy). [README](Hand_and_Wrists/README.md) |
| `robot_agent/` | The LLM tool-calling agent that runs the robot. [README](robot_agent/README.md) |
| `remote_control/` | The phone remote: web server, app, voice, text-to-speech. [README](remote_control/README.md) |
| `integration/` | Glue between the subsystems. |
| `tests/`, `ci/` | The test suite and the CI scripts. [TESTING.md](TESTING.md) |

---

## Install

**You need:** Python **3.12 or 3.13** (NumPy 2.5 requires 3.12+), Git, about 3 GB
of disk for Python packages and model weights, and ideally 16 GB of RAM. An
NVIDIA GPU speeds up YOLO and Whisper, but everything also runs on the CPU. Works
on Windows, Linux and macOS.

### 1. Get the code

```bash
git clone https://github.com/EliteAtlantico/Team-14-Battle-of-The-Schools.git
cd Team-14-Battle-of-The-Schools
git checkout main
```

### 2. Make a virtual environment

Windows (PowerShell):
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Linux / macOS:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch, then everything else

PyTorch first, so the right build is chosen. CPU only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
or with an NVIDIA GPU (CUDA 12.8 build):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```
Then:
```bash
pip install -r requirements-test.txt            # simulator, YOLO-World, CLIP, tests
pip install -r remote_control/requirements.txt  # phone remote + Whisper speech recognition
pip install -r comp_vision_sim/requirements-voice.txt   # optional: talk to it through the PC's microphone
```

The first time the robot looks for something it downloads the YOLO-World and CLIP
weights (about 430 MB) into `weights/`. The first voice command downloads
Whisper's model.

### 4. Start the local language model

The agent, the natural-language search and the phone's command box talk to any
**OpenAI-compatible server with tool calling** at `http://localhost:8080/v1`. The
team uses [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`
with Qwen3.8-27B (a vision-capable model, so it can also describe what the camera
sees). The model files are not in this repository; download a GGUF and start it
with tool support:

```bash
llama-server -m <model>.gguf --mmproj <mmproj>.gguf --jinja --port 8080 -ngl 99
```

Check it's up:
```bash
curl http://localhost:8080/v1/models
```

A different model or address works too: pass `--llm-url` / `--llm-model` to the
commands below. Without an LLM the robot still drives, finds and picks things by
name; only free-form conversation needs it.

### 5. Check the install

```bash
python -m pytest tests/test_household_looks.py -q
```

---

## Run it on your computer

Every command opens a MuJoCo window unless you add `--headless`. In the viewer,
drag to orbit and scroll to zoom.

**The agent: talk to the robot**
```bash
python -m robot_agent --viewer                    # type requests one after another
python -m robot_agent --voice --viewer            # or say them
python -m robot_agent --scene apartment "bring the remote to the coffee table"
```

**Search the apartment for something**, find it, then pick it up:
```bash
cd comp_vision_sim
python run_navigation.py --scene home_search.xml --command "find my TV remote" --pick --viewer
```
If the search gives up, control passes to the phone remote on port 8000 (see
below); `--no-remote` turns that off.

**Chores in the living room**, with no LLM needed:
```bash
cd Hand_and_Wrists
python run_task.py fetch --item remote            # find the remote, hand it to the person
python run_task.py put --item mug --to basket
python run_task.py tidy                           # everything on the coffee table -> basket
```

**Just the robot**, balancing and driving itself around obstacles:
```bash
cd main_mujoco
python run.py --algorithm avoid
```

---

## Control it from your phone

The phone remote is a web page served by your computer. Nothing is installed on
the phone.

### 1. Start the server (on the computer)

From the repository root:
```bash
python -m remote_control.server             # add --viewer to watch in 3-D as well
```
It prints the addresses to open, for example:
```
BracketBot Remote is ready
Computer: http://127.0.0.1:8000
Phone:    http://192.168.1.23:8000
```

Useful options: `--scene comp_vision_sim/home_search.xml` (the apartment; the
default is the living room), `--brain chores` (the four fixed chores instead of the
LLM agent), `--port 8000`, `--llm-url ...`.

### 2. Open it on the phone

1. Put the phone on the **same Wi-Fi** as the computer.
2. Open the printed `http://<computer-IP>:8000` address in the phone's browser.
3. Windows: if a firewall prompt appears, allow Python on **private networks**.
   Linux with a firewall: `sudo ufw allow 8000/tcp`.
4. The page should say **CONNECTED** with a moving camera image.

To find the computer's address yourself: `ipconfig` (Windows) or `ip addr`
(Linux).

### 3. Use it

- **Camera views:** SCENE (third-person, follows the robot), HEAD L / HEAD R (with
  a depth toggle), WRIST L / WRIST R.
- **Joystick:** drive and turn. Let go and it stops at once; if the phone loses
  the connection, a watchdog stops the robot within 0.35 s.
- **Command box:** type "bring me the water bottle", "put the can in the
  basket", "tidy up the table". Each step shows as the robot does it, and the
  robot says its answer out loud when it finishes.
- **Take over any time:** touching the joystick, **CANCEL TASK**, or **E-STOP**
  interrupts the robot straight away. **RESET E-STOP** leaves it stopped until
  you drive again.

### 4. Voice from the phone (needs HTTPS)

Phones only allow the microphone on secure (HTTPS) pages, and a plain
`http://192.168…` address isn't one. Typing and the joystick work without it.
The easy way to get HTTPS is [Tailscale](https://tailscale.com) (free):

1. Install Tailscale on the computer and the phone, and sign in to the same
   account on both.
2. On the computer, while the server is running:
   ```bash
   tailscale serve --bg --https=10000 http://127.0.0.1:8000
   ```
   (On Linux, run `sudo tailscale set --operator=$USER` once first.)
3. Restart the remote. It prints `Phone with voice (HTTPS): https://<machine>.<tailnet>.ts.net:10000`,
   and the page shows a **NEEDS HTTPS** button that opens it.
4. Open that address on the phone, hold **TALK**, speak, and let go. The
   computer transcribes it with Whisper, and nothing goes to the cloud.

To stop sharing: `tailscale serve --https=10000 off`. Prefer your own
certificate? `--cert-file` / `--key-file` also work; see the
[remote_control README](remote_control/README.md).

**The robot's voice:** replies are spoken by Piper, a local neural voice:
```bash
pip install piper-tts==1.8.0
python -m piper.download_voices en_US-lessac-medium --data-dir ~/.local/share/piper-voices
```
Without it the server falls back to `espeak-ng`, if that's installed.

---

## Tests

```bash
python -m pytest                                   # the whole suite
python -m pytest -m "not rendering and not slow"   # quick: no OpenGL, no long runs
python ci/smoke_pipeline.py                        # end to end: search -> give up -> phone; drive -> pick
```

CI (GitHub Actions, `.github/workflows/ci.yml`) runs lint, the suite on Python
3.12 and 3.13, the smoke pipeline, and the pick and chore benchmarks against the
baselines in `Hand_and_Wrists/results/`. It runs on pushes to `main` and on pull
requests. See [TESTING.md](TESTING.md).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| Black window, or an `EGL` / `GLFW` error | Set the OpenGL backend: `MUJOCO_GL=egl` on a headless Linux machine, `glfw` for a desktop with a viewer. On Windows the code picks `wgl` itself. |
| "LLM unavailable" | Start `llama-server` (step 4) and check `curl http://localhost:8080/v1/models`. Chores and driving still work without it. |
| Nothing found by the camera | The first detection downloads YOLO-World into `weights/`, so it needs internet once. Keep the object in front of the robot and 1.3–2 m away: the head camera cannot see the floor closer than about 1.2 m. |
| Phone cannot connect | Same Wi-Fi? Firewall allows port 8000? Use the printed LAN address, not `127.0.0.1`. |
| TALK button greyed out on the phone | The page isn't HTTPS. Use Tailscale (above) or type the command instead. |
| Runs out of memory | YOLO-World and CLIP take about 1.5 GB per process. Close other simulator windows, and run evaluations with fewer `--workers`. |

---

## Credits

Built by Team 14 for Battle of the Schools on the open-source BracketBot. Object
models and textures are generated by `comp_vision_sim/tools/make_assets.py`,
except the basket weave, a supplied photo (see
`comp_vision_sim/assets/src/README.md` for its licence status).
