"""Talk to the BracketBot: the local LLM carries out requests with tool calls.

    python -m robot_agent "put the remote in the basket"       # one request, then exit
    python -m robot_agent                                       # type requests, one per line
    python -m robot_agent --voice                               # speak them (Whisper, local)
    python -m robot_agent --scene apartment --viewer            # another scene, watched
    python -m robot_agent --list-tools

Scenes: living_room (Hand_and_Wrists/scenes/scene_home.xml, the default),
apartment (comp_vision_sim/home_search.xml), or a path to any MJCF that
includes the robot. Surfaces are found in whichever scene is loaded, and
objects are found by name with YOLO-World (the vision LLM when YOLO misses).
Say "quit" to stop.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m robot_agent", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("request", nargs="*", help="what to do; omit to be asked")
    p.add_argument("--scene", default="living_room",
                   help="living_room, apartment, or a scene file (default: %(default)s)")
    p.add_argument("--viewer", action="store_true", help="show the MuJoCo viewer")
    p.add_argument("--speed", type=float, default=1.0, help="viewer realtime factor")
    p.add_argument("--detector", default="auto", choices=["auto", "yolo", "llm"],
                   help="how objects are found: YOLO-World, the vision LLM, or YOLO then the LLM")
    p.add_argument("--truth", action="store_true",
                   help="measure objects from the simulator instead of the camera (debugging)")
    p.add_argument("--llm-url", default="http://localhost:8080/v1")
    p.add_argument("--llm-model", default="Qwen/Qwen3.8-27B")
    p.add_argument("--thinking", action="store_true",
                   help="let the model reason before each tool call (slower)")
    p.add_argument("--max-steps", type=int, default=40, help="tool calls per request")
    p.add_argument("--voice", action="store_true", help="speak requests into the microphone")
    p.add_argument("--voice-seconds", type=float, default=10.0)
    p.add_argument("--whisper-model", default="small.en")
    p.add_argument("--whisper-device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--mic", default=None, help="input device index or name substring")
    p.add_argument("--quiet", action="store_true", help="hide skill and tool logs")
    p.add_argument("--list-tools", action="store_true", help="print the tools and exit")
    return p.parse_args(argv)


def requests_from(args, ask=input):
    """Yield requests: the command-line one, or typed / spoken ones until quit."""
    if args.request:
        yield " ".join(args.request)
        return
    stt = None
    if args.voice:
        sys.path.insert(0, str(REPO / "comp_vision_sim"))
        from vision_sim.speech import SpeechToText, listen
        stt = SpeechToText(args.whisper_model, device=args.whisper_device, verbose=not args.quiet)
        mic = int(args.mic) if args.mic and args.mic.isdigit() else args.mic
    while True:
        try:
            if stt is not None:
                text = listen(stt, args.voice_seconds, ask=ask, device=mic)
                print(f"you (voice): {text}" if text else "(heard nothing)")
            else:
                text = ask("you: ")
        except (EOFError, KeyboardInterrupt):
            return
        text = text.strip()
        if text.lower().strip(".!") in ("quit", "exit", "stop listening", "goodbye"):
            return
        if text:
            yield text


def main(argv=None):
    args = parse_args(argv)
    if args.list_tools:
        from robot_agent.tools import tool_catalogue
        for t in tool_catalogue():
            print(f"{t.name}({', '.join(t.parameters)})\n    {t.description}")
        return
    if args.viewer:
        os.environ["MUJOCO_GL"] = "glfw"

    from robot_agent.agent import RobotAgent
    from robot_agent.tools import RobotTools

    tools = RobotTools(scene=args.scene, truth=args.truth, detector=args.detector,
                       llm_url=args.llm_url, llm_model=args.llm_model, verbose=not args.quiet)
    viewer = None
    clock = {"wall": time.time(), "sim": tools.bot.time}
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(tools.bot.model, tools.bot.data)

        def on_step(bot):
            if not viewer.is_running():
                raise KeyboardInterrupt("viewer closed")
            viewer.sync()
            lag = (bot.time - clock["sim"]) / max(args.speed, 1e-6) - (time.time() - clock["wall"])
            if lag > 0:
                time.sleep(lag)
            elif lag < -0.5:        # after a long LLM wait, don't sprint to catch up
                clock["wall"], clock["sim"] = time.time(), bot.time

        tools.on_step = on_step

    agent = RobotAgent(tools, base_url=args.llm_url, model=args.llm_model,
                       max_steps=args.max_steps, thinking=args.thinking, verbose=not args.quiet)
    print(f"BracketBot ready in {tools.scene.name}: {len(tools.names)} tools, "
          f"{len(tools.surfaces)} surfaces, "
          f"{'ground-truth' if args.truth else args.detector + ' detector'}, model {args.llm_model}")
    try:
        for request in requests_from(args):
            clock["wall"], clock["sim"] = time.time(), tools.bot.time
            print(f"robot: {agent.run(request)}")
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if viewer is not None:
            viewer.close()
        tools.close()


if __name__ == "__main__":
    main()
