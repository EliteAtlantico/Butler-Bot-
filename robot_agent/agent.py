"""The tool-calling loop: request -> model -> tool -> result -> model -> ... -> reply.

    agent = RobotAgent(RobotTools(scene="apartment"))
    print(agent.run("put the mug from the coffee table on the sideboard"))

Talks to any OpenAI-compatible server that supports `tools` (llama-server with
--jinja does). The conversation is kept between requests, so "now put it on the
side table" knows what "it" is; `reset()` forgets.
"""
from __future__ import annotations

import json
import time

SYSTEM_PROMPT = (
    'You are the mind of BracketBot, a two-wheeled self-balancing home robot with two arms, parallel '
    'grippers and a head RGB-D camera. A person talks to you; you act by calling tools, one at a time, '
    'reading each result before deciding the next.\n'
    '\n'
    'Nothing is pre-programmed as a chore. You build every task from steps and decide how each step is '
    'done:\n'
    '1. Find it. look_around turns a full circle and lists every match in the room; detect_objects looks '
    'only at the current view; search_for explores other rooms for something not found here; '
    'describe_view answers questions about the view. The head camera cannot see the floor within about '
    '1.2 m or a table top within about 0.9 m, so look from 1.5-2.5 m away.\n'
    '2. Get into position: go_near the object (about 1.5 m, facing it) or go_to_surface for things on a '
    'surface, so it is in view.\n'
    '3. Measure it: inspect_object(object, shape). Check fits_gripper and what it rests on.\n'
    '4. Choose the grasp, and check it with plan_grasp before acting:\n'
    '   - grasp "top" for most things; "side" for tall, narrow things such as bottles.\n'
    '   - align_wrist true for anything clearly longer than wide (a remote, keys, a box).\n'
    '   - grip_at "center" for balls; handle true for mugs and cups.\n'
    '   - arm "either" unless one fails; then try the other arm, or the other grasp.\n'
    '   No plan: move so it can be approached from another side, try another grasp, or explain why it '
    'cannot be picked up.\n'
    '5. pick_up with the parameters you chose. If it fails, read the phase and the failure, change '
    'something (grasp, arm, squeeze_mm, a better viewpoint) and retry, at most twice.\n'
    '6. Put it down: list_surfaces shows every surface in this scene; place_held_item sets things on '
    "tables, seats or a person's open hand and drops them into containers. Pass x, y to choose the spot.\n"
    'Chores are sequences of these: tidying a table is detecting what is on it, then picking and placing '
    "each thing in turn; fetching is picking it and placing it on the person's hand or the surface "
    'nearest them.\n'
    '\n'
    'Detections match words, not certainty: a stack of cartons can come back as "box". Check every '
    "candidate's size_cm, rests_on and fits_gripper before going for one; if they contradict the request "
    '(asked for something on the floor and it is 0.5 m up, or it is 50 cm across), it is something else, '
    'so choose another candidate or keep looking. Never drive onto an object or furniture: go_near a '
    'thing to look at it, go_to_surface to work at a surface.\n'
    '\n'
    'Rules: carry at most two things, one in each hand (pick_up uses the free hand; place_held_item '
    'takes object when both are full). Never repeat an identical call that just failed. Coordinates are '
    'metres in the room frame; headings are degrees, 0 = +x, counter-clockwise positive. When the request'
    ' is done, or cannot be done, stop calling tools and answer the person in one or two short, friendly '
    'sentences saying what happened.\n'
    '\n'
    '{world}'
)


class RobotAgent:
    def __init__(self, tools, base_url: str = "http://localhost:8080/v1",
                 model: str = "Qwen/Qwen3.8-27B", max_steps: int = 40,
                 temperature: float = 0.2, thinking: bool = False, timeout: float = 300.0,
                 verbose: bool = True, post=None):
        self.tools = tools
        self.base_url, self.model = base_url.rstrip("/"), model
        self.max_steps, self.temperature = max_steps, temperature
        self.thinking, self.timeout, self.verbose = thinking, timeout, verbose
        if post is None:
            import requests
            post = requests.post
        self._post = post
        self.messages: list[dict] = []
        self.trace: list[dict] = []          # every tool call of the last run
        self.max_repeats = 2                 # identical failing calls run before refusing
        self._failed: dict[tuple, int] = {}
        self.reset()

    def reset(self):
        self.messages = [{"role": "system",
                          "content": SYSTEM_PROMPT.format(world=self.tools.world_summary())}]

    def _chat(self) -> dict:
        payload = {"model": self.model, "messages": self.messages,
                   "tools": self.tools.specs(), "tool_choice": "auto",
                   "temperature": self.temperature, "max_tokens": 1024,
                   "chat_template_kwargs": {"enable_thinking": self.thinking}}
        r = self._post(f"{self.base_url}/chat/completions", json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"]

    def _say(self, text):
        if self.verbose:
            print(text)

    def run(self, request: str) -> str:
        """Carry out one request; returns the model's reply to the person."""
        self.messages.append({"role": "user", "content": request})
        self.trace = []
        self._failed = {}
        for _ in range(self.max_steps):
            t0 = time.time()
            message = self._chat()
            calls = message.get("tool_calls") or []
            content = (message.get("content") or "").strip()
            # Keep only what the chat template needs; reasoning_content would
            # bloat every later request.
            turn = {"role": "assistant", "content": content}
            if calls:
                turn["tool_calls"] = calls
            self.messages.append(turn)
            if not calls:
                self._say(f"  [agent] replied after {time.time() - t0:.1f}s")
                return content
            if content:
                self._say(f"  [agent] {content}")
            for call in calls:
                fn = call.get("function", {})
                name, raw = fn.get("name", ""), fn.get("arguments") or "{}"
                self._say(f"  [agent] -> {name}({raw})  ({time.time() - t0:.1f}s to decide)")
                started = time.time()
                result = self._call(name, raw)
                self._say(f"  [agent] <- {name}: {_brief(result)}  ({time.time() - started:.1f}s)")
                self.trace.append({"tool": name, "arguments": raw, "result": result})
                self.messages.append({"role": "tool", "tool_call_id": call.get("id", name),
                                      "name": name, "content": json.dumps(result)})
        self.messages.append({"role": "user", "content": "(Step limit reached. Stop calling "
                              "tools and tell the person where things stand.)"})
        message = self._chat()
        reply = (message.get("content") or "").strip() or "I ran out of steps before finishing."
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def _call(self, name, raw):
        """Run a tool, unless this exact call already failed max_repeats times.

        The model does not always take "never repeat a failing call" to heart:
        in one run it sent the same unreachable go_to thirty times in a row.
        """
        try:
            key = (name, json.dumps(json.loads(raw), sort_keys=True))
        except (TypeError, ValueError):
            key = (name, str(raw))
        n = self._failed.get(key, 0)
        if n >= self.max_repeats:
            return {"ok": False, "error": f"not run: this exact {name} call has already failed "
                                          f"{n} times. Change the arguments, use another tool, "
                                          "or tell the person it cannot be done."}
        result = self.tools.call(name, raw)
        if result.get("ok") is False:
            self._failed[key] = n + 1
        else:
            self._failed.pop(key, None)
        return result


def _brief(result: dict, limit: int = 260) -> str:
    text = json.dumps({k: v for k, v in result.items() if k != "robot"})
    return text if len(text) <= limit else text[:limit] + "..."
