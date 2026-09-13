"""Natural-language search commands: "find me a key" -> what to look for, and where.

One text-only query to the local LLM turns the request into a SearchTask:

    task = CommandInterpreter().parse("find me a key")
    task.target          # "keys"            -- the name the detector and prompts use
    task.description     # "a set of keys"
    task.likely_places   # ["entryway table", "kitchen counter", "sideboard", ...]
    task.reply           # "I'll look for your keys, starting with the tables and counters."

The navigator then runs the explore loop (`llm_explore.py`) with the task: at
every stop the LLM checks the survey photos for the object and picks the most
likely place it has not checked yet as the next waypoint, while YOLO watches
every frame on the way. After a few rounds without it the search gives up and
`run_navigation.py` hands the robot to the phone remote (Seam D).

Small things like keys are beyond the pretrained YOLO on these renders (best
confidence 0.17 even at 640 px), but the vision LLM boxes them from 3 m away,
which is why the LLM checks every survey rather than only choosing waypoints.

If the server is down or the answer is unusable, `fallback_task` pulls the
object out of the sentence ("find me a key" -> "key") with no place hints, so a
command never leaves the robot without something to search for.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm_reasoner import DEFAULT_BASE_URL, DEFAULT_MODEL, LLMClient, _extract_json

COMMAND_PROMPT = """You control a home robot that can drive around, look with its cameras, and pick up small objects it can reach.
The user said: "{command}"

Work out what the user wants the robot to find, go to, or fetch, and where that thing is usually found in a home.
Answer with ONLY a single-line compact JSON object -- no prose, no markdown fences. Keys, in this order:
{{"target": "<short name an object detector understands, e.g. keys, mug, remote control, sofa>", "description": "<the thing with an article, e.g. a set of keys>", "likely_places": ["<furniture or spot where it is usually found, most likely first, 3 to 6 of them>"], "reply": "<one short friendly sentence the robot says back>"}}
If the request is not about finding, going to or fetching something, set target to null and say so in reply."""

# Words around the object in a spoken request. Stripped from the front of the
# sentence, repeatedly, by the no-model fallback.
_LEADING = re.compile(
    r"^(?:(?:hey|hi|ok|okay|robot|bracketbot|please|can you|could you|would you|will you|"
    r"i need you to|i want you to|help me|go and|find|look for|search for|locate|get|fetch|"
    r"bring|show|where did i (?:put|leave)|where (?:is|are|did)|go to|drive to|head to|"
    r"take me to|go|me|us|my|our|the|a|an|some|to)\b[\s,]*)+",
    re.IGNORECASE)
_TRAILING = re.compile(r"(?:\s+(?:for me|for us|please|now|again))+$", re.IGNORECASE)


@dataclass
class SearchTask:
    command: str
    target: str | None                    # what the detector looks for
    description: str = ""                 # how prompts refer to it
    likely_places: list[str] = field(default_factory=list)
    reply: str = ""
    source: str = "model"                 # "model" | "fallback"
    error: str | None = None
    latency: float | None = None
    raw: dict | None = None

    @property
    def ok(self) -> bool:
        return bool(self.target)

    def summary(self) -> str:
        places = ", ".join(self.likely_places) or "no hints"
        return f"{self.target!r} ({self.source}); likely places: {places}"


def _clean(text) -> str:
    return " ".join(str(text).split()).strip(" .,!?;:\"'")


def fallback_task(command: str, error: str | None = None) -> SearchTask:
    """The object named in the sentence, found without a model."""
    text = _clean(command)
    obj = _TRAILING.sub("", _LEADING.sub("", text)).strip(" .,!?")
    target = obj.lower() or None
    reply = (f"I'll look for {obj}." if target else
             "Sorry, I couldn't tell what you want me to find.")
    return SearchTask(command=text, target=target, description=obj, reply=reply,
                      source="fallback", error=error)


class CommandInterpreter:
    """Turns a spoken request into a SearchTask with one text-only LLM query."""

    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str = DEFAULT_MODEL,
                 max_tokens: int = 400, temperature: float = 0.1, timeout: float = 120.0,
                 thinking: bool = False, verbose: bool = False):
        self.client = LLMClient(base_url, model, max_tokens=max_tokens, temperature=temperature,
                                timeout=timeout, verbose=verbose, thinking=thinking)
        self.verbose = verbose

    def prompt(self, command: str) -> str:
        return COMMAND_PROMPT.format(command=command.replace('"', "'"))

    def parse(self, command: str) -> SearchTask:
        command = _clean(command)
        if not command:
            raise ValueError("empty command")
        try:
            content, dt = self.client._chat(
                [{"type": "text", "text": self.prompt(command) + self.client._suffix()}])
        except Exception as e:  # server down, timeout, HTTP error
            return self._note(fallback_task(command, f"{type(e).__name__}: {e}"))
        try:
            raw = _extract_json(content)
        except ValueError as e:
            task = fallback_task(command, f"unparseable answer ({e})")
            task.latency = dt
            return self._note(task)
        task = self.interpret(command, raw)
        task.latency = dt
        return self._note(task)

    def interpret(self, command: str, raw: dict) -> SearchTask:
        target = raw.get("target")
        target = _clean(target).lower() if isinstance(target, str) else None
        if target in ("", "null", "none"):
            target = None
        places = raw.get("likely_places") or []
        if isinstance(places, str):
            places = places.split(",")
        places = [_clean(p) for p in places if isinstance(p, str) and _clean(p)]
        reply = " ".join(str(raw.get("reply") or "").split())   # a sentence keeps its full stop
        if target is None and "target" not in raw:
            # A malformed answer, not a refusal: fall back to the sentence itself.
            task = fallback_task(command, "answer had no target")
            task.likely_places, task.raw = places, raw
            return task
        description = _clean(raw.get("description") or "") or (target or "")
        return SearchTask(command=command, target=target, description=description,
                          likely_places=places[:6],
                          reply=reply or (f"I'll look for {description}." if target else ""),
                          raw=raw)

    def _note(self, task: SearchTask) -> SearchTask:
        if self.verbose:
            took = f" in {task.latency:.1f}s" if task.latency is not None else ""
            print(f"    [command] {task.summary()}{took}"
                  f"{f' -- {task.error}' if task.error else ''}")
        return task

    def __repr__(self):
        return f"<CommandInterpreter {self.client.model}>"
