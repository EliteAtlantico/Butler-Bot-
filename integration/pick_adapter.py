"""Seam C: the navigator has arrived, so hand the object to the arm skills.

    from integration.pick_adapter import resolve_object, pick_after_arrival

    outcome = pick_after_arrival(bot, "a set of keys")
    print(outcome.summary())

What this actually has to bridge is narrower than it looks. `Pick` is already
a controller in the same `f(bot, t)` shape the navigator uses, and it does its
own looking, planning, parking and retrying -- so this module does not
reimplement any of that, and deliberately does not pass the navigator's world
XY across. `Pick` takes a CATALOGUE *name* and an estimator callable, and
finding the object is the estimator's job.

The real gap is vocabulary. The navigator's target is whatever the language
model called it -- "a set of keys", "the remote control", "my mug" -- while
`handwrist.objects.CATALOGUE` is keyed on exactly seven short names. Mapping
one to the other, and reporting cleanly when no mapping exists, is the whole
job here.

Nothing in `Hand_and_Wrists/handwrist/` is modified or subclassed.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "Hand_and_Wrists", REPO / "main_mujoco", REPO / "comp_vision_sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# Spoken names for the seven catalogue objects. Only aliases a person would
# actually say -- this is a lookup table, not an ontology, and an unknown word
# is reported rather than guessed at.
ALIASES: dict[str, str] = {
    "key": "keys", "keys": "keys", "keyring": "keys", "key ring": "keys",
    "keychain": "keys", "car keys": "keys", "house keys": "keys",
    "mug": "mug", "cup": "mug", "coffee": "mug", "coffee cup": "mug",
    "coffee mug": "mug", "tea cup": "mug",
    "can": "can", "tin": "can", "soda": "can", "soda can": "can",
    "drink can": "can", "beer can": "can",
    "bottle": "bottle", "water bottle": "bottle", "flask": "bottle",
    "remote": "remote", "remote control": "remote", "tv remote": "remote",
    "controller": "remote", "clicker": "remote", "zapper": "remote",
    "ball": "ball", "tennis ball": "ball", "toy ball": "ball",
    "box": "box", "carton": "box", "small box": "box", "block": "box",
}

_ARTICLES = re.compile(
    r"^(?:(?:a|an|the|my|our|your|some|that|this|his|her|their)\s+)+", re.I)
_NOISE = re.compile(r"[^a-z0-9\s]+")


def normalise(text: str) -> str:
    """Spoken phrase -> bare noun. 'The Remote Control!' -> 'remote control'."""
    s = _NOISE.sub(" ", (text or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = _ARTICLES.sub("", s).strip()
    return s


def resolve_object(target: str | None) -> str | None:
    """A navigator target -> a CATALOGUE key, or None if nothing matches.

    None is a real answer, not a failure to try harder: the robot can only
    pick up the seven things the arm skills know how to hold, and guessing at
    an eighth would fail in the gripper rather than here, where it can be
    explained.
    """
    from handwrist.objects import CATALOGUE

    s = normalise(target)
    if not s:
        return None
    if s in CATALOGUE:
        return s
    if s in ALIASES:
        return ALIASES[s]
    # Singularise a trailing plural ("mugs" -> "mug") before giving up on it.
    if s.endswith("s") and s[:-1] in CATALOGUE:
        return s[:-1]
    # Fall back to a whole-word hit anywhere in the phrase, longest first, so
    # "remote control" cannot be decided by the word "remote" alone when a
    # longer alias also matches.
    words = set(s.split())
    for phrase in sorted(ALIASES, key=len, reverse=True):
        if " " in phrase:
            if phrase in s:
                return ALIASES[phrase]
        elif phrase in words:
            return ALIASES[phrase]
    return None


@dataclass
class PickOutcome:
    """What happened when the arm was asked to pick the thing up."""
    attempted: bool
    target: str | None            # what the navigator was looking for
    object_name: str | None       # the CATALOGUE key it mapped to
    succeeded: bool = False
    failure: str | None = None    # Pick's own reason, when it ran and failed
    skipped: str | None = None    # why it never ran
    phase: str | None = None
    sim_seconds: float = 0.0

    def summary(self) -> str:
        if not self.attempted:
            return f"pick skipped: {self.skipped}"
        if self.succeeded:
            return (f"picked up the {self.object_name} "
                    f"in {self.sim_seconds:.1f}s of arm time")
        return (f"pick failed on the {self.object_name}: "
                f"{self.failure or 'ran out of time in ' + str(self.phase)}")


def make_pick(bot, object_name: str, use_camera: bool = True):
    """Build the Pick controller, with the head camera as its estimator.

    The camera estimator is the honest default: the truth estimator reads the
    object's pose straight out of the simulator, which no robot can do. It is
    still reachable with use_camera=False for isolating an arm failure from a
    perception one.
    """
    from handwrist.skills import Pick

    if not use_camera:
        return Pick(bot, object_name)
    from handwrist.vision import CameraEstimator
    return Pick(bot, object_name, estimator=CameraEstimator())


def pick_after_arrival(bot, target: str | None, *, use_camera: bool = True,
                       timeout: float = 90.0, control_dt: float = 0.02,
                       verbose: bool = True) -> PickOutcome:
    """Run a full pick attempt, starting from wherever the navigator stopped.

    Returns rather than raises: a pick that cannot happen is an outcome the
    caller reports, not an error that should tear down a run which has already
    driven across the house.
    """
    name = resolve_object(target)
    if name is None:
        from handwrist.objects import CATALOGUE
        return PickOutcome(
            attempted=False, target=target, object_name=None,
            skipped=(f"{target!r} is not something the arm knows how to hold "
                     f"(it knows: {', '.join(sorted(CATALOGUE))})"))

    pick = make_pick(bot, name, use_camera=use_camera)
    if verbose:
        print(f"pick: {target!r} -> {name!r}, "
              f"{'head camera' if use_camera else 'scene truth'} estimator")

    started = bot.time
    while bot.time - started < timeout and not pick.done:
        bot.step(control_dt, controller=pick)

    elapsed = bot.time - started
    return PickOutcome(
        attempted=True, target=target, object_name=name,
        succeeded=bool(pick.succeeded),
        failure=pick.failure or (None if pick.succeeded else
                                 f"ran out of time in {pick.phase}"),
        phase=pick.phase, sim_seconds=float(elapsed))
