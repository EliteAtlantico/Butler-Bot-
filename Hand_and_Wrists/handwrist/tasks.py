"""Chores: skills chained into tasks you can ask for by name.

    task = make_task(bot, "fetch", item="remote")        # hand it to the person
    task = make_task(bot, "put", item="remote", to="basket")
    task = make_task(bot, "tidy")                        # coffee table -> basket
    while not task.done:
        bot.step(0.1, controller=task)
    print(task.status, task.succeeded)

`make_task(bot, action, **arguments)` takes an action name and keyword
arguments -- the same shape as a parsed command (action + payload) -- so any
front end, a CLI, a benchmark, a phone remote or a voice interface, can
start a chore without knowing how it is done. Every task is an ordinary
`f(bot, t)` controller with `.done`, `.succeeded`, `.failure`, `.status` (one
line: what it is doing now) and `.results` (each step and how it went).
"""
from __future__ import annotations

import numpy as np

from .objects import CATALOGUE
from .place import Place
from .places import PLACES, surface_of
from .skills import Pick

ACTIONS = {
    "pick": "pick up an item and hold it                 item",
    "put": "pick up an item and put it at a place       item, to (default basket)",
    "fetch": "find an item and hand it to the person      item",
    "tidy": "move every item on a surface to the basket  surface (default coffee_table), into",
}


class Task:
    """A chore run as one f(bot, t) controller.

    Wraps a generator that yields skills one at a time. Each skill runs to
    completion before the next is asked for, so a later step can depend on
    how an earlier one went (no hand-over after a failed pick; tidy moves on
    to the next item if one is out of reach).
    """

    def __init__(self, action, label, steps):
        self.action, self.label = action, label
        self._steps = steps
        self.current = None
        self.last = None
        self.results = []            # (step label, succeeded, failure)
        self.finished = False
        self.note = ""               # e.g. what tidy found to do

    @property
    def done(self):
        return self.finished

    @property
    def succeeded(self):
        return self.finished and bool(self.results) and all(ok for _, ok, _ in self.results)

    @property
    def failure(self):
        for label, ok, why in self.results:
            if not ok:
                return f"{label}: {why}"
        if not self.finished:
            return None                      # still running, nothing failed yet
        return None if self.results else (self.note or "nothing to do")

    @property
    def status(self):
        if not self.finished:
            return self.current.status if self.current else f"{self.label}: starting"
        return f"{self.label}: " + ("done" if self.succeeded else f"failed ({self.failure})")

    def __call__(self, bot, t):
        if self.finished:
            if self.last is not None:
                self.last(bot, t)            # keep holding / keep the arms home
            else:
                bot.drive(0.0, 0.0)
            return
        if self.current is None:
            try:
                self.current = next(self._steps)
            except StopIteration:
                self.finished = True
                return
        self.current(bot, t)
        if self.current.done:
            self.results.append((self.current.label, self.current.succeeded,
                                 self.current.failure))
            self.last, self.current = self.current, None


# ------------------------------------------------------------------- chores
def _pick_then_place(bot, item, to, estimator, verbose):
    pick = Pick(bot, item, estimator=estimator, verbose=verbose)
    yield pick
    if pick.succeeded and to is not None:
        yield Place(bot, pick, to, verbose=verbose)


def _tidy(bot, surface, into, estimator, verbose, task):
    found = items_on(bot, surface, estimator)
    task.note = f"found {', '.join(found) or 'nothing'} on the {surface}"
    if verbose:
        print(f"    tidy: {task.note}")
    for name in found:
        pick = Pick(bot, name, estimator=estimator, verbose=verbose)
        yield pick
        if pick.succeeded:
            yield Place(bot, pick, into, verbose=verbose)


def items_on(bot, surface, estimator):
    """Items the estimator can see resting on `surface`, nearest first.

    "floor" means resting at floor level and not inside any named place.
    """
    here = bot.position[:2]
    found = []
    places = [surface_of(bot.model, bot.data, p) for p in PLACES]
    s = None if surface == "floor" else surface_of(bot.model, bot.data, surface)
    for name, spec in CATALOGUE.items():
        est = estimator(bot, spec)
        if est is None:
            continue
        if s is None:
            on = est.bottom_z < 0.05 and not any(p.contains(est.center, 0.02) for p in places)
        else:
            on = s.contains(est.center, 0.02) and abs(est.bottom_z - s.top) < 0.05
        if on:
            found.append((float(np.linalg.norm(est.center[:2] - here)), name))
    return [n for _, n in sorted(found)]


def _item(kwargs):
    name = kwargs.get("item")
    if name not in CATALOGUE:
        raise ValueError(f"I don't know an item called {name!r}; "
                         f"I know {', '.join(sorted(CATALOGUE))}")
    return name


def _place(name, default):
    name = name or default
    if name not in PLACES:
        raise ValueError(f"I don't know a place called {name!r}; "
                         f"I know {', '.join(sorted(PLACES))}")
    return name


def make_task(bot, action, estimator=None, verbose=True, **kwargs) -> Task:
    """Start a chore by name. Raises ValueError with a readable message for an
    unknown action, item or place.

    estimator: how items are found. Default: the head camera, finding each
    item by what it is (detection.DetectionEstimator). Pass
    skills.truth_estimator to be told where things are (tests, debugging), or
    vision.CameraEstimator for the calibrated-colour finder.
    """
    if estimator is None:
        from .detection import DetectionEstimator
        estimator = DetectionEstimator()
    action = str(action).lower()
    if action == "pick":
        item = _item(kwargs)
        return Task(action, f"pick up the {item}",
                    _pick_then_place(bot, item, None, estimator, verbose))
    if action == "put":
        item, to = _item(kwargs), _place(kwargs.get("to"), "basket")
        return Task(action, f"put the {item} {PLACES[to].preposition} the {to}",
                    _pick_then_place(bot, item, to, estimator, verbose))
    if action == "fetch":
        item, to = _item(kwargs), _place(kwargs.get("to"), "person")
        return Task(action, f"fetch the {item}",
                    _pick_then_place(bot, item, to, estimator, verbose))
    if action == "tidy":
        surface = kwargs.get("surface") or "coffee_table"
        if surface != "floor":
            surface = _place(surface, "coffee_table")
        into = _place(kwargs.get("into"), "basket")
        task = Task(action, f"tidy the {surface} into the {into}", iter(()))
        task._steps = _tidy(bot, surface, into, estimator, verbose, task)
        return task
    raise ValueError(f"I don't know how to {action!r}; I can {', '.join(ACTIONS)}")
