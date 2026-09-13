"""Every BracketBot capability as a function tool the LLM can call -- in any scene.

    tools = RobotTools(scene="apartment")
    tools.call("inspect_object", {"object": "mug"})
    tools.call("pick_up", {"object": "mug", "grasp": "top", "handle": True})
    tools.call("place_held_item", {"surface": "sideboard"})
    tools.specs()                                     # OpenAI `tools` schemas

Manipulation is not a fixed list of chores. The model composes it from the
steps the chores were built from, and chooses how each step is done:

  find      detect_objects, search_for, describe_view      YOLO-World, the vision LLM
  measure   inspect_object     open-vocabulary box + depth -> centre, size, axis, support
  decide    plan_grasp         top or side, which arm, wrist aligned, where to grip
  act       pick_up            handwrist.skills.Pick with the ObjectSpec it chose
  put down  list_surfaces, place_held_item    handwrist.place.Place on any surface
                               found in the scene's geometry (handwrist.surfaces)

A tool runs its controller to completion -- the sim is stepped until the skill
is done, the robot falls, or a time limit passes -- and returns a small
JSON-able dict: whether it worked, why not in words the model can act on, and
where the robot ended up. The sim only advances while a tool runs, so the robot
stands frozen (and balanced) while the model thinks.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO / "Hand_and_Wrists", REPO / "main_mujoco", REPO / "comp_vision_sim"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
os.environ.setdefault("MUJOCO_GL", "wgl" if os.name == "nt" else "egl")

import numpy as np  # noqa: E402

CHUNK = 0.1            # sim seconds stepped between progress checks
STANDOFF = 0.75        # m from a surface's footprint to park in front of it
SHORT_DRIVE = 1.2      # m: nearer than this, drive straight without mapping first
FAR_TO_PICK = 3.0      # m: further than this, get closer before picking
ROBOT_CLEARANCE = 0.40 # m from the base centre to furniture for a spot the robot fits in
LEAD_IN = 0.8          # m before a skill's parking spot, on its heading, to hand over from
STRAIGHT_OK = 1.5      # m: a skill's own straight approach is trusted up to this far
# m/s top speed while holding something (ApproachPose's cruise). At full speed the
# robot fell over carrying a mug; re-servoing the stowed arm while driving did not
# help and left the grasp planner unable to place afterwards.
CARRY_SPEED = 0.25
GRIPPER_JOINTS = {"left": ("left_left_gripper", "left_right_gripper"),
                  "right": ("right_left_gripper", "right_right_gripper")}
SCENES = {"living_room": REPO / "Hand_and_Wrists" / "scenes" / "scene_home.xml",
          "apartment": REPO / "comp_vision_sim" / "home_search.xml"}
ARMS = {"either": ("right", "left"), "left": ("left",), "right": ("right",)}


def _wrap(a):
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def blocked_by(point, footprints, clearance=ROBOT_CLEARANCE):
    """Name of the first footprint closer than `clearance` to `point`, else None."""
    q = np.asarray(point, float)[:2]
    for name, lo, hi in footprints:
        if float(np.linalg.norm(np.maximum(np.maximum(lo - q, q - hi), 0.0))) < clearance:
            return name
    return None


def nearest_free(point, footprints, toward=None, clearance=ROBOT_CLEARANCE,
                 max_radius=2.0, step=0.1):
    """(spot, blocker): `point` itself if the robot fits there, else the closest
    spot it does -- on the nearest ring that has one, the side nearest `toward`."""
    p = np.asarray(point, float)[:2]
    hit = blocked_by(p, footprints, clearance)
    if hit is None:
        return p, None
    free, first = [], None
    for i in range(1, int(round(max_radius / step)) + 1):
        r = i * step
        if first is not None and r > first + step + 1e-9:
            break
        ring = [p + r * np.array([np.cos(a), np.sin(a)])
                for a in np.linspace(-np.pi, np.pi, 36, endpoint=False)]
        found = [q for q in ring if blocked_by(q, footprints, clearance) is None]
        if found and first is None:
            first = r
        free += found
    if not free:
        return None, hit
    # The first free ring and the next: a spot exactly at the clearance is decided
    # by rounding, and the side facing the robot should not lose to that.
    if toward is None:
        return free[0], hit
    return min(free, key=lambda q: float(np.linalg.norm(q - toward))), hit


def lead_in(base_xy, base_yaw, distance=LEAD_IN):
    """The point `distance` behind a parking spot on its heading, facing it."""
    h = np.array([np.cos(base_yaw), np.sin(base_yaw)])
    return np.asarray(base_xy, float)[:2] - distance * h


def resolve_scene(scene) -> Path:
    if scene is None:
        return SCENES["living_room"]
    return Path(SCENES.get(str(scene), scene)).resolve()


# ------------------------------------------------------------ tool schemas
@dataclass
class Tool:
    name: str
    description: str
    parameters: dict = field(default_factory=dict)   # JSON schema properties
    required: tuple = ()
    physical: bool = True                            # moves the robot

    def spec(self) -> dict:
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": {"type": "object", "properties": self.parameters,
                           "required": list(self.required)}}}


def _str(desc, enum=None):
    out = {"type": "string", "description": desc}
    if enum:
        out["enum"] = list(enum)
    return out


def _num(desc):
    return {"type": "number", "description": desc}


def _bool(desc):
    return {"type": "boolean", "description": desc}


SHAPE = _str("how its size is read from depth: 'round' (upright cylinder: mug, can, bottle), "
             "'box' (remote, keys, book), 'sphere' (ball)", ["round", "box", "sphere"])
GRASP = {
    "object": _str("the object, named as when it was detected, e.g. 'mug'"),
    "grasp": _str("'top': fingers come down from above (most things); 'side': the hand comes in "
                  "horizontally (tall narrow things such as bottles)", ["top", "side"]),
    "arm": _str("which arm; 'either' lets the planner choose", list(ARMS)),
    "align_wrist": _bool("turn the wrist to the object's long axis and grip across its narrow "
                         "side: anything clearly longer than wide (remote, keys, box)"),
    "grip_at": _str("grip just below the top, or at mid-height (balls, round things)",
                    ["top", "center"]),
    "handle": _bool("it has a handle sticking out (mug, cup): keep the fingers clear of it"),
    "shape": SHAPE,
    "squeeze_mm": _num("how far past its width the fingers close, 5-30 (default 20; less for "
                       "small or delicate things)"),
    "looks_like": _str("what the detector should look for when the name alone finds nothing: "
                       "describe it, e.g. 'small grey block on the floor' (optional; remembered "
                       "for this object)"),
}


def tool_catalogue(surfaces=(), joints=()) -> list[Tool]:
    """The tools, independent of any running sim (for schemas and tests)."""
    surface = _str("a surface from list_surfaces", surfaces or None)
    return [
        # --- information
        Tool("get_status", "Where the robot is, which way it faces, whether it is upright, what "
             "each hand holds, the last action's result, and objects measured so far.",
             physical=False),
        Tool("detect_objects", "Run the open-vocabulary detector on the head camera and list what "
             "it finds, with world positions and what each rests on. Name what to look for; "
             "with no names it looks for common household things. Only sees what is in front.",
             {"names": {"type": "array", "items": {"type": "string"},
                        "description": "things to look for, e.g. ['mug', 'tv remote']"}},
             physical=False),
        Tool("look_around", "Turn a full circle on the spot, running the detector every 45 deg, "
             "and list every match found: position, size, what it rests on, distance. The way to "
             "find something in a room, and to see all the candidates before choosing one.",
             {"names": {"type": "array", "items": {"type": "string"},
                        "description": "things to look for, e.g. ['box', 'small block']"}}),
        Tool("describe_view", "Ask the vision model what the head camera sees. Use it for "
             "anything the detector does not name (people, obstacles, where things are). The "
             "camera looks slightly down from 1.5 m and cannot see the floor within about "
             "1.2 m or a table top within about 0.9 m.",
             {"question": _str("what you want to know about the view")}, physical=False),
        Tool("list_surfaces", "Every surface in this scene the robot can put things on or into, "
             "found from its furniture: name, set-on or drop-into, height, centre, size, distance.",
             physical=False),
        Tool("inspect_object", "Measure an object in view for grasping: its centre, width across "
             "the fingers, length, height, long-axis direction, what it rests on, and whether "
             "the gripper opens wide enough. It must be visible from here.",
             {"object": GRASP["object"], "shape": SHAPE, "looks_like": GRASP["looks_like"]},
             ("object",), physical=False),
        Tool("plan_grasp", "Check a grasp without moving: returns the ways the planner found to "
             "do it (arm, where to park, reach, wrist angle, finger gaps), or why there are none. "
             "Use it to compare grasp choices before pick_up.", GRASP, ("object",),
             physical=False),
        # --- manipulation
        Tool("pick_up", "Pick the object up with the grasp you chose: plans, drives to the "
             "parking spot, reaches, closes, checks the grip by touch, lifts and stows. Retries "
             "a failed grasp twice; turns on the spot to look if it is not in view. Reports "
             "each phase it went through and where it failed.", GRASP, ("object",)),
        Tool("place_held_item", "Put down what the robot is holding: set it on a surface, or drop "
             "it into a container. Optionally at a world x, y on that surface.",
             {"surface": surface,
              "mode": _str("'set' on top, 'drop' into; default: what the surface is",
                           ["auto", "set", "drop"]),
              "x": _num("world x to put it at (optional)"),
              "y": _num("world y to put it at (optional)")}, ("surface",)),
        # --- navigation
        Tool("go_to", "Drive to a world coordinate in metres where the robot can stand (not onto "
             "furniture or an object: use go_near for those). Longer drives first spin to map the "
             "room with the depth camera and plan a path round walls and furniture (through "
             "doorways). Optional final heading in degrees (0 = +x, counter-clockwise positive).",
             {"x": _num("x in metres"), "y": _num("y in metres"),
              "heading_deg": _num("final heading in degrees (optional)")}, ("x", "y")),
        Tool("go_near", "Stand about `distance_m` from a point, facing it: the way to get a "
             "good look at an object before inspecting or picking it up. Picks a spot the robot "
             "fits in, on the side nearest the robot where possible.",
             {"x": _num("x of the thing to look at"), "y": _num("y of the thing to look at"),
              "distance_m": _num("how far from it to stand, 0.8-3 (default 1.5)")}, ("x", "y")),
        Tool("go_to_surface", "Drive to about 0.75 m in front of a surface's nearest side, facing "
             "it: close enough to see and reach what is on it.", {"surface": surface},
             ("surface",)),
        Tool("move", "Drive straight forward (positive) or backward (negative) a short distance.",
             {"distance_m": _num("metres, -2 to 2")}, ("distance_m",)),
        Tool("turn", "Turn on the spot: positive is left (counter-clockwise), negative right.",
             {"degrees": _num("degrees to turn, -360 to 360")}, ("degrees",)),
        Tool("search_for", "Search the rooms for something not in view: the robot photographs its "
             "surroundings, the detector checks every photo, the vision model says where to look "
             "next, and it drives there until it finds the object and stops about 1 m away.",
             {"object": _str("what to find, in plain words")}, ("object",)),
        Tool("wait", "Stand still, balancing, for a few seconds.",
             {"seconds": _num("0 to 30")}, ("seconds",)),
        # --- direct arm control
        Tool("set_gripper", "Open or close a gripper directly. pick_up and place_held_item grasp "
             "and release properly; opening a hand that holds something drops it.",
             {"side": _str("which hand", ["left", "right", "both"]),
              "action": _str("what to do", ["open", "close"])}, ("side", "action")),
        Tool("move_arm_joint", "Move one arm joint and wait for it. Joint 0 of each arm (lj0 / "
             "rj0) is the lift in metres, the others radians. Out-of-range values are clamped.",
             {"joint": _str("joint name", joints or None), "value": _num("target position")},
             ("joint", "value")),
        Tool("stow_arms", "Return both arms to their starting pose (keeps any grip)."),
    ]


# ------------------------------------------------------- small controllers
class _Straight:
    """Drive a signed distance along the current heading.

    Braking is ApproachPose's: pin the balance reference on the stopping point
    and wait until the base stays still. Switching between cruise and creep
    speeds instead limit-cycled +/-0.2 m around the goal after a turn.
    """

    def __init__(self, bot, distance, tol=0.05):
        from handwrist.skills import ApproachPose
        self.start, self.heading = bot.position[:2].copy(), float(bot.yaw)
        self.goal = self.start + float(distance) * np.array([np.cos(self.heading),
                                                             np.sin(self.heading)])
        self._pose = ApproachPose(self.goal, self.heading)
        self.tol = tol
        self.done = False

    def travelled(self, bot):
        u = np.array([np.cos(self.heading), np.sin(self.heading)])
        return float((bot.position[:2] - self.start) @ u)

    def __call__(self, bot, t):
        if not self.done:
            self.done = self._pose._drive(bot, self.goal, line_yaw=self.heading, tol=self.tol)
        else:
            bot.drive(0.0, 0.0)


class _Turn:
    """Turn on the spot by any angle, in legs under 180 deg (yaw wraps)."""

    def __init__(self, bot, degrees):
        from handwrist.skills import turn_in_place
        self._turn = turn_in_place
        total = np.deg2rad(float(degrees))
        n = max(1, int(np.ceil(abs(total) / np.deg2rad(170))))
        self.targets = [_wrap(bot.yaw + total * (i + 1) / n) for i in range(n)]
        self.x_hold = float(bot.state[0])
        self.done = False

    def __call__(self, bot, t):
        if self.targets and self._turn(bot, self.targets[0], self.x_hold):
            self.targets.pop(0)
        self.done = not self.targets


class _Hold:
    def __init__(self, seconds):
        self.seconds, self.t0, self.done = float(seconds), None, False

    def __call__(self, bot, t):
        self.t0 = t if self.t0 is None else self.t0
        bot.drive(0.0, 0.0)
        self.done = t - self.t0 >= self.seconds


class _Near:
    """A controller, finished early once the base is within `tol` and slow."""

    def __init__(self, ctl, goal, tol=0.2):
        self.ctl, self.goal, self.tol, self.done = ctl, np.asarray(goal, float), tol, False

    def __call__(self, bot, t):
        self.ctl(bot, t)
        self.done = bool(self.ctl.done) or (
            float(np.linalg.norm(bot.position[:2] - self.goal)) < self.tol
            and bot.ground_speed < 0.1)


class _UntilDone:
    """Run a navigator that holds station once finished, and stop there."""

    def __init__(self, nav):
        self.nav, self.done = nav, False

    def __call__(self, bot, t):
        self.nav(bot, t)
        self.done = bool(self.nav.done)


def object_spec(object, grasp="top", align_wrist=False, grip_at="top", handle=False,  # noqa: A002
                shape="round", squeeze_mm=20.0, arm="either"):
    """The model's grasp choices as an ObjectSpec, validated with readable errors."""
    from handwrist.objects import ObjectSpec
    name = str(object).strip()
    if not name:
        raise ValueError("name the object")
    for value, allowed, what in ((grasp, ("top", "side"), "grasp"),
                                 (grip_at, ("top", "center"), "grip_at"),
                                 (shape, ("round", "box", "sphere"), "shape"),
                                 (arm, tuple(ARMS), "arm")):
        if value not in allowed:
            raise ValueError(f"{what} must be one of {', '.join(allowed)}, not {value!r}")
    return ObjectSpec(name=name, grasp_geom="", grasp=grasp, aligned=bool(align_wrist),
                      handle_geom="handle" if handle else None,
                      grip_at_center=grip_at == "center",
                      squeeze=float(np.clip(float(squeeze_mm), 5.0, 30.0)) / 1000.0, shape=shape)


# ------------------------------------------------------------------ tools
class RobotTools:
    """One BracketBot in a scene, and the tools that drive it."""

    def __init__(self, scene=None, bot=None, truth: bool = False, detector: str = "auto",
                 llm_url: str = "http://localhost:8080/v1", llm_model: str = "Qwen/Qwen3.8-27B",
                 on_step=None, verbose: bool = True, task_seconds: float = 360.0,
                 drive_seconds: float = 240.0, search_seconds: float = 400.0):
        import mujoco

        import handwrist
        from handwrist import scenarios
        from handwrist.detection import DetectionEstimator, TruthByName
        from handwrist.gripper import Gripper
        from handwrist.places import reset_drops
        from handwrist.surfaces import find_surfaces

        self._mujoco = mujoco
        self.scene = resolve_scene(scene)
        if bot is None:
            from bracketbot_sim.robot import BracketBot
            bot = BracketBot(xml=str(self.scene))
            bot.reset()
            reset_drops()
            if self.scene == Path(handwrist.HOME_SCENE).resolve():
                scenarios.place_robot(bot, [0.0, 0.0], 0.0)
            mujoco.mj_forward(bot.model, bot.data)
            bot.balance.enable(bot.state)
        self.bot = bot
        self.truth = truth
        self.detection = DetectionEstimator(backend=detector, llm_url=llm_url, llm_model=llm_model)
        self.estimator = TruthByName() if truth else self.detection
        self.llm_url, self.llm_model = llm_url, llm_model
        self.on_step = on_step
        self.verbose = verbose
        self.task_seconds, self.drive_seconds = task_seconds, drive_seconds
        self.search_seconds = search_seconds
        self.arm_joints = [j for j in bot.arm_joints if "gripper" not in j]
        self.home = {j: float(bot.joint_position(j)) for j in self.arm_joints}
        self.grippers = {s: Gripper(bot, s) for s in ("left", "right")}
        self.max_gap = float(self.grippers["right"].max_gap)
        self.surfaces = {s.name: s for s in find_surfaces(bot.model, bot.data)}
        self.estimates = {}          # object name -> last ObjectEstimate
        self.held = None             # the successful Pick whose object is in hand
        self.last_result = None
        self._tools = {t.name: t for t in tool_catalogue(list(self.surfaces), self.arm_joints)}

    # ---------------------------------------------------------- plumbing
    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    @property
    def names(self):
        return list(self._tools)

    def call(self, name: str, arguments) -> dict:
        """Run a tool by name. Never raises: failures come back as ok=False."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"no tool called {name!r}; tools: {', '.join(self._tools)}"}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"arguments are not valid JSON: {e}"}
        arguments = {k: v for k, v in (arguments or {}).items() if v is not None}
        if tool.physical and self.bot.fallen:
            return {"ok": False, "error": "the robot has fallen over and cannot move; "
                                          "a person has to stand it back up"}
        try:
            result = getattr(self, f"tool_{name}")(**arguments)
        except (ValueError, KeyError, TypeError) as e:
            msg = e.args[0] if isinstance(e, KeyError) and e.args else str(e)
            result = {"ok": False, "error": str(msg)}
        except Exception as e:  # a skill blew up: report it, keep the session alive
            result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        result.setdefault("ok", True)
        result["robot"] = self.pose()
        if tool.physical:
            self.last_result = {"tool": name, "arguments": arguments,
                                **{k: v for k, v in result.items() if k != "robot"}}
        return result

    def _run(self, controller, limit: float) -> dict:
        bot, t0 = self.bot, self.bot.time
        while not controller.done and bot.time - t0 < limit and not bot.fallen:
            bot.step(CHUNK, controller=controller)
            if self.on_step is not None:
                self.on_step(bot)
        out = {"sim_seconds": round(bot.time - t0, 1)}
        if not controller.done and not bot.fallen:
            out["timed_out"] = True
        if bot.fallen:
            out["fallen"] = True
        return out

    def _settle(self):
        """Stop and let the balance loop come to rest after a manoeuvre."""
        self.bot.drive(0.0, 0.0)
        self._run(_Hold(0.3), 1.0)

    def pose(self) -> dict:
        b = self.bot
        return {"x": round(float(b.position[0]), 2), "y": round(float(b.position[1]), 2),
                "heading_deg": round(float(np.rad2deg(b.yaw))), "fallen": bool(b.fallen)}

    def holding(self) -> dict:
        m, out = self.bot.model, {}
        for side, g in self.grippers.items():
            body = int(g.pinched_body())
            out[side] = (None if body < 0 else
                         self._mujoco.mj_id2name(m, self._mujoco.mjtObj.mjOBJ_BODY, body)
                         or "something")
        if self.held is not None and out.get(self.held.plan.side):
            out[self.held.plan.side] = self.held.spec.name
        return out

    def _surface(self, name):
        if name not in self.surfaces:
            raise ValueError(f"no surface called {name!r}; surfaces here: "
                             f"{', '.join(self.surfaces) or 'none'}")
        return self.surfaces[name]

    def _rests_on(self, xy, bottom_z) -> str:
        for s in self.surfaces.values():
            if s.surface.contains(np.asarray(xy)[:2], 0.03) and abs(bottom_z - s.top) < 0.06:
                return s.name
        return "floor" if bottom_z < 0.05 else f"something {bottom_z:.2f} m up"

    def world_summary(self) -> str:
        lines = [f"Scene: {self.scene.name}. Surfaces (name: set on / drop into, top height, "
                 "centre x, y):"]
        for s in self.surfaces.values():
            c = s.surface.center
            lines.append(f"  {s.name}: {'drop into' if s.mode == 'drop' else 'set on'}, "
                         f"{s.top:.2f} m, ({c[0]:.1f}, {c[1]:.1f}), "
                         f"{s.size[0]:.2f} x {s.size[1]:.2f} m")
        lines.append(f"The gripper opens to {self.max_gap * 1000:.0f} mm, so it holds things up "
                     f"to about {(self.max_gap - 0.02) * 1000:.0f} mm across.")
        lines.append(f"Robot starts at {self.pose()}. Arm joints: {', '.join(self.arm_joints)}.")
        return "\n".join(lines)

    def _hands_free(self):
        if self.held is not None and any(self.holding().values()):
            raise ValueError(f"already holding the {self.held.spec.name}; put it down first "
                             "with place_held_item")
        self.held = None

    def _estimate(self, spec):
        est = self.estimator(self.bot, spec)
        if est is not None:
            self.estimates[spec.name] = est
        return est

    def _describe_estimate(self, est) -> dict:
        c = est.center
        return {"center": {"x": round(float(c[0]), 3), "y": round(float(c[1]), 3),
                           "z": round(float(c[2]), 3)},
                "width_mm": round(est.width * 1000), "length_mm": round(est.length * 1000),
                "height_mm": round(est.height * 1000),
                "axis_deg": None if est.axis_yaw is None else round(float(np.rad2deg(est.axis_yaw))),
                "rests_on": self._rests_on(c, est.bottom_z),
                "distance_m": round(float(np.linalg.norm(c[:2] - self.bot.position[:2])), 2),
                "fits_gripper": bool(est.width + 0.02 <= self.max_gap)}

    def _not_seen(self, name) -> dict:
        box = None if self.truth else self.detection.last_box
        if box is not None:
            why = (f"the detector boxed the {name} but got too little depth on it: it is probably "
                   "too close (camera blind zone) or too far; move back or closer")
        elif self.detection.yolo_error and not self.truth:
            why = f"YOLO is unavailable ({self.detection.yolo_error}) and the vision model did not find it"
        else:
            why = (f"can't see the {name} from here; turn, move, search_for it, or describe what "
                   "it looks like with looks_like")
        return {"ok": False, "error": why}

    # ------------------------------------------------------------- tools
    def tool_get_status(self) -> dict:
        return {"scene": self.scene.name, "holding": self.holding(),
                "sim_time": round(self.bot.time, 1), "last_action": self.last_result,
                "measured": {n: self._describe_estimate(e)["center"]
                             for n, e in self.estimates.items()}}

    def tool_detect_objects(self, names=None) -> dict:
        from vision_sim.perception import observe
        from vision_sim.yolo_detector import HOUSEHOLD_CLASSES
        if isinstance(names, str):
            names = [names]
        names = [str(n).strip() for n in (names or []) if str(n).strip()] or list(HOUSEHOLD_CLASSES)
        det = self.detection
        det._bot = self.bot
        obs = observe(self.bot, det.camera, det.width, det.height, 8.0)
        seen = []
        for hit in det.yolo_boxes(obs.rgb, names)[:15]:
            where = det.locate(obs, hit)
            if where is None:
                continue
            c = where["center"]
            rel = c[:2] - self.bot.position[:2]
            seen.append({"object": hit["name"], "confidence": round(hit["confidence"], 2),
                         "x": round(float(c[0]), 2), "y": round(float(c[1]), 2),
                         "z": round(float(c[2]), 2),
                         "size_cm": [round(float(v) * 100) for v in where["size"]],
                         "rests_on": self._rests_on(c, where["bottom_z"]),
                         "distance_m": round(float(np.linalg.norm(rel)), 2),
                         "bearing_deg": round(float(np.rad2deg(_wrap(
                             np.arctan2(rel[1], rel[0]) - self.bot.yaw))))})
        out = {"seen": seen, "looked_for": names if len(names) <= 10 else "household things"}
        if not seen:
            out["hint"] = ("nothing found in view: turn to look elsewhere, step back if it may be "
                           "close below the camera, or ask describe_view")
        return out

    def tool_look_around(self, names=None) -> dict:
        found = []
        for i in range(8):
            for s in self.tool_detect_objects(names)["seen"]:
                same = next((f for f in found if f["object"] == s["object"] and
                             np.hypot(f["x"] - s["x"], f["y"] - s["y"]) < 0.3), None)
                if same is None:
                    found.append(s)
                elif s["confidence"] > same["confidence"]:
                    same.update(s)
            if i < 7:
                self._run(_Turn(self.bot, 45.0), 9.0)
                self._settle()
        if self.bot.fallen:
            return {"ok": False, "error": "fell over while turning", "seen": found}
        here = self.bot.position[:2]
        for f in found:
            f.pop("bearing_deg", None)
            f["distance_m"] = round(float(np.hypot(f["x"] - here[0], f["y"] - here[1])), 2)
        return {"seen": sorted(found, key=lambda f: f["distance_m"]),
                "looked_for": names or "household things"}

    def tool_describe_view(self, question: str = "") -> dict:
        from vision_sim.llm_reasoner import LLMClient
        prompt = ("You are the eyes of a home robot. This is its head camera, looking ahead and "
                  "slightly down. Describe briefly what is in view -- objects, furniture, people, "
                  "open floor -- and roughly where (left/centre/right, near/far). Its own white "
                  "arms may be in frame; ignore them. Answer in at most four sentences.")
        if question:
            prompt += f"\nAlso answer this: {question}"
        client = LLMClient(self.llm_url, self.llm_model, max_tokens=400, timeout=180)
        text, dt = client.ask_image(self.bot.camera("head_rgb", 640, 480), prompt)
        return {"description": text.strip(), "seconds": round(dt, 1)}

    def tool_list_surfaces(self) -> dict:
        here = self.bot.position[:2]
        rows = []
        for s in self.surfaces.values():
            c = s.surface.center
            rows.append({"surface": s.name, "mode": s.mode, "height_m": round(s.top, 2),
                         "center": {"x": round(float(c[0]), 2), "y": round(float(c[1]), 2)},
                         "size_m": [round(float(v), 2) for v in s.size],
                         "distance_m": round(float(np.linalg.norm(c[:2] - here)), 2)})
        return {"surfaces": sorted(rows, key=lambda r: r["distance_m"])}

    def tool_inspect_object(self, object: str, shape: str = "round",  # noqa: A002
                            looks_like: str | None = None) -> dict:
        spec = object_spec(object, shape=shape)
        self._alias(spec.name, looks_like)
        est = self._estimate(spec)
        if est is None:
            return self._not_seen(spec.name)
        out = {"object": spec.name, **self._describe_estimate(est),
               "gripper_opens_mm": round(self.max_gap * 1000),
               "seen_by": "ground truth" if self.truth else (self.detection.last_box or {}).get("source")}
        if not out["fits_gripper"]:
            out["hint"] = "wider than the gripper opens: it cannot be picked up with one hand"
        return out

    def _alias(self, name, looks_like):
        if looks_like and str(looks_like).strip():
            self.detection.aliases[name] = str(looks_like).strip()

    def _grasp_args(self, kw):
        arm = kw.pop("arm", "either")
        looks_like = kw.pop("looks_like", None)
        spec = object_spec(arm=arm, **kw)
        self._alias(spec.name, looks_like)
        return spec, ARMS[arm]

    def tool_plan_grasp(self, **kw) -> dict:
        from handwrist.grasping import GraspPlanner
        spec, sides = self._grasp_args(kw)
        est = self._estimate(spec) or self.estimates.get(spec.name)
        if est is None:
            return self._not_seen(spec.name)
        planner = GraspPlanner(self.bot)
        plans = planner.plan(est, spec, sides)
        here = self.bot.position[:2]
        out = {"object": self._describe_estimate(est)}
        if not plans:
            if est.width + 0.02 > planner.max_gap:
                why = (f"{est.width * 1000:.0f} mm wide, more than the gripper can close round "
                       f"({(planner.max_gap - 0.02) * 1000:.0f} mm)")
            elif self._rests_on(est.center, est.bottom_z) not in ("floor",):
                why = ("no reachable grasp from any side: it may be too far in from every edge "
                       f"(top grasps reach {planner.MAX_REACH['top']:.2f} m, side grasps "
                       f"{planner.MAX_REACH['side']:.2f} m past the edge), or the arm cannot get "
                       "the hand there at this height; try the other grasp or arm")
            else:
                why = "the arm cannot reach it with this grasp from any heading; try another grasp"
            return {"ok": False, "error": why, **out}
        out["plans"] = [{"arm": p.side, "grasp": p.kind,
                         "park_at": {"x": round(float(p.base_xy[0]), 2),
                                     "y": round(float(p.base_xy[1]), 2),
                                     "heading_deg": round(float(np.rad2deg(p.base_yaw)))},
                         "drive_m": round(float(np.linalg.norm(p.base_xy - here)), 2),
                         "reach_m": round(float(p.reach), 2),
                         "wrist_deg": round(float(np.rad2deg(p.wrist_yaw))),
                         "fingers_mm": [round(p.open_gap * 1000), round(p.close_gap * 1000)]}
                        for p in plans[:3]]
        return out

    def tool_pick_up(self, **kw) -> dict:
        from handwrist.skills import Pick
        self._hands_free()
        spec, sides = self._grasp_args(kw)
        est = self._estimate(spec)
        if est is not None:
            far = float(np.linalg.norm(est.center[:2] - self.bot.position[:2]))
            if far > FAR_TO_PICK:
                raise ValueError(f"the {spec.name} is {far:.1f} m away; the last-metre approach "
                                 "does not steer round furniture, so go_to (or go_to_surface) "
                                 "within about 2 m first")
        lead = None
        if est is not None:
            from handwrist.grasping import GraspPlanner
            plans = GraspPlanner(self.bot).plan(est, spec, sides)
            if plans:
                lead = self._drive_to_lead_in(plans[0].base_xy, plans[0].base_yaw)
        pick = Pick(self.bot, spec, estimator=self.estimator, sides=sides, verbose=self.verbose)
        run = self._run(pick, self.task_seconds)
        if pick.succeeded:
            self.held = pick
        phases = []
        for _, phase in pick.history:
            if not phases or phases[-1] != phase:
                phases.append(phase)
        out = {"ok": bool(pick.succeeded), "status": pick.status, "phases": phases,
               "retries": pick.retries, **run}
        if lead is not None:
            out["drove_to_lead_in"] = bool(lead.get("ok"))
        if pick.plan is not None:
            out["grasp"] = pick.plan.describe()
        if pick.failure:
            out["failure"] = pick.failure
        return out

    def tool_place_held_item(self, surface: str, mode: str = "auto", x: float | None = None,
                             y: float | None = None) -> dict:
        from handwrist.place import Place
        from handwrist.surfaces import obstacle_boxes
        if self.held is None or not any(self.holding().values()):
            self.held = None
            raise ValueError("not holding anything; pick_up something first")
        found = self._surface(surface)
        if mode not in ("auto", "set", "drop"):
            raise ValueError("mode must be auto, set or drop")
        at = (x, y) if x is not None and y is not None else None
        spec = found.spec(at=at, mode=None if mode == "auto" else mode,
                          obstacles=obstacle_boxes(self.bot.model, self.bot.data,
                                                   exclude_body=found.body))
        carry = self._carry_height()
        if found.mode == "set" and found.surface.rim + 0.02 > carry:
            raise ValueError(
                f"the {found.name} ({found.surface.rim:.2f} m) is too high to reach over: the "
                f"object is carried at {carry:.2f} m and the hand would hit its edge; choose a "
                "lower surface")
        skill = Place(self.bot, self.held, spec, verbose=self.verbose)
        targets = skill.planner.plan(spec, skill.side, skill.rel_mat, skill.grip_above_bottom,
                                     skill.kind)
        lead = None
        if targets:
            lead = self._drive_to_lead_in(targets[0].base_xy, targets[0].base_yaw, found.body)
            if lead is not None and lead.get("ok"):
                skill.phase = "plan"             # already clear of where it picked up
        run = self._run(skill, self.task_seconds)
        if skill.succeeded or not any(self.holding().values()):
            self.held = None
        out = {"ok": bool(skill.succeeded), "status": skill.status, **run}
        if lead is not None:
            out["drove_to_lead_in"] = bool(lead.get("ok"))
        if skill.target is not None:
            out["plan"] = skill.target.describe()
        if skill.failure:
            out["failure"] = skill.failure
        return out

    def _speed(self, normal):
        return min(normal, CARRY_SPEED) if self.held is not None else normal

    def _footprints(self):
        from handwrist.surfaces import obstacle_footprints
        return obstacle_footprints(self.bot.model, self.bot.data)

    def tool_go_to(self, x: float, y: float, heading_deg: float | None = None) -> dict:
        from bracketbot_sim.algorithms import NavigateTo
        from vision_sim.navigation import VisualNavigator
        goal = np.array([float(x), float(y)])
        spot, blocker = nearest_free(goal, self._footprints(), toward=self.bot.position[:2])
        if blocker is not None:
            msg = f"({goal[0]:.2f}, {goal[1]:.2f}) is too close to the {blocker} for the robot to stand"
            if spot is not None:
                msg += (f"; the nearest spot it fits is ({spot[0]:.2f}, {spot[1]:.2f}). To look at "
                        "or reach something there, use go_near")
            return {"ok": False, "error": msg}
        out = {}
        from handwrist.places import route_clear
        here = self.bot.position[:2]
        boxes = [(lo, hi) for _, lo, hi in self._footprints()]
        # The straight-line drive only steers round what the camera sees ahead; past a
        # corner it clipped the cartons and fell. Anything near the line: plan with A*.
        if (np.linalg.norm(goal - here) < SHORT_DRIVE
                and route_clear([here, goal], boxes, 0.35, skip_start=0.15)):
            run = self._run(_Near(NavigateTo(goal, v_max=self._speed(0.4)), goal), 60.0)
        else:
            nav = VisualNavigator.for_bot(self.bot, goal=goal, stop_distance=0.0,
                                          v_max=self._speed(0.42), verbose=self.verbose)
            run = self._run(_UntilDone(nav), self.drive_seconds)
            if nav.state == nav.STUCK:
                out["why"] = nav.give_up_reason or "no path found"
        self._settle()
        if heading_deg is not None:
            turn = np.rad2deg(_wrap(np.deg2rad(float(heading_deg)) - self.bot.yaw))
            self._run(_Turn(self.bot, turn), 6.0 + abs(turn) / 20.0)
            self._settle()
        miss = float(np.linalg.norm(self.bot.position[:2] - goal))
        return {"ok": miss < 0.35 and not run.get("fallen"),
                "distance_from_goal_m": round(miss, 2), **out, **run}

    def tool_go_near(self, x: float, y: float, distance_m: float = 1.5) -> dict:
        target = np.array([float(x), float(y)])
        d = float(np.clip(distance_m, 0.8, 3.0))
        here = self.bot.position[:2]
        away = here - target
        base = (float(np.arctan2(away[1], away[0])) if np.linalg.norm(away) > 1e-3
                else _wrap(self.bot.yaw + np.pi))
        footprints = self._footprints()
        for offset in np.deg2rad([0, 20, -20, 40, -40, 60, -60, 90, -90, 120, -120, 150, -150, 180]):
            a = base + offset
            spot = target + d * np.array([np.cos(a), np.sin(a)])
            if blocked_by(spot, footprints) is None:
                facing = np.rad2deg(np.arctan2(*(target - spot)[::-1]))
                result = self.tool_go_to(spot[0], spot[1], facing)
                result["stood_at_m"] = d
                return result
        return {"ok": False, "error": f"no free spot {d:.1f} m from ({target[0]:.2f}, "
                                      f"{target[1]:.2f}); try another distance"}

    def _carry_height(self) -> float:
        """How high the held object's hand is carried (Pick's stow height).

        A 0.76 m dining table stopped the stowed hand at its edge, 10 cm short
        of parking; carrying higher to clear it made the robot fall over. So
        surfaces at or above this are refused rather than attempted."""
        from handwrist.skills import STOW_Z
        plan = getattr(self.held, "plan", None)
        return max(STOW_Z, float(plan.lift_pos[2])) if plan is not None else STOW_Z

    def _drive_to_lead_in(self, base_xy, base_yaw, target_body=None) -> dict | None:
        """Get the skill's last metre down to a straight line.

        Pick and Place park with ApproachPose, which drives straight and never
        detours: from beside a dining table every route to a spot on its far
        side ran through the table and timed out. When the spot is far, or the
        straight route clips furniture, the A* navigator first takes the robot to
        a lead-in point in front of it. None if no drive was needed.
        """
        from handwrist.places import route_clear
        here = self.bot.position[:2]
        start = lead_in(base_xy, base_yaw)
        boxes = [(lo, hi) for name, lo, hi in self._footprints() if name != target_body]
        if self.held is not None:
            # Carrying, the skill's own slow approach (it backs away first) is the
            # safer drive: a navigator lead-in knocked the robot over with a can,
            # triggered by the coffee table it had just picked from. Only a wall
            # between here and the spot -- a room divider -- is worth it then.
            boxes = [(lo, hi) for lo, hi in boxes if float(np.max(hi - lo)) >= 1.5]
            if route_clear([here, start, base_xy], boxes, 0.28, skip_start=0.3):
                return None
        else:
            far = float(np.linalg.norm(np.asarray(base_xy) - here)) > STRAIGHT_OK
            if not far and route_clear([here, start, base_xy], boxes, 0.28, skip_start=0.3):
                return None
        if blocked_by(start, self._footprints()) is not None:
            # e.g. the lead-in falls inside the wall a doorway goes through: stand
            # at the nearest free spot instead, if it is still close to the spot
            spot, _ = nearest_free(start, self._footprints(), toward=np.asarray(base_xy))
            if spot is None or float(np.linalg.norm(spot - np.asarray(base_xy))) > STRAIGHT_OK:
                return None
            start = spot
        facing = np.rad2deg(np.arctan2(*(np.asarray(base_xy) - start)[::-1])) \
            if float(np.linalg.norm(np.asarray(base_xy) - start)) > 0.2 else np.rad2deg(base_yaw)
        return self.tool_go_to(start[0], start[1], facing)

    def standoff(self, surface: str):
        """A spot STANDOFF m in front of a surface's nearest side, and the heading facing it."""
        from handwrist.places import footprint_of
        found = self._surface(surface)
        lo, hi = footprint_of(self.bot.model, self.bot.data, found.body)
        here = self.bot.position[:2]
        edge = np.clip(here, lo, hi)
        away = here - edge
        if np.linalg.norm(away) < 1e-3:              # standing over it: back out the near side
            centre = (lo + hi) / 2
            away = here - centre if np.linalg.norm(here - centre) > 1e-3 else np.array([-1.0, 0.0])
        away = away / np.linalg.norm(away)
        goal = edge + away * STANDOFF
        return goal, float(np.arctan2(-away[1], -away[0]))

    def tool_go_to_surface(self, surface: str) -> dict:
        goal, yaw = self.standoff(surface)
        result = self.tool_go_to(goal[0], goal[1], np.rad2deg(yaw))
        result["surface"] = surface
        return result

    def tool_move(self, distance_m: float) -> dict:
        d = float(np.clip(distance_m, -2.0, 2.0))
        ctl = _Straight(self.bot, d)
        run = self._run(ctl, 10.0 + 8.0 * abs(d))
        self._settle()
        moved = ctl.travelled(self.bot)
        return {"ok": abs(moved - d) < 0.15 and not run.get("fallen"),
                "moved_m": round(moved, 2), **run}

    def tool_turn(self, degrees: float) -> dict:
        deg = float(np.clip(degrees, -360.0, 360.0))
        start = self.bot.yaw
        ctl = _Turn(self.bot, deg)
        run = self._run(ctl, 6.0 + abs(deg) / 20.0)
        self._settle()
        return {"ok": bool(ctl.done), "turned_deg": round(float(np.rad2deg(
            _wrap(self.bot.yaw - start)))), **run}

    def tool_search_for(self, object: str) -> dict:  # noqa: A002 (the model's word for it)
        import run_navigation as rn
        from vision_sim.navigation import VisualNavigator
        from vision_sim.scene import SceneInfo
        detector = "yolo" if importlib.util.find_spec("ultralytics") else "llm"
        args = rn.parse_args(["--explore", "--target", str(object), "--detector", detector,
                              "--llm-url", self.llm_url, "--llm-model", self.llm_model])
        args.task = None
        info = SceneInfo.from_model(self.bot.model, self.bot.data)
        nav = VisualNavigator.for_bot(
            self.bot, scan_turns=args.seed_scan, verbose=self.verbose,
            detector=rn.build_detector(args, info), explorer=rn.build_explorer(args),
            max_explore_steps=rn.resolve_rounds(args), track=rn.resolve_track(args),
            lost_after=args.lost_after, max_researches=args.max_researches)
        run = self._run(_UntilDone(nav), self.search_seconds)
        self._settle()
        out = {"ok": nav.outcome == "found", "outcome": nav.outcome or "unfinished",
               "detector": detector,
               "places_explored": len(getattr(nav, "explored", []) or []), **run}
        if nav.goal_xy is not None:
            out["object_at"] = {"x": round(float(nav.goal_xy[0]), 2),
                                "y": round(float(nav.goal_xy[1]), 2)}
            out.update(self._describe_found(str(object), nav.goal_xy))
        if nav.give_up_reason:
            out["why"] = nav.give_up_reason
        return out

    def _describe_found(self, name, xy) -> dict:
        """Size and support of what a search found: words match more than the thing
        meant (a stack of cartons is a 'box'), and size tells them apart."""
        from vision_sim.perception import observe
        det = self.detection
        det._bot = self.bot
        try:
            obs = observe(self.bot, det.camera, det.width, det.height, 8.0)
            best = None
            for hit in det.yolo_boxes(obs.rgb, [name])[:5]:
                where = det.locate(obs, hit)
                if where is not None:
                    gap = float(np.linalg.norm(where["center"][:2] - np.asarray(xy)[:2]))
                    if gap < 0.6 and (best is None or gap < best[0]):
                        best = (gap, where)
        except Exception:        # no YOLO here: the position alone still stands
            return {}
        if best is None:
            return {"hint": "not in the camera's view from here (probably below it); go_near "
                            "it from further back, then inspect_object"}
        where = best[1]
        size = where["size"]
        return {"size_cm": [round(float(v) * 100) for v in size],
                "rests_on": self._rests_on(where["center"], where["bottom_z"]),
                "fits_gripper": bool(float(min(size[0], size[1])) + 0.02 <= self.max_gap)}

    def tool_wait(self, seconds: float) -> dict:
        return self._run(_Hold(float(np.clip(seconds, 0.0, 30.0))), 31.0)

    def tool_set_gripper(self, side: str, action: str) -> dict:
        sides = ("left", "right") if side == "both" else (side,)
        if any(s not in GRIPPER_JOINTS for s in sides):
            raise ValueError("side must be left, right or both")
        if action not in ("open", "close"):
            raise ValueError("action must be open or close")
        for s in sides:
            for joint in GRIPPER_JOINTS[s]:
                self.bot.set_arm_target(joint, 0.0 if action == "open" else 1.0)
        run = self._run(_Hold(1.0), 2.0)
        if action == "open" and self.held is not None and self.held.plan.side in sides:
            self.held = None
        return {"holding": self.holding(), **run}

    def tool_move_arm_joint(self, joint: str, value: float) -> dict:
        if joint not in self.arm_joints:
            raise ValueError(f"no arm joint {joint!r}; joints: {', '.join(self.arm_joints)}")
        j = self.bot._jnt_id(joint)
        lo, hi = (float(v) for v in self.bot.model.jnt_range[j])
        target = float(np.clip(value, lo, hi))
        self.bot.set_arm_target(joint, target)
        run = self._run(_Hold(2.0), 3.0)
        return {"target": round(target, 3), "position": round(float(self.bot.joint_position(joint)), 3),
                "range": [round(lo, 3), round(hi, 3)], **run}

    def tool_stow_arms(self) -> dict:
        for joint, q in self.home.items():
            self.bot.set_arm_target(joint, q)
        return self._run(_Hold(2.5), 3.0)

    def close(self):
        self.bot.close()
