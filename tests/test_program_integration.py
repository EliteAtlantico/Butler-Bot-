from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mujoco
import numpy as np
import pytest

from vision_sim import perception


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_mujoco"
VISION = ROOT / "comp_vision_sim"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_sim = load_script("run_sim_integration", MAIN / "run.py")
run_nav = load_script("run_nav_integration", VISION / "run_navigation.py")
train = load_script("train_yolo_integration", VISION / "train_yolo.py")


class FakeBalance:
    def __init__(self):
        self.enabled = []
    def enable(self, state): self.enabled.append(state)


class FakePlant:
    def describe(self): return "plant"


class FakeRunBot:
    instances = []

    def __init__(self, xml):
        self.xml = xml; self.time = 0.; self.fallen = False
        self.balance = FakeBalance(); self.state = np.zeros(6)
        self.plant = FakePlant(); self.camera_names = ["cam"]
        self.position = np.array([0., 0., 0.]); self.pitch = 0.; self.closed = False
        self.model = SimpleNamespace(nbody=0, body_mocapid=np.array([], int),
                                     body_pos=np.empty((0, 3)))
        self.data = SimpleNamespace(mocap_pos=np.empty((0, 3)))
        self.step_calls = []
        self.__class__.instances.append(self)

    def step(self, duration, controller):
        self.step_calls.append((duration, controller)); self.time += duration

    def reset(self): self.time = 0
    def close(self): self.closed = True


def run_args(**changes):
    base = dict(algorithm="stand", scene=None, headless=True, duration=.2, speed=100,
                no_balance=False, cameras=False, shot=None)
    base.update(changes)
    return SimpleNamespace(**base)


def test_run_main_headless_executes_algorithm_and_closes(monkeypatch, capsys):
    import bracketbot_sim.robot as module
    FakeRunBot.instances.clear()
    algorithm = SimpleNamespace(done=False)
    monkeypatch.setattr(run_sim, "parse_args", lambda: run_args())
    monkeypatch.setattr(run_sim, "make_algorithm", lambda name, bot: algorithm)
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    run_sim.main()
    bot = FakeRunBot.instances[-1]
    assert bot.xml == "scene_dynamic.xml" and bot.closed
    assert bot.balance.enabled and len(bot.step_calls) == 2
    assert all(call[1] is algorithm for call in bot.step_calls)
    assert "scene: scene_dynamic.xml" in capsys.readouterr().out


def test_run_main_pick_default_scene_and_done_break(monkeypatch):
    import bracketbot_sim.robot as module
    FakeRunBot.instances.clear()

    class DoneAfterStep:
        done = True

    monkeypatch.setattr(run_sim, "parse_args", lambda: run_args(algorithm="pick", duration=None))
    monkeypatch.setattr(run_sim, "make_algorithm", lambda *_a: DoneAfterStep())
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    run_sim.main()
    bot = FakeRunBot.instances[-1]
    assert bot.xml == "scene_table.xml" and len(bot.step_calls) == 1


def test_run_main_camera_mode_calls_montage_without_enabling_balance(monkeypatch):
    import bracketbot_sim.robot as module
    FakeRunBot.instances.clear(); calls = []
    monkeypatch.setattr(run_sim, "parse_args", lambda: run_args(cameras=True, shot=None))
    monkeypatch.setattr(run_sim, "montage", lambda bot, path: calls.append((bot, path)))
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    run_sim.main()
    bot = FakeRunBot.instances[-1]
    assert calls == [(bot, "cameras.png")]
    assert not bot.balance.enabled and bot.closed


def test_run_main_no_balance_and_shot(monkeypatch):
    import bracketbot_sim.robot as module
    FakeRunBot.instances.clear(); shots = []
    monkeypatch.setattr(run_sim, "parse_args", lambda: run_args(no_balance=True, shot="end.png"))
    monkeypatch.setattr(run_sim, "make_algorithm", lambda *_a: SimpleNamespace(done=True))
    monkeypatch.setattr(run_sim, "montage", lambda bot, path: shots.append(path))
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    run_sim.main()
    assert not FakeRunBot.instances[-1].balance.enabled
    assert shots == ["end.png"]


class FakeViewer:
    def __init__(self, running=(True, False)):
        self.running = iter(running); self.syncs = 0
    def __enter__(self): return self
    def __exit__(self, *_exc): return False
    def is_running(self): return next(self.running, False)
    def sync(self): self.syncs += 1


def test_run_main_viewer_steps_and_handles_pause_resume_reset(monkeypatch):
    import bracketbot_sim.robot as module
    import glfw
    import mujoco.viewer

    FakeRunBot.instances.clear(); algorithms = []
    viewer = FakeViewer()

    def make(name, bot):
        value = SimpleNamespace(name=name, number=len(algorithms))
        algorithms.append(value)
        return value

    def launch(*_a, key_callback, **_k):
        key_callback(glfw.KEY_SPACE)
        key_callback(glfw.KEY_P)
        key_callback(glfw.KEY_R)
        return viewer

    monkeypatch.setattr(run_sim, "parse_args",
                        lambda: run_args(headless=False, duration=.01))
    monkeypatch.setattr(run_sim, "make_algorithm", make)
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(run_sim.time, "sleep", lambda *_a: None)
    run_sim.main()
    bot = FakeRunBot.instances[-1]
    assert len(algorithms) == 2
    assert bot.step_calls[0][1] is algorithms[1]
    assert viewer.syncs == 1 and bot.closed


def test_run_main_viewer_paused_does_not_step(monkeypatch):
    import bracketbot_sim.robot as module
    import glfw
    import mujoco.viewer

    FakeRunBot.instances.clear(); viewer = FakeViewer()

    def launch(*_a, key_callback, **_k):
        key_callback(glfw.KEY_SPACE)
        return viewer

    monkeypatch.setattr(run_sim, "parse_args",
                        lambda: run_args(headless=False, duration=.01))
    monkeypatch.setattr(run_sim, "make_algorithm", lambda *_a: object())
    monkeypatch.setattr(module, "BracketBot", FakeRunBot)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", launch)
    monkeypatch.setattr(run_sim.time, "sleep", lambda *_a: None)
    run_sim.main()
    assert FakeRunBot.instances[-1].step_calls == []
    assert viewer.syncs == 1


class FakeNav:
    SCAN = "scan"
    @classmethod
    def for_bot(cls, _bot, **kwargs): return cls(**kwargs)
    def __init__(self, **kwargs):
        self.kwargs = kwargs; self.state = "scan"; self.done = False
        self.obs = None; self.best_time = 0; self.best_detections = []
        self.detections = []; self.goal_xy = None; self.goal_label = "target"


class FakeNavBot(FakeRunBot):
    def step(self, duration, controller):
        self.time += duration
        controller.obs = SimpleNamespace(rgb=np.zeros((2, 2, 3), np.uint8))
        controller.state = "arrived"
        controller.done = True


def nav_args(**changes):
    base = dict(scene="course.xml", duration=.2, viewer=False, speed=100,
                out="result.png", frames=False, seed_scan=.5,
                detector="colour", weights=None, conf=.35, camera=None,
                goal=None, animate_movers=0.0)
    base.update(changes)
    return SimpleNamespace(**base)


def test_navigation_main_headless_runs_scores_figures_and_closes(monkeypatch, capsys):
    import bracketbot_sim.robot as robot_mod
    import vision_sim.navigation as nav_mod
    FakeNavBot.instances.clear(); figures = []
    monkeypatch.setattr(run_nav, "parse_args", lambda: nav_args())
    monkeypatch.setattr(run_nav, "build_detector", lambda *_a: "detector")
    import vision_sim.scene as scene_mod
    info = SimpleNamespace(describe=lambda: "scene info")
    monkeypatch.setattr(scene_mod.SceneInfo, "from_model", lambda *_a, **_k: info)
    monkeypatch.setattr(run_nav, "figure", lambda *args, **kwargs: figures.append(args))
    monkeypatch.setattr(robot_mod, "BracketBot", FakeNavBot)
    monkeypatch.setattr(nav_mod, "VisualNavigator", FakeNav)
    run_nav.main()
    bot = FakeNavBot.instances[-1]
    assert bot.closed and bot.balance.enabled
    assert figures and figures[0][3] == "result.png"
    assert "state=arrived" in capsys.readouterr().out


def test_navigation_main_without_observation_skips_figure(monkeypatch):
    import bracketbot_sim.robot as robot_mod
    import vision_sim.navigation as nav_mod
    FakeNavBot.instances.clear(); figures = []
    monkeypatch.setattr(run_nav, "parse_args", lambda: nav_args(duration=0))
    monkeypatch.setattr(run_nav, "build_detector", lambda *_a: None)
    import vision_sim.scene as scene_mod
    monkeypatch.setattr(scene_mod.SceneInfo, "from_model",
                        lambda *_a, **_k: SimpleNamespace(describe=lambda: "scene info"))
    monkeypatch.setattr(run_nav, "figure", lambda *args, **kwargs: figures.append(args))
    monkeypatch.setattr(robot_mod, "BracketBot", FakeNavBot)
    monkeypatch.setattr(nav_mod, "VisualNavigator", FakeNav)
    run_nav.main()
    assert figures == []


def test_navigation_main_viewer_ticks_frames_and_syncs(monkeypatch):
    import bracketbot_sim.robot as robot_mod
    import mujoco.viewer
    import vision_sim.navigation as nav_mod

    FakeNavBot.instances.clear(); viewer = FakeViewer(); saved = []; figures = []
    monkeypatch.setattr(run_nav, "parse_args",
                        lambda: nav_args(viewer=True, frames=True, duration=.04))
    monkeypatch.setattr(run_nav, "build_detector", lambda *_a: None)
    import vision_sim.scene as scene_mod
    monkeypatch.setattr(scene_mod.SceneInfo, "from_model",
                        lambda *_a, **_k: SimpleNamespace(describe=lambda: "scene info"))
    monkeypatch.setattr(run_nav, "figure", lambda *a, **k: figures.append(a))
    monkeypatch.setattr(run_nav, "annotate",
                        lambda *_a, **_k: SimpleNamespace(save=lambda path: saved.append(path)))
    monkeypatch.setattr(robot_mod, "BracketBot", FakeNavBot)
    monkeypatch.setattr(nav_mod, "VisualNavigator", FakeNav)
    monkeypatch.setattr(mujoco.viewer, "launch_passive", lambda *_a, **_k: viewer)
    monkeypatch.setattr(run_nav.time, "sleep", lambda *_a: None)
    run_nav.main()
    assert viewer.syncs == 1
    assert saved == ["rgbd_frame_00.png"]
    assert figures and FakeNavBot.instances[-1].closed


def test_navigation_main_coordinate_goal_animates_mocap_and_reports_truth(monkeypatch, capsys):
    import bracketbot_sim.robot as robot_mod
    import vision_sim.navigation as nav_mod
    import vision_sim.scene as scene_mod

    class MovingBot(FakeNavBot):
        def __init__(self, xml):
            super().__init__(xml)
            self.model = SimpleNamespace(nbody=1, body_mocapid=np.array([0]),
                                         body_pos=np.array([[1., 2., 3.]]))
            self.data = SimpleNamespace(mocap_pos=np.zeros((1, 3)))

    class GoalNav(FakeNav):
        @classmethod
        def for_bot(cls, _bot, goal=None, **kwargs):
            nav = cls(**kwargs)
            nav.goal_xy = np.asarray(goal, float)
            nav.best_detections = [SimpleNamespace(
                label="target", position=np.array([6., 0., 0.]),
                distance=1., bearing=0., extent=np.zeros(3), pixels=1,
                bbox=(0, 0, 0, 0))]
            return nav

    MovingBot.instances.clear(); figures = []
    args = nav_args(scene="obstacle_course.xml", goal="4, 5",
                    animate_movers=.8, duration=.2)
    monkeypatch.setattr(run_nav, "parse_args", lambda: args)
    monkeypatch.setattr(run_nav, "build_detector", lambda *_a: None)
    monkeypatch.setattr(run_nav, "figure", lambda *a, **k: figures.append((a, k)))
    monkeypatch.setattr(scene_mod.SceneInfo, "from_model",
                        lambda *_a, **_k: SimpleNamespace(describe=lambda: "scene info"))
    monkeypatch.setattr(robot_mod, "BracketBot", MovingBot)
    monkeypatch.setattr(nav_mod, "VisualNavigator", GoalNav)
    run_nav.main()
    bot = MovingBot.instances[-1]
    assert bot.data.mocap_pos[0] == pytest.approx([1, -3, 3])
    assert figures[0][1]["truth"] is True
    output = capsys.readouterr().out
    assert "coordinate (4.0, 5.0)" in output
    assert "animating 1 mocap body" in output
    assert "m from the goal" in output and "position error" in output


def train_args(tmp_path, **changes):
    base = dict(scene="scene.xml", data=str(tmp_path / "data"), train_images=2,
                val_images=1, width=16, height=12, epochs=1, batch=1,
                model="base.pt", device=None, dataset_only=True, keep=True, seed=4)
    base.update(changes)
    return SimpleNamespace(**base)


def test_train_main_reuses_dataset_and_stops_in_dataset_only_mode(monkeypatch, tmp_path, capsys):
    data = tmp_path / "data"; data.mkdir(); (data / "data.yaml").write_text("x")
    monkeypatch.setattr(train, "parse_args", lambda: train_args(tmp_path))
    train.main()
    output = capsys.readouterr().out
    assert "reusing dataset" in output and "dataset ready" in output


def test_train_main_regenerates_dataset_and_closes_bot(monkeypatch, tmp_path):
    import bracketbot_sim.robot as robot_mod
    from vision_sim import yolo_dataset
    made = []

    class Bot:
        def __init__(self, xml): self.xml = xml; self.closed = False; made.append(self)
        def close(self): self.closed = True

    def generate(bot, out, **kwargs):
        out.mkdir(parents=True)
        result = out / "data.yaml"; result.write_text("x")
        made.append(kwargs)
        return result

    stale = tmp_path / "data"; stale.mkdir(); (stale / "stale.txt").write_text("old")
    monkeypatch.setattr(train, "parse_args", lambda: train_args(tmp_path, keep=False))
    monkeypatch.setattr(robot_mod, "BracketBot", Bot)
    monkeypatch.setattr(yolo_dataset, "generate", generate)
    train.main()
    assert made[0].closed and made[1]["n_train"] == 2 and made[1]["seed"] == 4
    assert not (stale / "stale.txt").exists()


def test_train_main_invokes_yolo_training_with_cpu_fallback(monkeypatch, tmp_path, capsys):
    data = tmp_path / "data"; data.mkdir(); yaml = data / "data.yaml"; yaml.write_text("x")
    args = train_args(tmp_path, dataset_only=False, keep=True, device=None)
    calls = []

    import torch
    import ultralytics

    class YOLO:
        def __init__(self, model): calls.append(("init", model))
        def train(self, **kwargs): calls.append(("train", kwargs))

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(ultralytics, "YOLO", YOLO)
    train.main()
    assert calls[0] == ("init", "base.pt")
    assert calls[1][1]["device"] == "cpu" and calls[1][1]["epochs"] == 1
    assert "best weights" in capsys.readouterr().out


def diagnostic_observation():
    return perception.Observation(
        rgb=np.zeros((4, 6, 3), np.uint8), depth=np.ones((4, 6), np.float32),
        points=np.zeros((4, 6, 3)), valid=np.ones((4, 6), bool),
        cam_pos=np.zeros(3), cam_mat=np.eye(3),
        intrinsics=perception.Intrinsics(1, 1, 3, 2, 6, 4), robot_xy=np.zeros(2))


def test_navigation_figure_writes_full_diagnostic(tmp_path):
    shape = (8, 6)
    grid = SimpleNamespace(
        occupied=np.zeros(shape, bool), unknown=np.ones(shape, bool),
        seen=np.zeros(shape, bool), origin=(-1., -1.), size=shape, resolution=.5)
    obs = diagnostic_observation()
    det = perception.Detection("target", np.array([6., 0., .5]), 2, 0,
                               np.ones(3), 20, (0, 0, 2, 2))
    nav = SimpleNamespace(grid=grid, best_obs=obs, obs=obs,
                          best_detections=[det], detections=[det], best_time=1.,
                          blocked=None, path=[np.array([1., 1.]), np.array([2., 1.])],
                          goal_xy=np.array([6.1, 0.]), state="arrived", max_range=12)
    bot = SimpleNamespace(position=np.array([2., 1., 0.]), time=5.)
    out = tmp_path / "diagnostic.png"
    run_nav.figure(bot, nav, np.array([[0., 0.], [1., 1.]]), out, elapsed=.2)
    assert out.exists() and out.stat().st_size > 1000


def test_navigation_figure_without_truth_uses_coordinate_caption(tmp_path):
    shape = (4, 4)
    grid = SimpleNamespace(
        occupied=np.zeros(shape, bool), unknown=np.ones(shape, bool),
        seen=np.zeros(shape, bool), origin=(10., 20.), size=shape, resolution=.5)
    obs = diagnostic_observation()
    nav = SimpleNamespace(grid=grid, best_obs=None, obs=obs,
                          best_detections=[], detections=[], best_time=0.,
                          blocked=np.zeros(shape, bool), path=[],
                          goal_xy=np.array([11., 21.]), state="navigate", max_range=8)
    bot = SimpleNamespace(position=np.array([10.5, 20.5, 0.]), time=1.)
    out = tmp_path / "coordinate.png"
    run_nav.figure(bot, nav, np.empty((0, 2)), out, elapsed=.1, truth=False)
    assert out.exists() and out.stat().st_size > 1000


def test_robot_camera_methods_with_fake_renderer(monkeypatch, robot):
    class Renderer:
        def __init__(self, depth=False): self.depth = depth; self.scenes = []
        def update_scene(self, data, camera): self.scenes.append((data, camera))
        def render(self):
            return (np.array([[1., 99.]], np.float32) if self.depth
                    else np.full((2, 3, 3), 7, np.uint8))

    rgb, depth = Renderer(False), Renderer(True)
    monkeypatch.setattr(robot, "_renderer",
                        lambda cache, height, width, depth: (depth_renderer if depth else rgb))
    depth_renderer = depth
    image = robot.camera("head_rgb", 3, 2)
    ranges = robot.depth("head_depth", 2, 1, max_range=10)
    assert image.shape == (2, 3, 3) and image.dtype == np.uint8
    assert ranges[0, 0] == 1 and np.isinf(ranges[0, 1])
    left, right = robot.stereo(3, 2)
    assert left.shape == right.shape == (2, 3, 3)
    all_views = robot.all_cameras(3, 2)
    assert set(all_views) == set(robot.camera_names)


def test_robot_close_swallows_renderer_close_errors(robot):
    class Bad:
        def close(self): raise RuntimeError("driver gone")
    robot._renderers[(1, 1)] = Bad()
    robot._depth_renderers[(1, 1)] = Bad()
    robot.close()
    assert not robot._renderers and not robot._depth_renderers


@pytest.mark.rendering
@pytest.mark.integration
def test_real_offscreen_rgb_depth_and_registered_observation(robot):
    robot.reset()
    try:
        rgb = robot.camera("head_depth", width=32, height=24)
        depth = robot.depth("head_depth", width=32, height=24, max_range=12)
    except (mujoco.FatalError, RuntimeError) as exc:
        pytest.skip(f"off-screen OpenGL unavailable: {exc}")
    assert rgb.shape == (24, 32, 3) and rgb.dtype == np.uint8
    assert depth.shape == (24, 32) and depth.dtype == np.float32
    obs = perception.observe(robot, width=32, height=24)
    assert obs.points.shape == (24, 32, 3)
    assert np.array_equal(obs.valid, np.isfinite(obs.depth))
    robot.close()


@pytest.mark.slow
@pytest.mark.integration
def test_dynamic_model_builder_round_trip(monkeypatch):
    builder = load_script("build_dynamic_model_integration", MAIN / "build_dynamic_model.py")
    generated = MAIN / "_pytest_chopped_dynamic.xml"
    monkeypatch.setattr(builder, "DST", generated)
    try:
        model = builder.build()
        assert generated.exists() and model.ncam == 7 and model.nu >= 20
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_grasp") >= 0
    finally:
        generated.unlink(missing_ok=True)


@pytest.mark.parametrize(
    ("script", "needle"),
    [(MAIN / "run.py", "Run the simulated BracketBot"),
     (VISION / "run_navigation.py", "Object detection + path planning"),
     (VISION / "train_yolo.py", "Generate an auto-labelled dataset")],
)
def test_command_line_program_help_smoke(script, needle):
    result = subprocess.run(
        [sys.executable, str(script), "--help"], cwd=script.parent,
        capture_output=True, text=True, timeout=20, check=False)
    assert result.returncode == 0
    assert needle in result.stdout and "usage:" in result.stdout
