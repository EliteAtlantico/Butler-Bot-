from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import mujoco
import numpy as np
import pytest
from PIL import Image

from vision_sim import perception, yolo_dataset
from vision_sim.yolo_detector import YoloDetector


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_mujoco"
VISION = ROOT / "comp_vision_sim"


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run_sim = load_script("run_sim_script", MAIN / "run.py")
run_nav = load_script("run_nav_script", VISION / "run_navigation.py")
train_yolo = load_script("train_yolo_script", VISION / "train_yolo.py")


def obs_for_yolo():
    depth = np.array([[1, 1, 1, 5], [1, 1, 1, 5],
                      [1, 1, 1, 5], [1, 1, 1, 5]], np.float32)
    points = np.zeros((4, 4, 3), float)
    points[..., 0] = depth
    points[..., 2] = .5
    return perception.Observation(
        rgb=np.zeros((4, 4, 3), np.uint8), depth=depth, points=points,
        valid=np.ones((4, 4), bool), cam_pos=np.zeros(3), cam_mat=np.eye(3),
        intrinsics=perception.Intrinsics(1, 1, 2, 2, 4, 4),
        robot_xy=np.zeros(2),
    )


def test_box_yolo_line_coordinates_and_format():
    line = yolo_dataset.Box(2, 10, 20, 30, 60).yolo_line(100, 100)
    assert line == "2 0.200000 0.400000 0.200000 0.400000"


def labelled_model():
    bodies = "".join(
        f"<body name='{name}' pos='{i} 0 0'><geom name='g{i}' type='box' size='.1 .1 .1'/></body>"
        for i, name in enumerate(yolo_dataset.BODY_CLASSES, start=1)
    )
    xml = f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"
    return mujoco.MjModel.from_xml_string(xml)


def test_scene_randomiser_records_jitters_restores_and_reads_position():
    model = labelled_model()
    rand = yolo_dataset.SceneRandomiser(model, jitter=.5)
    assert len(rand._home) == len(yolo_dataset.BODY_CLASSES)
    homes = {k: v.copy() for k, v in rand._home.items()}
    rand.randomise(np.random.default_rng(1))
    assert any(not np.array_equal(model.body_pos[k], v) for k, v in homes.items())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    assert rand.body_xy(data, "target_column").shape == (2,)
    rand.restore()
    for k, v in homes.items():
        assert model.body_pos[k] == pytest.approx(v)


def test_place_robot_sets_pose_zeroes_velocity_and_forwards():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body><freejoint/><geom type='sphere' size='.1'/></body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    data.qvel[:] = 3
    yolo_dataset.place_robot(model, data, (2, -1), np.pi / 2, height=.4)
    assert data.qpos[:3] == pytest.approx([2, -1, .4])
    assert data.qpos[3:7] == pytest.approx([np.sqrt(.5), 0, 0, np.sqrt(.5)])
    assert not data.qvel.any()


def test_boxes_from_segmentation_filters_type_body_pixels_and_side():
    model = labelled_model()
    target_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g1")
    unknown_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g6")
    seg = np.full((8, 8, 2), -1, int)
    seg[1:6, 2:7, 0] = target_gid
    seg[1:6, 2:7, 1] = mujoco.mjtObj.mjOBJ_GEOM
    boxes = yolo_dataset.boxes_from_segmentation(model, seg, min_pixels=10, min_side=2)
    assert boxes == [yolo_dataset.Box(0, 2, 1, 6, 5)]
    assert yolo_dataset.boxes_from_segmentation(model, seg, min_pixels=100) == []
    seg[:, :, 1] = mujoco.mjtObj.mjOBJ_BODY
    assert yolo_dataset.boxes_from_segmentation(model, seg, min_pixels=1) == []
    assert unknown_gid >= 0


def test_boxes_from_segmentation_skips_unlabelled_body():
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='other'><geom name='g' type='box' size='.1 .1 .1'/></body></worldbody></mujoco>"
    )
    seg = np.zeros((5, 5, 2), int)
    seg[..., 0] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g")
    seg[..., 1] = mujoco.mjtObj.mjOBJ_GEOM
    assert yolo_dataset.boxes_from_segmentation(model, seg, min_pixels=1) == []


def test_yolo_init_checks_weights(monkeypatch, tmp_path):
    fake = ModuleType("ultralytics")
    fake.YOLO = lambda _path: None
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    # Weights are now either a path that exists or a pretrained Ultralytics
    # model name to fetch, so a missing path reports that rather than
    # suggesting training a net of our own.
    with pytest.raises(FileNotFoundError, match="no YOLO weights at"):
        YoloDetector(tmp_path / "missing.pt")


def test_yolo_init_loads_model_options_and_repr(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"x")
    # `.model` is inspected to tell an open-vocabulary net (YOLO-World) from a
    # fixed-class one; a plain object is neither, so this takes the fixed path.
    model = SimpleNamespace(names={0: "target", 1: "pillar"}, model=object())
    fake = ModuleType("ultralytics")
    fake.YOLO = lambda path: model
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    detector = YoloDetector(weights, conf=.2, iou=.6, imgsz=128, device="cuda")
    assert detector.model is model and detector.names == model.names
    assert (detector.conf, detector.iou, detector.imgsz, detector.device) == (.2, .6, 128, "cuda")
    # repr now names the weights file, and the target when there is one (a
    # fixed-class net without an explicit --target has none).
    assert repr(detector) == "<YoloDetector model.pt 2 classes conf=0.2>"


@pytest.mark.integration
def test_repository_yolo_weights_load_with_expected_classes():
    weights = VISION / "runs" / "bracketbot_yolo" / "weights" / "best.pt"
    detector = YoloDetector(weights, imgsz=32)
    assert detector.names == {0: "target", 1: "barrier", 2: "pillar"}


def bare_detector(**kwargs):
    d = object.__new__(YoloDetector)
    d.min_height = kwargs.get("min_height", .08)
    d.max_height = kwargs.get("max_height", 2.5)
    d.min_range = kwargs.get("min_range", .4)
    d.self_radius = kwargs.get("self_radius", .55)
    # label_for() maps the target class onto the goal label; None means the
    # model has no designated target and every class keeps its own name.
    d.target = kwargs.get("target")
    d.goal_label = kwargs.get("goal_label", "target")
    return d


def test_yolo_range_box_clips_box_and_computes_near_surface():
    detector = bare_detector(self_radius=.1)
    det = detector._range_box(obs_for_yolo(), "target", -10, -10, 10, 10, .2, .9)
    assert det is not None
    assert det.label == "target" and det.bbox == (0, 0, 3, 3)
    assert det.pixels == 16 and det.distance < 2
    assert det.bearing == pytest.approx(-.2)


@pytest.mark.parametrize("box", [(1, 1, 1, 3), (1, 1, 3, 1), (9, 9, 12, 12)])
def test_yolo_range_box_rejects_empty_or_invalid_box(box):
    assert bare_detector()._range_box(obs_for_yolo(), "x", *box, 0, .5) is None


def test_yolo_range_box_rejects_too_few_good_depth_pixels():
    obs = obs_for_yolo()
    obs.depth[:] = np.inf
    obs.depth.flat[:7] = 1
    assert bare_detector(self_radius=.1)._range_box(obs, "x", 0, 0, 3, 3, 0, .5) is None


def test_yolo_call_passes_predict_options_ranges_and_sorts():
    class Box:
        def __init__(self, xyxy, cls, conf):
            self.xyxy = np.array([xyxy], float)
            self.cls = cls
            self.conf = conf

    boxes = [Box([0, 0, 3, 3], 1, .7), Box([0, 0, 3, 3], 0, .8)]
    model = SimpleNamespace(predict=lambda *a, **k: [SimpleNamespace(boxes=boxes)])
    d = bare_detector(self_radius=.1)
    d.model = model; d.names = {0: "near", 1: "far"}
    d.conf=.3; d.iou=.5; d.imgsz=320; d.device="cpu"; d.verbose=False
    ranges = iter([SimpleNamespace(distance=5), SimpleNamespace(distance=1)])
    d._range_box = lambda *_a, **_k: next(ranges)
    out = d(obs_for_yolo(), robot_yaw=.2, ignored=True)
    assert [x.distance for x in out] == [1, 5]


@pytest.mark.parametrize(
    ("argv", "algorithm", "scene", "headless"),
    [(["run.py"], "stand", None, False),
     (["run.py", "-a", "pick", "--scene", "x.xml", "--headless"], "pick", "x.xml", True)],
)
def test_run_parse_args(monkeypatch, argv, algorithm, scene, headless):
    monkeypatch.setattr(sys, "argv", argv)
    args = run_sim.parse_args()
    assert (args.algorithm, args.scene, args.headless) == (algorithm, scene, headless)


def test_navigation_and_training_parse_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_navigation.py"])
    nav = run_nav.parse_args()
    # --detector defaults to None: the detector is chosen from the other
    # options (a --target implies YOLO, and so on) rather than fixed to colour.
    assert nav.duration == 90 and nav.detector is None and nav.seed_scan == 1
    monkeypatch.setattr(sys, "argv", ["train_yolo.py"])
    train = train_yolo.parse_args()
    assert train.train_images == 600 and train.val_images == 150
    assert train.epochs == 30 and not train.dataset_only and not train.keep


@pytest.mark.parametrize("name", ["stand", "square", "avoid", "waypoints", "drive", "spin"])
def test_make_algorithm_all_motion_choices(name):
    assert callable(run_sim.make_algorithm(name))


def test_make_algorithm_pick_and_unknown(monkeypatch):
    import bracketbot_sim.manipulation as manipulation
    sentinel = object()
    monkeypatch.setattr(manipulation, "PickCube", lambda bot: ("pick", bot))
    assert run_sim.make_algorithm("pick", sentinel) == ("pick", sentinel)
    with pytest.raises(ValueError):
        run_sim.make_algorithm("invalid")


def test_depth_image_colours_and_nonfinite():
    image = run_nav.depth_image(np.array([[0., 6., 12., np.inf]]), max_range=12)
    assert image.dtype == np.uint8 and image.shape == (1, 4, 3)
    assert np.array_equal(image[0, 0], [255, 0, 0])
    assert np.array_equal(image[0, 2], [0, 0, 255])
    assert np.array_equal(image[0, 3], [0, 0, 0])


def test_depth_image_all_nonfinite_is_black():
    image = run_nav.depth_image(np.full((2, 3), np.inf), max_range=12)
    assert not image.any()


@pytest.mark.parametrize(
    ("p", "a", "b", "distance"),
    [((1, 1), (0, 0), (2, 0), 1), ((-1, 0), (0, 0), (2, 0), 1),
     ((3, 0), (0, 0), (2, 0), 1), ((3, 4), (0, 0), (0, 0), 5)],
)
def test_point_segment_distance(p, a, b, distance):
    assert run_nav._point_segment_distance(p, a, b) == pytest.approx(distance)


def test_score_detections_known_and_unknown_labels():
    known = perception.Detection("target", np.array([6.3, 0, 0]), 1, 0, np.zeros(3), 1)
    unknown = perception.Detection("mystery", np.zeros(3), 1, 0, np.zeros(3), 1)
    rows = run_nav.score_detections([known, unknown])
    assert len(rows) == 1 and rows[0][0] is known and rows[0][1] == pytest.approx(.3)


def test_annotate_returns_scaled_pil_image():
    rgb = np.zeros((10, 20, 3), np.uint8)
    d = perception.Detection("target", np.ones(3), 2.3, 0, np.ones(3), 5,
                             bbox=(1, 2, 8, 9))
    image = run_nav.annotate(rgb, [d], scale=2)
    assert isinstance(image, Image.Image) and image.size == (40, 20)
    assert np.asarray(image).any()


def test_build_detector_colour_returns_none():
    assert run_nav.build_detector(SimpleNamespace(detector="colour")) is None


def test_build_detector_yolo_uses_default_or_explicit_weights(monkeypatch, capsys):
    import vision_sim.yolo_detector as module
    made = []
    monkeypatch.setattr(
        module, "YoloDetector",
        lambda weights, target=None, classes=None, conf=None:
        made.append((weights, conf)) or SimpleNamespace(
            names={0: "x"}, weights=weights, device="cpu", target=target,
            open_vocab=True))
    args = SimpleNamespace(detector="yolo", weights="custom.pt", conf=.7,
                           target="red cylinder", classes=None)
    run_nav.build_detector(args)
    assert made[0] == ("custom.pt", .7)
    # The summary now names the weights, target and class count instead of
    # listing the classes outright -- an open vocabulary can be dozens long.
    out = capsys.readouterr().out
    assert "custom.pt" in out and "'red cylinder'" in out and "1 classes" in out


def test_montage_writes_expected_sheet(tmp_path):
    class Bot:
        camera_names = ["a", "b"]
        def camera(self, name, width, height):
            return np.full((height, width, 3), 50 if name == "a" else 100, np.uint8)
        def depth(self, *_a, **_k):
            return np.array([[1, 2], [np.inf, 3]], np.float32)
    out = tmp_path / "montage.png"
    run_sim.montage(Bot(), out, width=2, height=2)
    image = Image.open(out)
    assert image.size == (8, 2) and image.mode == "RGB"


@pytest.mark.slow
def test_generated_dataset_minimal_with_fake_renderers(monkeypatch, tmp_path):
    names = list(yolo_dataset.BODY_CLASSES)
    bodies = "".join(
        f"<body name='{n}' pos='{i} 0 0'><geom name='g{i}' type='box' size='.1 .1 .1'/></body>"
        for i, n in enumerate(names, 1)
    )
    xml = (f"<mujoco><worldbody><body name='bot'><freejoint/>"
           f"<geom type='sphere' size='.1'/><camera name='head_depth'/></body>"
           f"{bodies}</worldbody></mujoco>")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    target_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "g1")

    class Renderer:
        def __init__(self, _model, height, width):
            self.height, self.width, self.seg = height, width, False
            self.closed = False
        def enable_segmentation_rendering(self): self.seg = True
        def update_scene(self, *_a, **_k): pass
        def render(self):
            if not self.seg:
                return np.full((self.height, self.width, 3), 80, np.uint8)
            out = np.full((self.height, self.width, 2), -1, int)
            out[1:-1, 1:-1, 0] = target_gid
            out[1:-1, 1:-1, 1] = mujoco.mjtObj.mjOBJ_GEOM
            return out
        def close(self): self.closed = True

    monkeypatch.setattr(yolo_dataset.mujoco, "Renderer", Renderer)
    bot = SimpleNamespace(model=model, data=data)
    out = tmp_path / "dataset"
    yaml = yolo_dataset.generate(bot, out, n_train=1, n_val=1, width=8, height=8,
                                 seed=2, verbose=False)
    assert yaml.exists()
    assert len(list((out / "images" / "train").glob("*.png"))) == 1
    assert len(list((out / "labels" / "val").glob("*.txt"))) == 1
    assert "names: ['target', 'barrier', 'pillar']" in yaml.read_text()
