"""The household items, furniture and person look real to a camera -- and are
physically exactly what they were.

The realistic models (comp_vision_sim/assets/household.xml, made by
tools/make_assets.py) are visual only. These check the promises that makes:
every item has a look, the look has no mass and no contacts, it fits inside the
collision primitives it dresses (the depth camera measures the look), and
stripping the looks out changes no body's mass or inertia. Plus the
object-first detector's colour handling, which is pure and needs no model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
VISION = ROOT / "comp_vision_sim"
SCENES = {
    "scene_home": ROOT / "Hand_and_Wrists" / "scenes" / "scene_home.xml",
    "home_search": VISION / "home_search.xml",
}
PLACE_BODIES = ("coffee_table", "side_table", "basket", "person")


@pytest.fixture(scope="module", params=list(SCENES))
def scene(request):
    return request.param, SCENES[request.param], mujoco.MjModel.from_xml_path(str(SCENES[request.param]))


def _body(m, name):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)


def _is_look(m, g):
    return m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0


def _aabb_in_body(m, g):
    """(lo, hi) of geom g's extent, in its body's frame. Meshes by their
    vertices: MuJoCo turns a mesh geom to its principal axes, and the box
    round that turned box would overstate it."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    R = R.reshape(3, 3)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        k = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]] @ R.T + m.geom_pos[g]
        return v.min(0), v.max(0)
    c = m.geom_pos[g] + R @ m.geom_aabb[g, :3]
    h = np.abs(R) @ m.geom_aabb[g, 3:]
    return c - h, c + h


def test_every_item_has_a_look_that_is_visual_only(scene):
    from handwrist.objects import CATALOGUE
    _, _, m = scene
    for name in CATALOGUE:
        b = _body(m, name)
        assert b >= 0, name
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        looks = [g for g in geoms if _is_look(m, g)]
        solid = [g for g in geoms if not _is_look(m, g)]
        assert looks, f"{name} has no look geom"
        for g in looks:
            assert m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH and m.geom_group[g] == 2
            assert m.geom_matid[g] >= 0 and m.mat_texid[m.geom_matid[g]].max() >= 0, \
                f"{name}'s look is untextured"
        # the primitives are still there, and hidden
        assert solid and all(m.geom_group[g] == 3 for g in solid), name


def test_every_look_fits_inside_its_primitives(scene):
    """The depth camera measures what is rendered: a look bigger than the
    primitives would be a bigger object to perception and to the planner."""
    from handwrist.objects import CATALOGUE
    label, _, m = scene
    bodies = list(CATALOGUE) + [p for p in PLACE_BODIES if _body(m, p) >= 0]
    for name in bodies:
        b = _body(m, name)
        geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        solid = [_aabb_in_body(m, g) for g in geoms if not _is_look(m, g)]
        lo = np.min([s[0] for s in solid], 0)
        hi = np.max([s[1] for s in solid], 0)
        for g in geoms:
            if _is_look(m, g):
                glo, ghi = _aabb_in_body(m, g)
                assert np.all(glo >= lo - 5e-4) and np.all(ghi <= hi + 5e-4), \
                    f"{label}: a look on {name} spills out of its primitives: " \
                    f"{np.round(glo, 4)}..{np.round(ghi, 4)} vs {np.round(lo, 4)}..{np.round(hi, 4)}"


def test_looks_add_no_mass_or_inertia(scene):
    label, path, m = scene
    spec = mujoco.MjSpec.from_file(str(path))
    for g in list(spec.geoms):
        # only the household looks: the plant's pot and canopy are older
        # visual meshes that do carry mass, and are not what is tested here
        if g.type == mujoco.mjtGeom.mjGEOM_MESH and g.meshname.startswith("look_"):
            spec.delete(g)
    bare = spec.compile()
    assert bare.nbody == m.nbody
    for b in range(m.nbody):
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_allclose(getattr(bare, field)[b], getattr(m, field)[b], atol=1e-12,
                                       err_msg=f"{label}: {field} of body {b}")


def _without_looks(path):
    """The scene compiled with every household look removed."""
    spec = mujoco.MjSpec.from_file(str(path))
    for g in list(spec.geoms):
        visual = g.group == 2 and g.contype == 0 and g.conaffinity == 0
        if visual and (g.meshname.startswith("look_") or g.type != mujoco.mjtGeom.mjGEOM_MESH):
            spec.delete(g)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def test_what_the_robot_reads_off_the_scene_ignores_the_looks(scene):
    """Surfaces, place footprints, rims, low obstacles and the navigation
    bounds come out the same with the looks as without them."""
    from types import SimpleNamespace

    from handwrist.grasping import GraspPlanner
    from handwrist.places import PLACES, footprint_of, surface_of
    from handwrist.surfaces import find_surfaces, obstacle_footprints
    from vision_sim import scene as vscene

    label, path, m = scene
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bm, bd = _without_looks(path)
    assert bm.ngeom < m.ngeom

    def surfaces(mm, dd):
        return [(s.name, s.body, s.mode, round(s.top, 9), round(s.surface.rim, 9),
                 tuple(np.round(s.surface.half, 9)), tuple(np.round(s.surface.center, 9)))
                for s in find_surfaces(mm, dd)]

    assert surfaces(m, d) == surfaces(bm, bd), label
    names = lambda fp: [(n, tuple(np.round(lo, 9)), tuple(np.round(hi, 9))) for n, lo, hi in fp]
    assert names(obstacle_footprints(m, d)) == names(obstacle_footprints(bm, bd))
    for p in PLACES.values():
        if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, p.surface_geom) >= 0:
            for a, b in zip(footprint_of(m, d, p.body), footprint_of(bm, bd, p.body)):
                np.testing.assert_array_equal(a, b)
            assert surface_of(m, d, p).rim == surface_of(bm, bd, p).rim
    low = lambda mm, dd: sorted((tuple(np.round(lo, 9)), tuple(np.round(hi, 9)), b) for lo, hi, b in
                                GraspPlanner._low_obstacles(SimpleNamespace(
                                    bot=SimpleNamespace(model=mm, data=dd), HULL_TOP=GraspPlanner.HULL_TOP)))
    assert low(m, d) == low(bm, bd)

    full, bare = vscene.SceneInfo.from_model(m, d), vscene.SceneInfo.from_model(bm, bd)
    np.testing.assert_allclose(full.bounds, bare.bounds, atol=1e-9)
    assert (full.floor_z, full.robot_radius, full.robot_top) == \
        (bare.floor_z, bare.robot_radius, bare.robot_top)


def test_a_generated_house_has_the_same_looks(tmp_path):
    sys.path.insert(0, str(VISION / "tools"))
    import make_house
    result = None
    seed = 0
    while result is None:
        result = make_house.build(seed)
        seed += 1000
    out = VISION / f"_test_house_{os.getpid()}.xml"      # its includes are relative to comp_vision_sim
    out.write_text(result[0], encoding="utf-8")
    try:
        m = mujoco.MjModel.from_xml_path(str(out))
    finally:
        out.unlink()
    for name in ("mug", "can", "remote", "box"):
        b = _body(m, name)
        assert any(_is_look(m, g) and m.geom_bodyid[g] == b for g in range(m.ngeom)), name


# ------------------------------------------------- object first, colour second
def test_every_catalogue_object_has_detector_queries():
    from handwrist.detection import QUERIES
    from handwrist.objects import CATALOGUE
    assert set(QUERIES) == set(CATALOGUE)
    assert all(QUERIES[n] and all(q.strip() for q in QUERIES[n]) for n in QUERIES)


@pytest.mark.parametrize("text, colour, rest", [
    ("red mug", "red", "mug"), ("the Red Mug", "red", "the mug"), ("keys", None, "keys"),
    ("a light blue bottle", "blue", "a bottle"), ("grey remote", "grey", "remote"),
    ("gray remote", "gray", "remote"), ("", None, ""),
])
def test_split_colour(text, colour, rest):
    from handwrist.detection import split_colour
    assert split_colour(text) == (colour, rest)


def test_colour_only_ranks_what_the_detector_found():
    from handwrist.detection import rank_by_colour
    rgb = np.zeros((100, 200, 3), np.uint8)
    rgb[:, :100] = (200, 30, 25)       # a red thing on the left
    rgb[:, 100:] = (240, 240, 235)     # a white thing on the right
    hits = [{"name": "mug", "confidence": 0.60, "box": (110, 10, 190, 90)},    # white, more confident
            {"name": "mug", "confidence": 0.45, "box": (10, 10, 90, 90)}]      # red
    red = rank_by_colour(rgb, [dict(h) for h in hits], "red")
    assert red[0]["box"] == (10, 10, 90, 90) and red[0]["colour_match"] == 1.0
    white = rank_by_colour(rgb, [dict(h) for h in hits], "white")
    assert white[0]["box"] == (110, 10, 190, 90)
    # a colour nothing matches still leaves every candidate, most confident first
    green = rank_by_colour(rgb, [dict(h) for h in hits], "green")
    assert [h["confidence"] for h in green] == [0.60, 0.45]


def test_find_asks_for_the_object_not_the_colour():
    from handwrist.detection import DetectionEstimator
    see = DetectionEstimator(backend="yolo")
    asked = []

    def boxes(rgb, names, zoom=False):
        asked.append(list(names))
        return [{"name": names[0], "confidence": 0.5, "box": (0, 0, 10, 10), "source": "yolo"}]

    see.yolo_boxes = boxes
    rgb = np.zeros((20, 20, 3), np.uint8)
    see.find(rgb, "red mug")
    see.find(rgb, "keys")
    see.find(rgb, "sunglasses")
    assert asked == [["mug", "cup"], ["keys", "car key"], ["sunglasses"]]


class _FakeYolo:
    """Stands in for a YoloDetector: canned boxes per view size, recorded calls."""

    def __init__(self, names, boxes):
        self.names = dict(enumerate(names))
        self.boxes, self.seen = boxes, []

    def _predict(self, bgr):
        from types import SimpleNamespace
        self.seen.append(bgr.shape[:2])
        return SimpleNamespace(boxes=[
            SimpleNamespace(cls=c, conf=conf, xyxy=[SimpleNamespace(tolist=lambda b=b: list(b))])
            for c, conf, b in self.boxes.get(bgr.shape[:2], [])])


def test_a_box_a_distractor_claims_more_confidently_is_not_the_object():
    from handwrist.detection import DetectionEstimator
    see = DetectionEstimator(backend="yolo")
    fake = _FakeYolo(["cardboard box", "box", "basket"], {(480, 640): [
        (0, 0.30, (360, 300, 425, 370)),        # the basket, scored as a box
        (2, 0.45, (358, 298, 426, 372)),        # ...and, higher, as the basket it is
        (0, 0.12, (310, 440, 330, 478)),        # the real box
    ]})
    see.yolo = lambda names: fake
    hits = see.yolo_boxes(np.zeros((480, 640, 3), np.uint8), ["cardboard box", "box"])
    assert [h["box"] for h in hits] == [(310, 440, 330, 478)]


def test_the_zoom_pass_finds_small_things_and_maps_them_back():
    from handwrist.detection import TILE, ZOOM, DetectionEstimator, _tiles
    assert _tiles(640, 480) == [(0, 0), (160, 0), (320, 0), (0, 160), (160, 160), (320, 160)]
    see = DetectionEstimator(backend="yolo")
    # nothing in the whole frame; in each blown-up tile the same 40 x 20 px keys
    fake = _FakeYolo(["keys", "car key"], {(TILE * ZOOM, TILE * ZOOM): [(0, 0.40, (100, 560, 140, 580))]})
    see.yolo = lambda names: fake
    rgb = np.zeros((480, 640, 3), np.uint8)
    assert see.yolo_boxes(rgb, ["keys", "car key"]) == []
    found = see.find(rgb, "keys")
    assert found["source"] == "yolo-zoom" and found["confidence"] == 0.40
    # tile (0, 0): (100, 560)/2 -> (50, 280); boxes repeated by other tiles are merged
    assert found["box"] == (50, 280, 70, 290)
    assert len(see.yolo_boxes(rgb, ["keys"], zoom=True)) == 6       # one per tile, all distinct


def test_an_unreachable_llm_is_not_asked_again():
    from handwrist.detection import DetectionEstimator
    see = DetectionEstimator(backend="auto")
    see.yolo_boxes = lambda rgb, names, zoom=False: []
    calls = []

    def down(rgb, name):
        calls.append(name)
        raise ConnectionRefusedError("no LLM server")

    see.llm_box = down
    rgb = np.zeros((20, 20, 3), np.uint8)
    assert see.find(rgb, "mug") is None and see.find(rgb, "mug") is None
    assert calls == ["mug"] and "ConnectionRefusedError" in see.llm_error
