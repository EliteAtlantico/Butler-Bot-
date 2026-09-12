"""Auto-labelled YOLO training data from MuJoCo segmentation renders.

Nothing here is hand-annotated. MuJoCo can render a buffer of geom ids in
place of pixels, so the exact silhouette of every object -- already correct
for occlusion and truncation -- comes straight out of the renderer. Turning
that into YOLO boxes is the whole trick behind training a detector in sim.

    python train_yolo.py           # generate a dataset, then train on it

Poses and object placements are randomised each frame so the detector learns
the objects rather than the layout it was born in.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

# Body name -> class. Everything else in the scene is background, including
# the robot's own arms, which the head camera does occasionally catch.
BODY_CLASSES = {
    "target_column": "target",
    "barrier_south": "barrier",
    "barrier_north": "barrier",
    "pillar_a": "pillar",
    "pillar_b": "pillar",
    "pillar_c": "pillar",
}
CLASS_NAMES = ["target", "barrier", "pillar"]
CLASS_INDEX = {n: i for i, n in enumerate(CLASS_NAMES)}


@dataclass
class Box:
    cls: int
    u0: int
    v0: int
    u1: int
    v1: int

    def yolo_line(self, width, height):
        cx = (self.u0 + self.u1) / 2 / width
        cy = (self.v0 + self.v1) / 2 / height
        w = (self.u1 - self.u0) / width
        h = (self.v1 - self.v0) / height
        return f"{self.cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


@dataclass
class SceneRandomiser:
    """Jitters body placements so the detector cannot memorise the layout."""
    model: object
    jitter: float = 0.9
    _home: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in BODY_CLASSES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self._home[bid] = self.model.body_pos[bid].copy()

    def randomise(self, rng):
        for bid, home in self._home.items():
            offset = np.zeros(3)
            offset[:2] = rng.uniform(-self.jitter, self.jitter, 2)
            self.model.body_pos[bid] = home + offset

    def restore(self):
        for bid, home in self._home.items():
            self.model.body_pos[bid] = home

    def body_xy(self, data, name):
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return data.xpos[bid][:2].copy()


def place_robot(model, data, xy, yaw, height=0.30):
    """Teleport the chassis free joint to a pose and settle the kinematics."""
    data.qpos[0:2] = xy
    data.qpos[2] = height
    data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def boxes_from_segmentation(model, seg, min_pixels=28, min_side=3):
    """Geom-id buffer -> one box per visible labelled body."""
    objid, objtype = seg[..., 0], seg[..., 1]
    is_geom = objtype == mujoco.mjtObj.mjOBJ_GEOM
    out: list[Box] = []
    for gid in np.unique(objid[is_geom]):
        if gid < 0:
            continue
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                 int(model.geom_bodyid[int(gid)]))
        label = BODY_CLASSES.get(body)
        if label is None:
            continue
        mask = is_geom & (objid == gid)
        if mask.sum() < min_pixels:
            continue
        vs, us = np.where(mask)
        u0, u1, v0, v1 = us.min(), us.max(), vs.min(), vs.max()
        if (u1 - u0) < min_side or (v1 - v0) < min_side:
            continue
        out.append(Box(CLASS_INDEX[label], int(u0), int(v0), int(u1), int(v1)))
    return out


def generate(bot, out_dir, n_train=600, n_val=150, width=320, height=240,
             camera="head_depth", seed=0, area=(-2.0, 6.5, -4.0, 4.5),
             verbose=True):
    """Render a randomised, auto-labelled YOLO dataset. Returns the data.yaml."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    rand = SceneRandomiser(bot.model)

    seg_r = mujoco.Renderer(bot.model, height=height, width=width)
    seg_r.enable_segmentation_rendering()
    rgb_r = mujoco.Renderer(bot.model, height=height, width=width)

    counts = {n: 0 for n in CLASS_NAMES}
    empties = 0
    try:
        for split, n in (("train", n_train), ("val", n_val)):
            (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
            made = 0
            attempts = 0
            while made < n and attempts < n * 12:
                attempts += 1
                rand.randomise(rng)
                x = rng.uniform(area[0], area[1])
                y = rng.uniform(area[2], area[3])
                # Aim at a random object most of the time, so the frames are
                # mostly useful; the rest are background negatives.
                if rng.random() < 0.85:
                    place_robot(bot.model, bot.data, (x, y), 0.0)
                    tgt = rand.body_xy(bot.data, rng.choice(list(BODY_CLASSES)))
                    yaw = np.arctan2(tgt[1] - y, tgt[0] - x) + rng.normal(0, 0.35)
                else:
                    yaw = rng.uniform(-np.pi, np.pi)
                place_robot(bot.model, bot.data, (x, y), yaw)

                seg_r.update_scene(bot.data, camera=camera)
                boxes = boxes_from_segmentation(bot.model, seg_r.render())
                if not boxes and empties > (n_train + n_val) * 0.1:
                    continue
                if not boxes:
                    empties += 1

                rgb_r.update_scene(bot.data, camera=camera)
                stem = f"{split}_{made:05d}"
                Image.fromarray(rgb_r.render()).save(
                    out_dir / "images" / split / f"{stem}.png")
                (out_dir / "labels" / split / f"{stem}.txt").write_text(
                    "\n".join(b.yolo_line(width, height) for b in boxes))
                for b in boxes:
                    counts[CLASS_NAMES[b.cls]] += 1
                made += 1
                if verbose and made % 100 == 0:
                    print(f"  {split}: {made}/{n}")
    finally:
        seg_r.close()
        rgb_r.close()
        rand.restore()
        mujoco.mj_forward(bot.model, bot.data)

    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n")
    if verbose:
        print(f"dataset at {out_dir}  instances: {counts}  "
              f"background frames: {empties}")
    return yaml_path
