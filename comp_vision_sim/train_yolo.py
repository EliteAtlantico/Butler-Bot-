#!/usr/bin/env python3
"""Generate an auto-labelled dataset from MuJoCo and fine-tune YOLO on it.

    ./train_yolo.py                      # generate + train (CPU, ~15-30 min)
    ./train_yolo.py --dataset-only       # just build the dataset
    ./train_yolo.py --epochs 10          # quicker, less accurate

Stock COCO weights are useless in this scene -- it contains coloured
cylinders and boxes, not people and cars -- so the detector has to be taught
what a "target", "barrier" and "pillar" look like. The labels come from
MuJoCo's segmentation renderer, so no frame is annotated by hand.

The result lands at runs/bracketbot_yolo/weights/best.pt, which run_navigation.py
picks up automatically with --detector yolo.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The robot model and its BracketBot wrapper live next door; this package
# deliberately does not depend on them, so only the entry points bridge over.
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "main_mujoco"))

DEFAULT_DATA = HERE / "yolo_dataset"
DEFAULT_RUN = HERE / "runs"
WEIGHTS = DEFAULT_RUN / "bracketbot_yolo" / "weights" / "best.pt"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=str(HERE / "obstacle_course.xml"))
    p.add_argument("--data", default=str(DEFAULT_DATA))
    p.add_argument("--train-images", type=int, default=600)
    p.add_argument("--val-images", type=int, default=150)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--model", default="yolov8n.pt", help="base weights")
    p.add_argument("--device", default=None,
                   help="torch device: 'cpu' (default) or '0' for the first GPU")
    p.add_argument("--dataset-only", action="store_true")
    p.add_argument("--keep", action="store_true",
                   help="reuse an existing dataset instead of regenerating")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.environ["MUJOCO_GL"] = "wgl" if os.name == "nt" else "egl"

    from bracketbot_sim.robot import BracketBot
    from vision_sim import yolo_dataset

    data_dir = Path(args.data)
    yaml_path = data_dir / "data.yaml"

    if args.keep and yaml_path.exists():
        print(f"reusing dataset at {data_dir}")
    else:
        if data_dir.exists():
            shutil.rmtree(data_dir)
        bot = BracketBot(xml=args.scene)
        print(f"generating {args.train_images} train / {args.val_images} val "
              f"frames at {args.width}x{args.height} from {args.scene}")
        yaml_path = yolo_dataset.generate(
            bot, data_dir, n_train=args.train_images, n_val=args.val_images,
            width=args.width, height=args.height, seed=args.seed)
        bot.close()

    if args.dataset_only:
        print(f"dataset ready: {yaml_path}")
        return

    import torch
    from ultralytics import YOLO
    model = YOLO(args.model)
    device = args.device if args.device else ("0" if torch.cuda.is_available() else "cpu")
    print(f"training {args.model} for {args.epochs} epochs on device={device} "
          f"(imgsz={args.width})")
    model.train(data=str(yaml_path), epochs=args.epochs, imgsz=args.width,
                batch=args.batch, device=device, project=str(DEFAULT_RUN),
                name="bracketbot_yolo", exist_ok=True, seed=args.seed,
                val=True, plots=False, verbose=True)

    best = DEFAULT_RUN / "bracketbot_yolo" / "weights" / "best.pt"
    print(f"\nbest weights: {best}")
    print("run the navigator with them:  ./run_navigation.py --detector yolo")


if __name__ == "__main__":
    main()
