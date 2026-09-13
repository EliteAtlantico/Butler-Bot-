"""Offline test for LlmGoalDetector without the LLM server.

Stubs the model's answer: forward-projects the *true* goal location into the
frame's pixels (the ground truth of where the VLM *should* point), feeds that
back through the detector's pixel->world machinery, and checks the round-trip
recovers the goal. Verifies the geometry (intrinsics, cam pose, depth fusion,
bearing sign) is right, independent of model quality.

    ../.venv/bin/python test_llm_reasoner.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "main_mujoco"))
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
from bracketbot_sim.robot import BracketBot  # noqa: E402
from vision_sim import perception  # noqa: E402
from vision_sim.llm_reasoner import LlmGoalDetector  # noqa: E402

TRUE_GOAL = np.array([6.0, 0.0, 0.6])


def project_goal(obs):
    """World goal -> pixel (u, v) using this frame's cam pose + intrinsics."""
    p_cam = (obs.cam_mat.T @ (TRUE_GOAL - obs.cam_pos))
    intr = obs.intrinsics
    u = intr.fx * p_cam[0] / (-p_cam[2]) + intr.cx
    v = intr.fy * (-p_cam[1]) / (-p_cam[2]) + intr.cy
    return int(round(u)), int(round(v))


def main():
    bot = BracketBot(xml=os.path.join(HERE, "obstacle_course.xml"))
    bot.balance.enable(bot.state)
    bot.step(0.05)

    obs = perception.observe(bot, width=320, height=240, max_range=12.0)
    u, v = project_goal(obs)
    print(f"true goal {TRUE_GOAL} -> pixel ({u}, {v})  cam at {np.round(obs.cam_pos,2)}")

    det = LlmGoalDetector(query_period=0.0, verbose=True)

    # Stub the model: return the true pixel as if the VLM had read it.
    fake = {
        "scene": "indoor course with a barrier ahead and a red column beyond it",
        "goal_found": True,
        "goal_horizontal_px": u,
        "goal_vertical_px": v,
        "goal_confidence": 0.95,
        "goal_depth_cue": "mid distance, centre, above the barrier",
        "other_objects": [{"label": "barrier", "horizontal_px": u,
                           "vertical_px": 160, "note": "orange wall"}],
        "movement": "rotate slightly to keep the column centred, then advance",
    }
    det.client.ask_image = lambda rgb, prompt: (json.dumps(fake), 0.0)

    dets = det(obs, robot_yaw=bot.yaw, t=0.0)
    print(f"detections: {dets}")
    assert dets, "expected one detection"
    d = dets[0]

    pos_err = float(np.linalg.norm(d.position[:2] - TRUE_GOAL[:2]))
    print(f"position error: {pos_err:.3f} m   (depth-fused pixels={d.pixels})")
    assert pos_err < 0.5, f"position error {pos_err:.2f} m too large"
    assert d.pixels > 0, "depth fusion should have found the column's surface"
    # goal is straight ahead (+x), robot yaw ~0 -> bearing near 0
    assert abs(d.bearing) < 0.25, f"bearing {np.degrees(d.bearing):.1f} deg off axis"

    # goal_found=false path -> no detection
    fake["goal_found"] = False
    dets_none = det(obs, robot_yaw=bot.yaw, t=10.0)
    assert dets_none == [], "goal_found=false must yield no detection"
    print("goal_found=false -> no detection  OK")

    # JSON extraction robustness: fences + trailing chatter
    from vision_sim.llm_reasoner import _extract_json
    for chunk in ['```json\n{"goal_found": true}\n```',
                  'Sure! Here is the JSON:\n{"goal_found": false}  done.',
                  '{"goal_found": true}']:
        _extract_json(chunk)
    print("JSON extraction (fences/trailing)  OK")

    print(f"\nPASS  {det}")
    bot.close()


if __name__ == "__main__":
    main()
