"""Computer-vision simulation for the BracketBot.

The pipeline, in the order the data flows:

    perception     registered RGB-D -> world points -> colour detections
    yolo_detector  the same detections, from a trained YOLO net
    llm_reasoner   the same detections, from the local VLM reasoning about the frame
    yolo_dataset   auto-labelled training frames from segmentation renders
    occupancy      depth -> log-odds occupancy grid -> costmap
    planning       costmap -> A* route -> smoothed waypoints
    navigation     the closed loop that drives the robot along it

The robot model itself lives in ../main_mujoco; nothing in this package
imports it, so the vision stack can be exercised against any MuJoCo scene
that exposes a camera.
"""
