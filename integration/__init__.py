"""Adapters between the subsystems. Glue only -- no behaviour of its own.

Each module here joins two stacks that were written independently and should
stay that way. Nothing in `comp_vision_sim`, `Hand_and_Wrists`, `main_mujoco`
or `remote_control` imports this package; the dependency runs one way, so
either side can still be used on its own.
"""
