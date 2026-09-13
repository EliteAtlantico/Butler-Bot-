"""Seam D: the search gave up, so hand the live robot to the phone remote.

    from integration.remote_adapter import hand_off

    hand_off(bot)          # prints the URL, blocks until Ctrl+C

The whole seam is one fact: the operator must drive the robot that got lost,
not a second one built at the origin of a different scene. So the remote's
adapter attaches to the running `BracketBot` instead of constructing its own,
and the server's control thread steps it from then on.

That is only safe because nothing else is stepping it. Call this after the
navigator's loop has exited -- never from inside `on_give_up`, which fires in
the middle of `bot.step()`.

Nothing in `remote_control/` is subclassed. The caller keeps ownership of the
bot and closes it afterwards; the adapter only stops it.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def hand_off(bot, *, host: str = "0.0.0.0", port: int = 8000,
             watchdog_seconds: float = 0.35, on_ready=None):
    """Serve the phone remote for `bot` until the operator presses Ctrl+C.

    `on_ready(server)` is passed through to `remote_control.server.serve`; it
    is how a test learns the bound port and stops the server. Returns the
    adapter, so a caller can confirm which robot was driven.
    """
    from remote_control.robot_adapter import SimulationRobotAdapter
    from remote_control.server import serve

    adapter = SimulationRobotAdapter.attach(bot)
    serve(adapter, host, port, watchdog_seconds=watchdog_seconds, on_ready=on_ready)
    return adapter
