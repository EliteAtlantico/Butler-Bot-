"""Seam D: a search that gives up hands the SAME robot to the phone remote.

Measured, not eyeballed: count BracketBot constructions, drive over HTTP, and
check that the robot the navigator was holding is the one that moved.
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main_mujoco"
VISION = ROOT / "comp_vision_sim"

pytestmark = pytest.mark.integration


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as reply:
        return reply.read()


def _post(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as reply:
        return json.loads(reply.read())


@pytest.fixture
def counted_bots():
    """Every BracketBot built while the fixture is live."""
    from bracketbot_sim.robot import BracketBot

    built = []
    real_init = BracketBot.__init__

    def counting_init(self, *args, **kwargs):
        built.append(self)
        real_init(self, *args, **kwargs)

    with mock.patch.object(BracketBot, "__init__", counting_init):
        yield BracketBot, built


def _serve_in_background(bot):
    from integration.remote_adapter import hand_off

    ready, box = threading.Event(), {}

    def on_ready(server):
        box["server"] = server
        ready.set()

    thread = threading.Thread(
        target=lambda: box.setdefault("adapter", hand_off(
            bot, host="127.0.0.1", port=0, on_ready=on_ready)),
        daemon=True)
    thread.start()
    assert ready.wait(30), "remote server never came up"
    return box, thread, f"http://127.0.0.1:{box['server'].server_address[1]}"


@pytest.mark.rendering
def test_hand_off_drives_the_one_robot_the_navigator_was_holding(counted_bots):
    BracketBot, built = counted_bots
    bot = BracketBot(xml=MAIN / "scene_flat.xml")
    try:
        bot.balance.enable(bot.state)
        bot.step(0.5)
        # The navigator has been looking through this camera all run, so its
        # renderer already exists on the main thread when the server attaches.
        bot.depth("head_depth", 64, 48)
        start = bot.position[:2].copy()

        box, thread, base = _serve_in_background(bot)
        try:
            status = json.loads(_get(base + "/api/status"))
            cameras = status["capabilities"]["cameras"]
            assert {"head_depth", "wrist_cam_left", "wrist_cam_right"} <= set(cameras)
            assert status["telemetry"]["position"][:2] == pytest.approx(start, abs=0.02)
            assert _get(base + "/api/camera?name=head_depth")[:2] == b"BM"

            for _ in range(20):                      # hold the stick; watchdog is 0.35 s
                _post(base + "/api/drive", {"linear": 1.0, "angular": 0.0})
                time.sleep(0.1)
            _post(base + "/api/stop", {})
            time.sleep(0.5)

            moved = bot.position[:2]
            assert np.linalg.norm(moved - start) > 0.1, "driving from the UI did not move the bot"
            reported = json.loads(_get(base + "/api/status"))["telemetry"]["position"][:2]
            assert reported == pytest.approx(bot.position[:2], abs=0.05)
        finally:
            box["server"].shutdown()
            thread.join(10)

        assert not thread.is_alive()
        assert box["adapter"].bot is bot
        assert len(built) == 1, f"{len(built)} BracketBots exist; the remote built its own"
        # Attached close() stops the robot but leaves its GL contexts to the owner.
        assert bot._depth_renderers, "the adapter closed a bot it does not own"
    finally:
        bot.close()


def test_standalone_adapter_still_owns_and_closes_its_bot(counted_bots):
    from remote_control.robot_adapter import SimulationRobotAdapter

    _, built = counted_bots
    adapter = SimulationRobotAdapter(MAIN / "scene_flat.xml")
    assert len(built) == 1 and adapter._owns_bot
    with mock.patch.object(adapter.bot, "close") as close:
        adapter.close()
    close.assert_called_once_with()
    adapter.bot.close()


def test_attached_adapter_does_not_close_the_bot(counted_bots):
    from remote_control.robot_adapter import SimulationRobotAdapter

    BracketBot, built = counted_bots
    bot = BracketBot(xml=MAIN / "scene_flat.xml")
    try:
        adapter = SimulationRobotAdapter.attach(bot)
        assert adapter.bot is bot and not adapter._owns_bot and len(built) == 1
        with mock.patch.object(bot, "close") as close:
            adapter.close()
        close.assert_not_called()
    finally:
        bot.close()


def _gave_up_nav(bot):
    return SimpleNamespace(
        state="stuck", done=True, give_up_reason="explored 2 place(s) without finding the goal",
        ARRIVED="arrived", goal_label="target", goal_xy=None, obs=None, best_time=0.0,
        best_detections=[], detector=None, survey=None, researches=0, explorer=None,
        outcome="gave up", bot=bot)


@pytest.mark.parametrize("extra, served", [([], True), (["--no-remote"], False)])
def test_run_navigation_hands_its_own_bot_off_after_giving_up(monkeypatch, capsys, extra, served):
    if str(VISION) not in sys.path:
        sys.path.insert(0, str(VISION))
    import run_navigation as rn
    from vision_sim.navigation import VisualNavigator

    navs = []

    def fake_for_bot(bot, **_kw):
        navs.append(_gave_up_nav(bot))
        return navs[-1]

    monkeypatch.setattr(sys, "argv", ["run_navigation.py", "--goal", "4,0",
                                      "--remote-port", "8123", *extra])
    with mock.patch.object(VisualNavigator, "for_bot", side_effect=fake_for_bot), \
            mock.patch("integration.remote_adapter.hand_off") as hand_off:
        rn.main()

    if served:
        hand_off.assert_called_once()
        assert hand_off.call_args.args[0] is navs[0].bot
        assert hand_off.call_args.kwargs["port"] == 8123
    else:
        hand_off.assert_not_called()
        assert "--no-remote" in capsys.readouterr().out
