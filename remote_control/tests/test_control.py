from __future__ import annotations

import unittest

from remote_control.command_parser import parse_command
from remote_control.control import ControlState


class ControlStateTests(unittest.TestCase):
    def test_drive_scales_and_watchdog_stops_stale_command(self):
        state = ControlState(watchdog_seconds=0.35, max_linear_mps=0.4,
                             max_angular_rads=1.0)
        state.drive(0.5, -0.25, now=10.0)
        active = state.motion(now=10.2)
        self.assertAlmostEqual(active.linear_mps, 0.2)
        self.assertAlmostEqual(active.angular_rads, -0.25)

        expired = state.motion(now=10.36)
        self.assertEqual((expired.linear_mps, expired.angular_rads), (0.0, 0.0))
        self.assertIn("watchdog", expired.reason.lower())

    def test_emergency_stop_latches_until_explicit_reset(self):
        state = ControlState()
        state.drive(1.0, 0.0, now=1.0)
        state.emergency_stop()
        self.assertTrue(state.snapshot(now=1.1)["emergency_stop"])
        self.assertEqual(state.motion(now=1.1).linear_mps, 0.0)
        with self.assertRaises(RuntimeError):
            state.drive(1.0, 0.0, now=1.2)

        state.reset_emergency_stop()
        snapshot = state.snapshot(now=1.3)
        self.assertFalse(snapshot["emergency_stop"])
        self.assertEqual(snapshot["linear_mps"], 0.0)

    def test_operator_stop_is_immediate(self):
        state = ControlState()
        state.drive(0.8, 0.4, now=3.0)
        state.stop("Joystick released")
        stopped = state.motion(now=3.01)
        self.assertEqual((stopped.linear_mps, stopped.angular_rads), (0.0, 0.0))
        self.assertEqual(stopped.reason, "Joystick released")


class CommandParserTests(unittest.TestCase):
    def test_movement_commands(self):
        self.assertEqual(parse_command("move forward").action, "drive")
        self.assertGreater(parse_command("move forward").payload["linear"], 0)
        self.assertGreater(parse_command("turn left").payload["angular"], 0)
        self.assertLess(parse_command("turn right").payload["angular"], 0)
        self.assertEqual(parse_command("STOP").action, "stop")

    def test_gripper_commands_retain_side(self):
        command = parse_command("open left gripper")
        self.assertEqual(command.action, "gripper")
        self.assertEqual(command.payload, {"side": "left", "action": "open"})

    def test_unknown_command_fails_closed(self):
        with self.assertRaises(ValueError):
            parse_command("please clean the entire kitchen")


if __name__ == "__main__":
    unittest.main()

