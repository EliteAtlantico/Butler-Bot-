# BracketBot mobile remote

This module serves a phone-friendly control page on the local network. It is
kept separate from `main_mujoco` and calls that package's existing public robot
methods:

- drive: `BracketBot.drive(v_mps, w_rads)`
- arm: `BracketBot.set_arm_target(joint, value)`
- gripper: the model's existing coupled gripper joints
- camera: `BracketBot.camera(name)`

The server currently runs the MuJoCo robot because the repository does not
contain the referenced physical `lib.odrive_uart.ODriveUART` module. The remote
and safety state are isolated from the simulator in `robot_adapter.py`, so a
hardware adapter can be connected later without changing the phone UI or HTTP
handlers.

## Start the remote

From the repository root:

```powershell
python -m pip install -r remote_control\requirements.txt
python -m remote_control.server
```

The server prints two addresses. Open the `Phone` address on a phone connected
to the same Wi-Fi network. If Windows Firewall asks, allow Python on private
networks.

To find the address manually:

```powershell
ipconfig
```

Use the IPv4 address of the active Wi-Fi adapter, for example
`http://192.168.1.42:8000`.

## Safe test sequence

1. Put the robot or simulation in a clear area and keep the emergency-stop
   control visible.
2. Touch the joystick briefly at low deflection, then release it. The current
   command must change to `Joystick released` and motion must stop.
3. While moving slowly, press `EMERGENCY STOP`. New drive, arm, and gripper
   commands remain blocked until `RESET E-STOP` is pressed.
4. Close or background the phone page while commanding motion. The 350 ms
   server watchdog must stop the robot.
5. Test arm sliders and grippers only after drive stop behavior passes.

The joystick sends at most about 11 commands per second. Pointer release,
pointer cancellation, page hiding, and page exit all request an immediate stop;
the server watchdog is the independent fallback.

## Tests

```powershell
python -m unittest discover -s remote_control\tests -v
node --check remote_control\static\app.js
```

