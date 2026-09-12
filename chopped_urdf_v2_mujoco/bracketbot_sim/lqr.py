"""LQR balance gains for the BracketBot, ported from the real robot's code.

This is BracketBotCapstone/quickstart `lib/lqr.py`, with two changes:

  * the plant parameters are arguments instead of module-level constants, so
    the gains can be derived from whatever the simulated body actually weighs
    (see `plant.py`) rather than from the hard-coded hardware numbers
  * `control.lqr` is replaced by `scipy.linalg.solve_continuous_are`, which is
    the same computation without the extra dependency

State is [x, x_dot, pitch, pitch_rate, yaw, yaw_rate]; input is
[pitch_torque, yaw_torque].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.linalg import solve_continuous_are

G = 9.807


@dataclass
class PlantParams:
    """Physical parameters of the two-wheel inverted pendulum."""
    Jr: float = 0.018     # wheel spin inertia [kg m^2]
    Mr: float = 2.2       # wheel mass [kg]
    Jpth: float = 1.46    # chassis pitch inertia about the CoM [kg m^2]
    Jpd: float = 0.041    # chassis yaw inertia [kg m^2]
    Mp: float = 3.38      # chassis (sprung) mass [kg]
    R: float = 0.0846     # wheel radius [m]
    D: float = 0.46       # track width [m]
    L: float = 0.367      # axle -> CoM distance [m]
    trim: float = 0.0     # pitch at which the CoM sits over the axle [rad]

    def describe(self):
        return (f"Mp={self.Mp:.3f}kg  L={self.L:.4f}m  Jpth={self.Jpth:.4f}  "
                f"Jpd={self.Jpd:.4f}  Mr={self.Mr:.2f}kg  Jr={self.Jr:.4f}  "
                f"R={self.R:.4f}m  D={self.D:.4f}m  "
                f"trim={np.rad2deg(self.trim):+.3f}deg")


def state_space(p: PlantParams):
    """Linearised A, B about the upright equilibrium.

    Kept in the original symbolic form so the model matches the hardware
    controller exactly -- the small-angle substitutions (cos t ~ 1, sin t ~ t)
    and the `theta**2 -> 0` truncation are BracketBot's, not ours.
    """
    theta = sp.Symbol("theta")

    M_pitch = sp.Matrix([
        [-1, 0, 0, -1 / (2 * p.Mr), 1 / (2 * p.Mr)],
        [-1, -p.L, 0, 1 / p.Mp, 0],
        [0, p.L * theta, 1 / p.Mp, 0, 0],
        [0, -1, p.L * theta / p.Jpth, -p.L / p.Jpth, 0],
        [-1, 0, 0, 0, -p.R ** 2 / (2 * p.Jr)],
    ])
    f_R, f_P, Cth = sp.symbols("f_R f_P Cth")
    b = sp.Matrix([-f_R / (2 * p.Mr), -f_P / p.Mp, G,
                   Cth / p.Jpth, -p.R * Cth / (2 * p.Jr)])
    q = (M_pitch.inv() * b).subs(theta ** 2, 0)

    A23 = float(q[0].diff(theta).evalf())
    A43 = float(q[1].diff(theta).evalf())
    B21 = float(q[0].diff(Cth).evalf())
    B41 = float(q[1].diff(Cth).evalf())

    M_yaw = sp.Matrix([
        [1, -p.D / (2 * p.Jpd), 0],
        [-2 / p.D, -1 / p.Mr, 1 / p.Mr],
        [-2 / p.D, 0, -p.R ** 2 / p.Jr],
    ])
    f_d, Cd = sp.symbols("f_d Cd")
    q = M_yaw.inv() * sp.Matrix([0, f_d / p.Mr, -Cd * p.R / p.Jr])
    B62 = float(q[0].diff(Cd).evalf())

    A = np.array([
        [0, 1, 0, 0, 0, 0],
        [0, 0, A23, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, A43, 0, 0, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0],
    ], dtype=float)
    B = np.array([[0, 0], [B21, 0], [0, 0], [B41, 0], [0, 0], [0, B62]],
                 dtype=float)
    return A, B


def LQR_gains(Q_diag, R_diag, p: PlantParams | None = None):
    """Infinite-horizon continuous LQR gain K (2x6), u = -K x."""
    p = p or PlantParams()
    A, B = state_space(p)
    Q = np.diag(np.asarray(Q_diag, float))
    R = np.diag(np.asarray(R_diag, float))
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.solve(R, B.T @ P)
