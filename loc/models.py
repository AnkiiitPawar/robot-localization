"""Motion and measurement models for a planar robot, with their Jacobians.

State x = (x, y, θ). Control u = (v, ω): forward speed and turn rate, held for dt.

Velocity motion model (Thrun, Burgard & Fox, *Probabilistic Robotics*, §5.3): the robot moves
along a circular arc of radius v/ω, so
    x' = x - (v/ω) sin θ + (v/ω) sin(θ + ω dt)
    y' = y + (v/ω) cos θ - (v/ω) cos(θ + ω dt)
    θ' = θ + ω dt
and along a straight line when ω → 0.

Range-bearing measurement of landmark m = (mₓ, m_y):
    r = √(Δx² + Δy²),  φ = atan2(Δy, Δx) - θ,   Δ = m - (x, y).
"""

from __future__ import annotations

import numpy as np

EPS_OMEGA = 1e-9


def wrap(a):
    """Map angles to [-π, π)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def motion(state: np.ndarray, u, dt: float) -> np.ndarray:
    """Apply the velocity model to one state (3,) or many states (n, 3) with controls u (2,) or (n, 2)."""
    s = np.atleast_2d(state).astype(float)
    uu = np.broadcast_to(np.asarray(u, float), (len(s), 2))
    v, w = uu[:, 0], uu[:, 1]
    x, y, th = s[:, 0], s[:, 1], s[:, 2]
    out = np.empty_like(s)
    turning = np.abs(w) > EPS_OMEGA
    ws = np.where(turning, w, 1.0)
    r = v / ws
    out[:, 0] = np.where(turning, x - r * np.sin(th) + r * np.sin(th + w * dt), x + v * dt * np.cos(th))
    out[:, 1] = np.where(turning, y + r * np.cos(th) - r * np.cos(th + w * dt), y + v * dt * np.sin(th))
    out[:, 2] = wrap(th + w * dt)
    return out[0] if np.ndim(state) == 1 else out


def motion_jacobians(state: np.ndarray, u, dt: float):
    """G = ∂g/∂state (3×3) and V = ∂g/∂u (3×2) at one state."""
    _, _, th = state
    v, w = u
    if abs(w) > EPS_OMEGA:
        s0, c0 = np.sin(th), np.cos(th)
        s1, c1 = np.sin(th + w * dt), np.cos(th + w * dt)
        G = np.array([[1, 0, (v / w) * (-c0 + c1)],
                      [0, 1, (v / w) * (-s0 + s1)],
                      [0, 0, 1]])
        V = np.array([[(-s0 + s1) / w, v * (s0 - s1) / w**2 + v * c1 * dt / w],
                      [(c0 - c1) / w, -v * (c0 - c1) / w**2 + v * s1 * dt / w],
                      [0, dt]])
    else:
        c0, s0 = np.cos(th), np.sin(th)
        G = np.array([[1, 0, -v * dt * s0], [0, 1, v * dt * c0], [0, 0, 1]])
        V = np.array([[dt * c0, -0.5 * v * dt**2 * s0], [dt * s0, 0.5 * v * dt**2 * c0], [0, dt]])
    return G, V


def measure(state: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """(range, bearing) of a landmark from one state (3,) or many states (n, 3)."""
    s = np.atleast_2d(state)
    d = landmark[None, :2] - s[:, :2]
    out = np.stack([np.hypot(d[:, 0], d[:, 1]), wrap(np.arctan2(d[:, 1], d[:, 0]) - s[:, 2])], axis=1)
    return out[0] if np.ndim(state) == 1 else out


def measurement_jacobian(state: np.ndarray, landmark: np.ndarray) -> np.ndarray:
    """H = ∂(r, φ)/∂(x, y, θ)."""
    dx, dy = landmark[0] - state[0], landmark[1] - state[1]
    q = dx * dx + dy * dy
    r = np.sqrt(q)
    return np.array([[-dx / r, -dy / r, 0.0], [dy / q, -dx / q, -1.0]])
