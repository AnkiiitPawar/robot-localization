"""Simulated world: landmarks, a commanded route, the true (noisy) motion, and sensor readings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .models import measure, motion

LANDMARKS = np.array([[-6.0, -4.0], [-6.0, 4.0], [0.0, 7.0], [0.0, -7.0], [6.0, 4.0], [6.0, -4.0],
                      [-2.0, 0.5], [2.5, -1.0]])


@dataclass
class Scenario:
    dt: float = 0.1
    steps: int = 720
    sigma_v: float = 0.10          # control noise: std of executed speed (m/s)
    sigma_w: float = 0.05          # std of executed turn rate (rad/s)
    sigma_r: float = 0.15          # range noise (m)
    sigma_b: float = 0.03          # bearing noise (rad)
    max_range: float = 6.0         # sensor sees landmarks closer than this
    min_range: float = 0.0         # ...and farther than this (real range sensors have a dead zone)
    measure_every: int = 2         # sensor rate relative to the control rate
    landmarks: np.ndarray = field(default_factory=lambda: LANDMARKS.copy())

    @property
    def M(self) -> np.ndarray:
        return np.diag([self.sigma_v**2, self.sigma_w**2])

    @property
    def Q(self) -> np.ndarray:
        return np.diag([self.sigma_r**2, self.sigma_b**2])


def figure_eight_controls(steps: int, dt: float, v: float = 1.0, w: float = 0.35) -> np.ndarray:
    """(v, ω) commands that trace a figure of eight: one full left circle, then one full right
    circle (radius v/ω), repeated."""
    loop = 2 * np.pi / w                     # seconds per circle
    t = (np.arange(steps) + 0.5) * dt
    turn = np.where((t // loop) % 2 == 0, w, -w)
    return np.stack([np.full(steps, v), turn], 1)


@dataclass
class Run:
    truth: np.ndarray                 # (steps + 1, 3)
    controls: np.ndarray              # (steps, 2), the commanded (not executed) controls
    observations: list                # per step: list of (landmark index, z)


def simulate(sc: Scenario, rng: np.random.Generator, start=(0.0, 0.0, 0.0)) -> Run:
    controls = figure_eight_controls(sc.steps, sc.dt)
    truth = [np.array(start, float)]
    obs = []
    for k in range(sc.steps):
        executed = controls[k] + rng.normal(0, [sc.sigma_v, sc.sigma_w])
        x = motion(truth[-1], executed, sc.dt)
        truth.append(x)
        seen = []
        if k % sc.measure_every == 0:
            for i, lm in enumerate(sc.landmarks):
                z = measure(x, lm)
                if sc.min_range < z[0] < sc.max_range:
                    seen.append((i, z + rng.normal(0, [sc.sigma_r, sc.sigma_b])))
        obs.append(seen)
    return Run(np.array(truth), controls, obs)


def run_filter(filt, run: Run, sc: Scenario):
    """Feed a recorded run to a filter. Returns (means, covariances), one per time step after the start."""
    means, covs = [], []
    for k, u in enumerate(run.controls):
        filt.predict(u, sc.dt)
        for i, z in run.observations[k]:
            filt.update(z, sc.landmarks[i])
        means.append(filt.mean.copy())
        covs.append(filt.cov.copy())
    return np.array(means), np.array(covs)
