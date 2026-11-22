"""Accuracy and consistency.

A filter is *consistent* if its errors are as large as its covariance says they should be.
The normalised estimation error squared, NEES = eᵀ Σ⁻¹ e, of a consistent filter is χ²-distributed
with 3 degrees of freedom, so its average over N Monte Carlo runs should lie in the two-sided
95 % interval of χ²(3N)/N (Bar-Shalom, Li & Kirubarajan, 2001, §5.4).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

from .models import wrap


def errors(means: np.ndarray, truth: np.ndarray) -> np.ndarray:
    e = means - truth
    e[:, 2] = wrap(e[:, 2])
    return e


def nees(means: np.ndarray, covs: np.ndarray, truth: np.ndarray) -> np.ndarray:
    e = errors(means, truth)
    return np.einsum("ti,tij,tj->t", e, np.linalg.inv(covs), e)


def nees_bounds(n_runs: int, dof: int = 3, level: float = 0.95):
    lo = chi2.ppf((1 - level) / 2, dof * n_runs) / n_runs
    hi = chi2.ppf(1 - (1 - level) / 2, dof * n_runs) / n_runs
    return lo, hi


def rmse(means: np.ndarray, truth: np.ndarray):
    """(position RMSE in metres, heading RMSE in degrees)."""
    e = errors(means, truth)
    return float(np.sqrt((e[:, 0] ** 2 + e[:, 1] ** 2).mean())), float(np.degrees(np.sqrt((e[:, 2] ** 2).mean())))
