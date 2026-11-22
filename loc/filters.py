"""Three Bayes filters for the same problem: EKF, UKF and a particle filter (MCL).

All share the interface ``predict(u, dt)``, ``update(z, landmark)``, and expose ``mean`` and
``cov`` so they can be scored the same way.
"""

from __future__ import annotations

import numpy as np

from .models import measure, measurement_jacobian, motion, motion_jacobians, wrap


class EKF:
    """Extended Kalman filter (Probabilistic Robotics, Table 7.2).

    Predict:  μ ← g(μ, u),            Σ ← G Σ Gᵀ + V M Vᵀ
    Update:   K = Σ Hᵀ (H Σ Hᵀ + Q)⁻¹, μ ← μ + K (z - h(μ)),  Σ ← (I - K H) Σ (I - K H)ᵀ + K Q Kᵀ
    The Joseph form of the covariance update keeps Σ symmetric and positive definite.
    """

    def __init__(self, mean, cov, M, Q) -> None:
        self.mean, self.cov = np.array(mean, float), np.array(cov, float)
        self.M, self.Q = np.asarray(M, float), np.asarray(Q, float)

    def predict(self, u, dt: float) -> None:
        G, V = motion_jacobians(self.mean, u, dt)
        self.mean = motion(self.mean, u, dt)
        self.cov = G @ self.cov @ G.T + V @ self.M @ V.T

    gate: float | None = None  # reject updates whose innovation is this unlikely (χ²₂ threshold)
    rejected: int = 0

    def update(self, z, landmark) -> None:
        H = measurement_jacobian(self.mean, landmark)
        innov = np.asarray(z) - measure(self.mean, landmark)
        innov[1] = wrap(innov[1])
        S = H @ self.cov @ H.T + self.Q
        if self.gate is not None and innov @ np.linalg.solve(S, innov) > self.gate:
            self.rejected += 1
            return
        K = self.cov @ H.T @ np.linalg.inv(S)
        self.mean = self.mean + K @ innov
        self.mean[2] = wrap(self.mean[2])
        A = np.eye(3) - K @ H
        self.cov = A @ self.cov @ A.T + K @ self.Q @ K.T


def sigma_points(mean, cov, alpha=0.5, beta=2.0, kappa=0.0):
    """Scaled unscented transform (van der Merwe): 2n+1 points and their mean/cov weights."""
    n = len(mean)
    lam = alpha**2 * (n + kappa) - n
    A = (n + lam) * 0.5 * (cov + cov.T)
    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        # Rounding can leave a covariance very slightly indefinite after a large update; use
        # the symmetric square root with negative eigenvalues clipped instead.
        w, U = np.linalg.eigh(A)
        L = U * np.sqrt(np.clip(w, 1e-12, None))
    pts = np.vstack([mean, mean + L.T, mean - L.T])
    wm = np.full(2 * n + 1, 1 / (2 * (n + lam)))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + 1 - alpha**2 + beta
    return pts, wm, wc


def angle_mean(angles, w):
    return np.arctan2((w * np.sin(angles)).sum(), (w * np.cos(angles)).sum())


class UKF:
    """Unscented Kalman filter. Process noise enters additively as V M Vᵀ at the mean, and
    angles are averaged on the circle."""

    def __init__(self, mean, cov, M, Q, alpha=0.5) -> None:
        self.mean, self.cov = np.array(mean, float), np.array(cov, float)
        self.M, self.Q, self.alpha = np.asarray(M, float), np.asarray(Q, float), alpha

    def predict(self, u, dt: float) -> None:
        pts, wm, wc = sigma_points(self.mean, self.cov, self.alpha)
        prop = motion(pts, u, dt)
        mean = np.array([wm @ prop[:, 0], wm @ prop[:, 1], angle_mean(prop[:, 2], wm)])
        d = prop - mean
        d[:, 2] = wrap(d[:, 2])
        _, V = motion_jacobians(self.mean, u, dt)
        self.mean = mean
        self.cov = (wc[:, None] * d).T @ d + V @ self.M @ V.T

    def update(self, z, landmark) -> None:
        pts, wm, wc = sigma_points(self.mean, self.cov, self.alpha)
        Z = measure(pts, landmark)
        z_mean = np.array([wm @ Z[:, 0], angle_mean(Z[:, 1], wm)])
        dz = Z - z_mean
        dz[:, 1] = wrap(dz[:, 1])
        dx = pts - self.mean
        dx[:, 2] = wrap(dx[:, 2])
        S = (wc[:, None] * dz).T @ dz + self.Q
        C = (wc[:, None] * dx).T @ dz
        K = C @ np.linalg.inv(S)
        innov = np.asarray(z) - z_mean
        innov[1] = wrap(innov[1])
        self.mean = self.mean + K @ innov
        self.mean[2] = wrap(self.mean[2])
        self.cov = self.cov - K @ S @ K.T


def low_variance_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling (Probabilistic Robotics, Table 4.4): one random offset, N evenly
    spaced pointers. A particle with weight w is copied ⌊Nw⌋ or ⌈Nw⌉ times."""
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(weights), positions)
    return np.minimum(idx, n - 1)


class ParticleFilter:
    """Monte Carlo localisation: sample the motion model with noisy controls, weight by the
    measurement likelihood, resample when the effective sample size drops below N/2."""

    def __init__(self, particles: np.ndarray, M, Q, rng: np.random.Generator, roughen: float = 0.0,
                 inject_bounds=None, alpha_slow: float = 0.001, alpha_fast: float = 0.1) -> None:
        self.p = np.array(particles, float)
        self.w = np.full(len(self.p), 1.0 / len(self.p))
        self.M, self.Q, self.rng = np.asarray(M, float), np.asarray(Q, float), rng
        self.roughen = roughen
        self.resamples = 0
        # Augmented MCL (Probabilistic Robotics, Table 8.3): track short- and long-term averages
        # of the measurement likelihood; when the short-term one drops, inject random poses.
        self.bounds, self.a_slow, self.a_fast = inject_bounds, alpha_slow, alpha_fast
        self.log_slow = self.log_fast = None
        self.injected = 0

    @classmethod
    def gaussian(cls, mean, cov, n, M, Q, rng, roughen: float = 0.0):
        return cls(rng.multivariate_normal(mean, cov, size=n), M, Q, rng, roughen)

    @classmethod
    def uniform(cls, lo, hi, n, M, Q, rng, roughen: float = 0.0, inject: bool = False):
        """Global localisation: no idea where the robot is. ``inject`` turns on augmented MCL."""
        pts = rng.uniform([lo[0], lo[1], -np.pi], [hi[0], hi[1], np.pi], size=(n, 3))
        return cls(pts, M, Q, rng, roughen, inject_bounds=(lo, hi) if inject else None)

    def random_poses(self, k: int) -> np.ndarray:
        lo, hi = self.bounds
        return self.rng.uniform([lo[0], lo[1], -np.pi], [hi[0], hi[1], np.pi], size=(k, 3))

    def predict(self, u, dt: float) -> None:
        noisy = np.asarray(u) + self.rng.multivariate_normal(np.zeros(2), self.M, size=len(self.p))
        self.p = motion(self.p, noisy, dt)

    def update(self, z, landmark) -> None:
        pred = measure(self.p, landmark)
        dr = z[0] - pred[:, 0]
        db = wrap(z[1] - pred[:, 1])
        logl = -0.5 * (dr**2 / self.Q[0, 0] + db**2 / self.Q[1, 1])
        inject = 0.0
        if self.bounds is not None:
            m = logl.max()
            log_avg = m + np.log(np.sum(self.w * np.exp(logl - m)))  # log of Σ wᵢ p(z | xᵢ)
            if self.log_slow is None:
                self.log_slow = self.log_fast = log_avg
            self.log_slow = np.logaddexp(np.log1p(-self.a_slow) + self.log_slow, np.log(self.a_slow) + log_avg)
            self.log_fast = np.logaddexp(np.log1p(-self.a_fast) + self.log_fast, np.log(self.a_fast) + log_avg)
            inject = max(0.0, 1.0 - np.exp(self.log_fast - self.log_slow))
        w = self.w * np.exp(logl - logl.max())
        self.w = w / w.sum()
        if inject > 0.01 or 1.0 / np.sum(self.w**2) < len(self.p) / 2:
            idx = low_variance_resample(self.w, self.rng)
            self.p, self.w = self.p[idx], np.full(len(self.p), 1.0 / len(self.p))
            self.resamples += 1
            k = self.rng.binomial(len(self.p), inject) if inject > 0.01 else 0
            if k:
                self.p[self.rng.choice(len(self.p), k, replace=False)] = self.random_poses(k)
                self.injected += k
            if self.roughen:
                # Roughening (Gordon, Salmond & Smith, 1993): jitter each coordinate by
                # K · (spread of the particles) · N^(-1/d) so duplicates separate again.
                spread = self.p.max(axis=0) - self.p.min(axis=0)
                spread[2] = min(spread[2], np.pi)
                self.p = self.p + self.rng.normal(0, 1, self.p.shape) * self.roughen * spread * len(self.p) ** (-1 / 3)
                self.p[:, 2] = wrap(self.p[:, 2])

    @property
    def mean(self) -> np.ndarray:
        return np.array([self.w @ self.p[:, 0], self.w @ self.p[:, 1], angle_mean(self.p[:, 2], self.w)])

    @property
    def cov(self) -> np.ndarray:
        d = self.p - self.mean
        d[:, 2] = wrap(d[:, 2])
        return (self.w[:, None] * d).T @ d + 1e-9 * np.eye(3)
