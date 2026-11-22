import numpy as np
import pytest
from scipy.stats import chi2

from loc import EKF, UKF, ParticleFilter, Scenario, run_filter, simulate
from loc.filters import low_variance_resample, sigma_points
from loc.metrics import nees, nees_bounds, rmse
from loc.models import measure, measurement_jacobian, motion, motion_jacobians, wrap


def num_jac(f, x, h=1e-7):
    x = np.asarray(x, float)
    cols = []
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = h
        d = f(x + e) - f(x - e)
        if d.shape[0] == 3:
            d[2] = wrap(d[2])
        elif d.shape[0] == 2:
            d[1] = wrap(d[1])
        cols.append(d / (2 * h))
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("u", [(1.2, 0.4), (0.8, -0.9), (1.0, 0.0)])
def test_motion_jacobians(u):
    x = np.array([0.3, -1.2, 0.7])
    G, V = motion_jacobians(x, u, 0.1)
    assert np.allclose(G, num_jac(lambda s: motion(s, u, 0.1), x), atol=1e-6)
    # At ω = 0 a tiny step puts the arc formula (radius v/ω) into catastrophic cancellation,
    # so differentiate with a larger step there; the O(h²) error is still about 1e-8.
    h = 1e-3 if u[1] == 0 else 1e-7
    assert np.allclose(V, num_jac(lambda uu: motion(x, uu, 0.1), np.array(u, float), h), atol=1e-5)


def test_measurement_jacobian():
    x, lm = np.array([1.0, 2.0, 0.4]), np.array([-2.0, 3.5])
    assert np.allclose(measurement_jacobian(x, lm), num_jac(lambda s: measure(s, lm), x), atol=1e-6)


def test_motion_is_continuous_as_turn_rate_goes_to_zero():
    x = np.array([0.0, 0.0, 0.3])
    assert np.allclose(motion(x, (1.0, 1e-7), 0.5), motion(x, (1.0, 0.0), 0.5), atol=1e-7)


def test_a_full_circle_returns_to_the_start():
    x = np.array([1.0, -2.0, 0.5])
    w = 0.5
    y = x
    for _ in range(100):
        y = motion(y, (1.0, w), 2 * np.pi / w / 100)
    assert np.allclose(y, x, atol=1e-9)


def test_wrap():
    a = wrap(np.array([np.pi, -np.pi, 3 * np.pi + 0.1, -0.2]))
    assert np.all((a >= -np.pi) & (a < np.pi))
    assert np.isclose(a[2], -np.pi + 0.1) and np.isclose(a[3], -0.2)


def test_sigma_points_reproduce_mean_and_covariance():
    mean = np.array([1.0, -2.0, 0.3])
    A = np.array([[0.5, 0.1, 0.0], [0.1, 0.3, 0.05], [0.0, 0.05, 0.2]])
    pts, wm, wc = sigma_points(mean, A)
    assert np.isclose(wm.sum(), 1.0)
    assert np.allclose(wm @ pts, mean)
    d = pts - mean
    assert np.allclose((wc[:, None] * d).T @ d, A)


def test_low_variance_resampling_copies_in_proportion():
    w = np.array([0.5, 0.25, 0.125, 0.125])
    for seed in range(20):
        counts = np.bincount(low_variance_resample(w, np.random.default_rng(seed)), minlength=4)
        assert np.all(counts >= np.floor(4 * w)) and np.all(counts <= np.ceil(4 * w))


def test_nees_of_correctly_sized_errors_sits_in_the_chi_square_band():
    rng = np.random.default_rng(0)
    P = np.diag([0.04, 0.09, 0.01])
    runs = 50
    e = rng.multivariate_normal(np.zeros(3), P, size=(runs, 200))
    n = np.einsum("rti,ij,rtj->rt", e, np.linalg.inv(P), e).mean(axis=0)
    lo, hi = nees_bounds(runs)
    assert lo < 3 < hi
    assert np.mean((n >= lo) & (n <= hi)) > 0.9  # about 95 % expected


@pytest.mark.parametrize("name", ["EKF", "UKF", "PF"])
def test_every_filter_localises(name):
    sc = Scenario(steps=300, min_range=0.3)
    rng = np.random.default_rng(3)
    run = simulate(sc, rng)
    P0 = np.diag([0.09, 0.09, np.radians(5) ** 2])
    x0 = run.truth[0] + rng.multivariate_normal(np.zeros(3), P0)
    f = {"EKF": lambda: EKF(x0, P0, sc.M, sc.Q), "UKF": lambda: UKF(x0, P0, sc.M, sc.Q),
         "PF": lambda: ParticleFilter.gaussian(x0, P0, 1000, sc.M, sc.Q, np.random.default_rng(4))}[name]()
    m, c = run_filter(f, run, sc)
    pos, head = rmse(m, run.truth[1:])
    assert pos < 0.1 and head < 2.0
    assert nees(m, c, run.truth[1:]).mean() < 10  # one run is noisy; consistency is judged over 20 (README)


def test_ekf_blows_up_near_a_landmark_and_a_sensor_dead_zone_fixes_it():
    """Seed 2's route passes 8 cm from a landmark: the bearing becomes so nonlinear that the
    EKF's update is badly overconfident. Ignoring readings closer than 0.3 m prevents it."""
    P0 = np.diag([0.09, 0.09, np.radians(5) ** 2])
    out = []
    for sc in (Scenario(), Scenario(min_range=0.3)):
        rng = np.random.default_rng(2)
        run = simulate(sc, rng)
        x0 = run.truth[0] + rng.multivariate_normal(np.zeros(3), P0)
        m, c = run_filter(EKF(x0, P0, sc.M, sc.Q), run, sc)
        out.append(nees(m, c, run.truth[1:]))
    assert out[0].max() > 1e4 and out[0].mean() > 50
    assert out[1].mean() < 5


def test_roughening_prevents_particle_collapse():
    """A 1 m, 30 degree prior and an accurate first reading leave only a handful of the 1,000
    particles with real weight. Without roughening the filter's covariance collapses."""
    sc = Scenario(sigma_b=0.1, sigma_w=0.15, sigma_v=0.2, max_range=4.0, measure_every=5)
    P0 = np.diag([1.0, 1.0, np.radians(30) ** 2])
    rng = np.random.default_rng(9)
    run = simulate(sc, rng)
    x0 = run.truth[0] + rng.multivariate_normal(np.zeros(3), P0)
    res = []
    for K in (0.0, 0.2):
        pf = ParticleFilter.gaussian(x0, P0, 1000, sc.M, sc.Q, np.random.default_rng(109), roughen=K)
        m, c = run_filter(pf, run, sc)
        res.append(nees(m, c, run.truth[1:]).mean())
    assert res[0] > 50 and res[1] < 6


def test_gate_rejects_a_gross_outlier():
    sc = Scenario()
    f = EKF(np.zeros(3), np.diag([0.01, 0.01, 0.001]), sc.M, sc.Q)
    f.gate = chi2.ppf(0.999, 2)
    lm = np.array([3.0, 0.0])
    f.update(np.array([10.0, 2.0]), lm)  # true reading would be about (3, 0)
    assert f.rejected == 1 and np.allclose(f.mean, 0)
    f.update(np.array([3.05, 0.01]), lm)
    assert f.rejected == 1 and not np.allclose(f.mean, 0)


def test_particle_filter_finds_the_robot_from_no_prior():
    sc = Scenario(steps=200)
    run = simulate(sc, np.random.default_rng(1))
    pf = ParticleFilter.uniform((-8, -8), (8, 8), 5000, sc.M, sc.Q, np.random.default_rng(8))
    m, _ = run_filter(pf, run, sc)
    err = np.hypot(*(m[-50:, :2] - run.truth[-50:, :2]).T)
    assert err.mean() < 0.15


def test_augmented_mcl_recovers_where_plain_mcl_locks_onto_the_wrong_pose():
    """Seed 3 with particle seed 10: plain MCL collapses onto a wrong pose in the first update
    and is still wrong 8 s later; injecting random particles when the likelihood drops fixes it."""
    sc = Scenario(steps=80)
    run = simulate(sc, np.random.default_rng(3))
    final = {}
    for inject in (False, True):
        pf = ParticleFilter.uniform((-8, -8), (8, 8), 5000, sc.M, sc.Q, np.random.default_rng(10), inject=inject)
        m, _ = run_filter(pf, run, sc)
        final[inject] = float(np.hypot(*(m[-1, :2] - run.truth[-1, :2])))
    assert final[False] > 1.0
    assert final[True] < 0.3
