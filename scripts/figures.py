"""README figures. usage: python scripts/figures.py   (after run_experiments.py; needs matplotlib)"""

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loc import EKF, UKF, ParticleFilter, Scenario, run_filter, simulate  # noqa: E402
from loc.metrics import errors  # noqa: E402

DOCS = Path("docs")
COLORS = {"EKF": "#c0504d", "UKF": "#2e8b57", "PF 1,000": "#8064a2", "PF 1,000 + roughening": "#1f5f99",
          "PF 5,000": "#e6a23c"}


def ellipse(ax, mean, cov, k, **kw):
    w, v = np.linalg.eigh(cov[:2, :2])
    ang = np.degrees(np.arctan2(v[1, 1], v[0, 1]))
    ax.add_patch(Ellipse(mean[:2], 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fill=False, **kw))


def trajectory():
    sc = Scenario(min_range=0.3)
    rng = np.random.default_rng(0)
    run = simulate(sc, rng)
    P0 = np.diag([0.09, 0.09, np.radians(5) ** 2])
    x0 = run.truth[0] + rng.multivariate_normal(np.zeros(3), P0)
    est = {"EKF": run_filter(EKF(x0, P0, sc.M, sc.Q), run, sc),
           "UKF": run_filter(UKF(x0, P0, sc.M, sc.Q), run, sc),
           "PF 1,000 + roughening": run_filter(ParticleFilter.gaussian(x0, P0, 1000, sc.M, sc.Q,
                                                                       np.random.default_rng(100), roughen=0.2), run, sc)}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.4]})
    a.plot(*run.truth[:, :2].T, color="k", lw=2.5, alpha=0.25, label="true path")
    for name, (m, _) in est.items():
        a.plot(*m[:, :2].T, color=COLORS[name], lw=0.9, label=name)
    m, c = est["EKF"]
    for k in range(0, len(m), 45):
        ellipse(a, m[k], c[k], 3 * 10, color=COLORS["EKF"], lw=0.7)
    a.plot(*sc.landmarks.T, "*", color="#333", ms=11, label="landmarks")
    a.set(aspect="equal", xlim=(-7.5, 7.5), ylim=(-8, 8), title="Figure-eight route (EKF 3σ ellipses drawn 10× larger)")
    a.legend(fontsize=7, loc="upper right", framealpha=0.9)
    t = np.arange(1, len(m) + 1) * sc.dt
    for name, (m, c) in est.items():
        e = errors(m, run.truth[1:])
        b.plot(t, 100 * np.hypot(e[:, 0], e[:, 1]), color=COLORS[name], lw=0.9, label=name)
    sig = 100 * 2 * np.sqrt(est["EKF"][1][:, 0, 0] + est["EKF"][1][:, 1, 1])
    b.fill_between(t, 0, sig, color=COLORS["EKF"], alpha=0.1, label="EKF 2σ (position)")
    b.set(xlabel="time (s)", ylabel="position error (cm)", ylim=(0, 30), title="Error over one run")
    b.legend(fontsize=7, frameon=False)
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "trajectory.png", dpi=120)


def consistency(res):
    lo, hi = res["nees_band"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.3))
    names = list(res["normal_nees_runs"])
    for i, name in enumerate(names):
        v = np.array(res["normal_nees_runs"][name])
        a.scatter(np.full(len(v), i) + np.linspace(-0.15, 0.15, len(v)), v, s=14, color=COLORS[name])
    a.axhspan(lo, hi, color="#2e8b57", alpha=0.12)
    a.axhline(3, color="#2e8b57", lw=0.8)
    a.set(yscale="log", xticks=range(len(names)), ylabel="mean NEES of one run (log scale)",
          title=f"Normal scenario, {res['runs']} runs (band: 95 % χ² range for a consistent filter)")
    a.set_xticklabels([n.replace(" + ", "\n+ ") for n in names], fontsize=8)
    seed = res["normal_ekf_worst"]["seed"]
    n = np.array(res["normal_ekf_worst"]["nees_t"])
    sc = Scenario()
    run = simulate(sc, np.random.default_rng(seed))
    d = np.array([min(np.hypot(*(lm - x[:2])) for lm in sc.landmarks) for x in run.truth[1:]])
    t = np.arange(1, len(n) + 1) * sc.dt
    b.semilogy(t, n, color=COLORS["EKF"], lw=1, label="EKF NEES")
    b.axhline(3, color="#2e8b57", lw=0.8)
    b.set(xlabel="time (s)", ylabel="NEES (log scale)", title=f"The EKF's bad run (seed {seed})")
    b2 = b.twinx()
    b2.plot(t, d, color="#555", lw=0.8, ls="--", label="distance to nearest landmark")
    b2.set_ylabel("nearest landmark (m)")
    k = int(np.argmax(n))
    b.annotate(f"robot passes {100 * d[k]:.0f} cm from a landmark", (t[k], n[k]), (t[k] + 6, n[k] / 3),
               arrowprops=dict(arrowstyle="->", lw=0.7), fontsize=8)
    b.legend(loc="upper left", fontsize=8, frameon=False)
    b2.legend(loc="upper right", fontsize=8, frameon=False)
    a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(DOCS / "consistency.png", dpi=120)


def global_localisation(seed: int = 3, pf_seed: int = 10):
    sc = Scenario()
    run = simulate(sc, np.random.default_rng(seed))
    shots = [0, 3, 10, 40, 80]
    fig, axes = plt.subplots(2, len(shots), figsize=(14, 6.2))
    for row, (inject, label) in enumerate(((False, "plain MCL"), (True, "augmented MCL"))):
        pf = ParticleFilter.uniform((-8, -8), (8, 8), 5000, sc.M, sc.Q, np.random.default_rng(pf_seed), inject=inject)
        snaps = {}
        for k, u in enumerate(run.controls[: shots[-1] + 1]):
            if k in shots:
                snaps[k] = pf.p.copy()
            pf.predict(u, sc.dt)
            for i, z in run.observations[k]:
                pf.update(z, sc.landmarks[i])
        for ax, k in zip(axes[row], shots):
            ax.scatter(*snaps[k][:, :2].T, s=1, color="#1f5f99", alpha=0.4)
            ax.plot(*sc.landmarks.T, "*", color="#333", ms=8)
            ax.plot(*run.truth[k, :2], "o", mfc="none", mec="#c0504d", ms=12, mew=2)
            ax.set(aspect="equal", xlim=(-8, 8), ylim=(-8, 8), xticks=[], yticks=[])
            ax.set_title(f"{label}, t = {k * sc.dt:.1f} s", fontsize=9)
    fig.suptitle("No idea where the robot starts: 5,000 particles over the whole map (red circle: true position)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(DOCS / "global_localisation.png", dpi=110)


if __name__ == "__main__":
    res = json.loads((DOCS / "results.json").read_text())
    trajectory()
    consistency(res)
    global_localisation()
    print("figures written to docs/")
