"""Monte Carlo comparison of the three filters in three situations; writes docs/results.json.

usage: python scripts/run_experiments.py [--runs 20]     (about 3 minutes)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loc import EKF, UKF, ParticleFilter, Scenario, run_filter, simulate  # noqa: E402
from loc.metrics import errors, nees, nees_bounds, rmse  # noqa: E402

P0_NORMAL = np.diag([0.3**2, 0.3**2, np.radians(5) ** 2])
P0_HARD = np.diag([1.0, 1.0, np.radians(30) ** 2])
NORMAL = Scenario()
DEAD_ZONE = Scenario(min_range=0.3)
HARD = Scenario(sigma_b=0.1, sigma_w=0.15, sigma_v=0.2, max_range=4.0, measure_every=5)

FILTERS = {
    "EKF": lambda x0, P0, sc, s: EKF(x0, P0, sc.M, sc.Q),
    "UKF": lambda x0, P0, sc, s: UKF(x0, P0, sc.M, sc.Q),
    "PF 1,000": lambda x0, P0, sc, s: ParticleFilter.gaussian(x0, P0, 1000, sc.M, sc.Q, np.random.default_rng(s + 100)),
    "PF 1,000 + roughening": lambda x0, P0, sc, s: ParticleFilter.gaussian(
        x0, P0, 1000, sc.M, sc.Q, np.random.default_rng(s + 100), roughen=0.2),
    "PF 5,000": lambda x0, P0, sc, s: ParticleFilter.gaussian(x0, P0, 5000, sc.M, sc.Q, np.random.default_rng(s + 100)),
}


def monte_carlo(sc, P0, filters, runs):
    out = {name: {"pos_rmse": [], "head_rmse": [], "nees_mean": [], "nees_t": [], "ms_per_step": []} for name in filters}
    for s in range(runs):
        rng = np.random.default_rng(s)
        run = simulate(sc, rng)
        x0 = run.truth[0] + rng.multivariate_normal(np.zeros(3), P0)
        for name, make in filters.items():
            t0 = time.perf_counter()
            m, c = run_filter(make(x0, P0, sc, s), run, sc)
            ms = 1000 * (time.perf_counter() - t0) / sc.steps
            pos, head = rmse(m, run.truth[1:])
            n = nees(m, c, run.truth[1:])
            r = out[name]
            r["pos_rmse"].append(pos), r["head_rmse"].append(head), r["nees_mean"].append(float(n.mean()))
            r["nees_t"].append(n.tolist()), r["ms_per_step"].append(ms)
    return out


def summary(res, runs):
    lo, hi = nees_bounds(runs)
    rows = {}
    for name, r in res.items():
        nt = np.array(r["nees_t"])
        rows[name] = {"pos_rmse_cm": 100 * float(np.mean(r["pos_rmse"])), "head_rmse_deg": float(np.mean(r["head_rmse"])),
                      "nees_mean": float(np.mean(r["nees_mean"])), "nees_median_run": float(np.median(r["nees_mean"])),
                      "nees_worst_run": float(np.max(r["nees_mean"])),
                      "frac_steps_in_band": float(np.mean((nt.mean(axis=0) >= lo) & (nt.mean(axis=0) <= hi))),
                      "ms_per_step": float(np.mean(r["ms_per_step"]))}
    return rows


def global_localisation(runs):
    """Start with no idea of the pose. 'settle' = time after which the error stays below 0.3 m."""
    sc = NORMAL
    out = {"pf_settle_s": [], "amcl_settle_s": [], "pf_final_err": [], "amcl_final_err": [],
           "ekf_final_err": [], "ukf_final_err": []}
    for s in range(runs):
        run = simulate(sc, np.random.default_rng(s))
        for key, inject in (("pf", False), ("amcl", True)):
            pf = ParticleFilter.uniform((-8, -8), (8, 8), 5000, sc.M, sc.Q, np.random.default_rng(s + 7), inject=inject)
            m, _ = run_filter(pf, run, sc)
            e = errors(m, run.truth[1:])
            d = np.hypot(e[:, 0], e[:, 1])
            above = np.flatnonzero(d >= 0.3)
            out[f"{key}_settle_s"].append(float((above[-1] + 1) * sc.dt) if len(above) else 0.0)
            out[f"{key}_final_err"].append(float(d[-100:].mean()))
        for name, F in (("ekf", EKF), ("ukf", UKF)):
            f = F(np.array([4.0, -4.0, np.pi]), np.diag([25.0, 25.0, np.pi**2]), sc.M, sc.Q)  # 5.7 m off, facing backwards
            m, _ = run_filter(f, run, sc)
            e = errors(m, run.truth[1:])
            out[f"{name}_final_err"].append(float(np.hypot(e[-100:, 0], e[-100:, 1]).mean()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()
    n = args.runs
    results = {"runs": n, "nees_band": nees_bounds(n)}
    t0 = time.perf_counter()
    normal = monte_carlo(NORMAL, P0_NORMAL, FILTERS, n)
    results["normal"] = summary(normal, n)
    results["normal_ekf_nees_runs"] = normal["EKF"]["nees_mean"]
    worst = int(np.argmax(normal["EKF"]["nees_mean"]))
    results["normal_ekf_worst"] = {"seed": worst, "nees_t": normal["EKF"]["nees_t"][worst]}
    results["normal_nees_t_mean"] = {k: np.mean(v["nees_t"], axis=0).tolist() for k, v in normal.items()}
    results["normal_nees_runs"] = {k: v["nees_mean"] for k, v in normal.items()}
    results["dead_zone"] = summary(monte_carlo(DEAD_ZONE, P0_NORMAL, FILTERS, n), n)
    hard = monte_carlo(HARD, P0_HARD, FILTERS, n)
    results["hard"] = summary(hard, n)
    results["hard_nees_runs"] = {k: v["nees_mean"] for k, v in hard.items()}
    results["global"] = global_localisation(n)
    results["seconds"] = time.perf_counter() - t0
    Path("docs").mkdir(exist_ok=True)
    Path("docs/results.json").write_text(json.dumps(results))
    for key in ("normal", "dead_zone", "hard"):
        print(key)
        for name, r in results[key].items():
            print(f"  {name:24s} pos {r['pos_rmse_cm']:5.1f} cm  heading {r['head_rmse_deg']:4.2f} deg  "
                  f"NEES mean {r['nees_mean']:8.2f}  median run {r['nees_median_run']:5.2f}  worst run {r['nees_worst_run']:9.1f}"
                  f"  steps in band {r['frac_steps_in_band']:.2f}  {r['ms_per_step']:.2f} ms/step")
    g = results["global"]
    for key in ("pf", "amcl"):
        print(f"global {key}: settles in median {np.median(g[key + '_settle_s']):.1f} s, worst {max(g[key + '_settle_s']):.1f} s;"
              f" final errors > 0.3 m: {sum(e > 0.3 for e in g[key + '_final_err'])}")
    print("global EKF/UKF from a wrong start, final error > 0.3 m:", sum(e > 0.3 for e in g["ekf_final_err"]),
          sum(e > 0.3 for e in g["ukf_final_err"]))
    print(f"{results['seconds']:.0f} s")


if __name__ == "__main__":
    main()
