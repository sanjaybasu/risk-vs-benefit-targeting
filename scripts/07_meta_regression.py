"""Frozen implementation of the Paper 1 cross-trial analyses (pre-registration v1, F30).

Two modes.
  simulate  Before unsealing. Power, type I error, and influence of the H1 test given the posted
            moderators and the design variances (F12, F13, F28). Reads results/scale_forecast.json
            and results/dataset_table.json only.
  analyze   After unsealing. Reads the custodians' scorer output(s) and runs H1 (strong and weak),
            H1b, the forecast contest for holdouts, leave-one-trial-out slopes and forecasts, the
            errors-in-variables sensitivity, the size placebo, and exploratory H2 and H3.

Meta-regression recipe (F28): random effects, tau^2 by REML (Fisher scoring), Knapp-Hartung
variance with t on k - p df, untruncated q as primary and truncated max(1, q) as sensitivity.
Primary weights (F16): sampling variances fixed before unsealing, v_j = (MDE_j / 2.8)^2 from the
posted power column; sensitivity: the scorer's bootstrap SD^2.
Units: outcomes improved per 1,000 actions at the 20% budget.

Usage:
  uv run python scripts/07_meta_regression.py simulate
  uv run python scripts/07_meta_regression.py analyze --scored s1.json [s2.json ...] --holdouts a,b
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
Q = "0.2"
SEED = 20261112
FORECAST_MARGIN_NATS = 1.0  # H1 forecasts must beat the intercept-only forecast by 1 nat (summed)


@dataclass
class Fit:
    beta: np.ndarray
    vcov: np.ndarray
    tau2: float
    df: int
    q: float


def reml_tau2(y, X, v, iters=200):
    tau2 = max(0.0, np.var(y) - np.mean(v))
    for _ in range(iters):
        w = 1 / (v + tau2)
        W = np.diag(w)
        XtWX_inv = np.linalg.inv(X.T @ W @ X)
        P = W - W @ X @ XtWX_inv @ X.T @ W
        score = -0.5 * np.trace(P) + 0.5 * y @ P @ P @ y
        info = 0.5 * np.trace(P @ P)
        new = max(0.0, tau2 + score / info)
        if abs(new - tau2) < 1e-10:
            break
        tau2 = new
    return tau2


def fit_re(y, X, v, truncate=False) -> Fit:
    y, X, v = np.asarray(y, float), np.asarray(X, float), np.asarray(v, float)
    k, p = X.shape
    tau2 = reml_tau2(y, X, v)
    w = 1 / (v + tau2)
    XtWX_inv = np.linalg.inv(X.T @ (X * w[:, None]))
    beta = XtWX_inv @ X.T @ (w * y)
    r = y - X @ beta
    q = float(np.sum(w * r**2) / (k - p))
    q_used = max(1.0, q) if truncate else q
    return Fit(beta=beta, vcov=q_used * XtWX_inv, tau2=tau2, df=k - p, q=q)


def slope_test(f: Fit) -> dict:
    se = float(np.sqrt(f.vcov[1, 1]))
    t = float(f.beta[1] / se)
    crit = stats.t.ppf(0.975, f.df)
    return {"slope": float(f.beta[1]), "se": se, "lo95": float(f.beta[1] - crit * se),
            "hi95": float(f.beta[1] + crit * se), "p_two_sided": float(2 * stats.t.sf(abs(t), f.df)),
            "intercept": float(f.beta[0]), "tau2": f.tau2, "q": f.q}


def joint_test(f: Fit, b0=(0.0, 1.0)) -> dict:
    d = f.beta - np.asarray(b0)
    F = float(d @ np.linalg.solve(f.vcov, d) / 2)
    return {"F": F, "p": float(stats.f.sf(F, 2, f.df))}


def predict(f: Fit, x_new: np.ndarray, v_new: float) -> tuple[float, float]:
    mean = float(x_new @ f.beta)
    var = float(x_new @ f.vcov @ x_new + f.tau2 + v_new)
    return mean, var


def lpd(y, mean, var, df):
    return float(stats.t.logpdf(y, df, loc=mean, scale=np.sqrt(var)))


# --------------------------------------------------------------------------- inputs
def posted() -> dict:
    fc = json.loads((RES / "scale_forecast.json").read_text())
    dt = json.loads((RES / "dataset_table.json").read_text())
    out = {}
    for t, r in fc.items():
        out[t] = {"x": r["gap"][Q], "x_se": r["gap_se"][Q], "beneficial": r["beneficial_sign"],
                  "v_design": (1000 * dt[t]["mde_per_action_at_q20"] / 2.8) ** 2,
                  "base": r["control_base_rate"], "pred_ate": r["pred_ate_per_1000"]}
    return out


def confirmatory(post: dict) -> list[str]:
    return [t for t, r in post.items() if r["beneficial"]]


# --------------------------------------------------------------------------- simulate
def simulate(n_sims=4000) -> dict:
    post = posted()
    trials = confirmatory(post)
    x = np.array([post[t]["x"] for t in trials])
    v = np.array([post[t]["v_design"] for t in trials])
    X = np.column_stack([np.ones_like(x), x])
    rng = np.random.default_rng(SEED)
    grid = {}
    for slope in (0.0, 0.5, 1.0):
        for tau in (0, 10, 20, 30):
            for trunc in (False, True):
                rej_weak = rej_joint = 0
                for _ in range(n_sims):
                    y = slope * x + rng.normal(0, tau, len(x)) + rng.normal(0, np.sqrt(v))
                    f = fit_re(y, X, v, truncate=trunc)
                    s = slope_test(f)
                    rej_weak += (s["lo95"] > 0)
                    rej_joint += joint_test(f)["p"] < 0.05
                grid[f"slope={slope},tau={tau},trunc={trunc}"] = {
                    "p_weak_form_positive": rej_weak / n_sims,
                    "p_reject_joint_0_1": rej_joint / n_sims}
    w = 1 / v
    Xw = X * w[:, None]
    H = X @ np.linalg.inv(X.T @ Xw) @ Xw.T
    xc = x - np.average(x, weights=w)
    info = w * xc**2 / np.sum(w * xc**2)
    influence = {t: {"hat": float(H[i, i]), "slope_information_share": float(info[i]),
                     "x": float(x[i]), "design_se": float(np.sqrt(v[i]))}
                 for i, t in enumerate(trials)}
    return {"confirmatory_trials": trials, "excluded_by_sign_rule": [t for t in post if t not in trials],
            "grid": grid, "influence": influence, "n_sims": n_sims}


# --------------------------------------------------------------------------- analyze
def observed(scored_files: list[Path]) -> dict:
    out = {}
    for f in scored_files:
        for t, r in json.loads(Path(f).read_text())["trials"].items():
            row = {(x["ranking"], x["budget"]): x for x in r["table"]}
            g = lambda k: row[(k, 0.2)]  # noqa: E731
            out[t] = {"scale_minus_risk": g("scale-minus-risk")["estimate"],
                      "scale_minus_risk_sd": g("scale-minus-risk")["boot_sd"],
                      "benefit_minus_scale": g("benefit-minus-scale")["estimate"],
                      "benefit_minus_scale_sd": g("benefit-minus-scale")["boot_sd"],
                      "benefit_minus_risk": g("benefit-minus-risk")["estimate"]}
    return out


def analyze(scored: list[Path], holdouts: list[str]) -> dict:
    post, obs = posted(), observed(scored)
    trials = [t for t in confirmatory(post) if t in obs]
    x = np.array([post[t]["x"] for t in trials])
    v = np.array([post[t]["v_design"] for t in trials])
    y = np.array([obs[t]["scale_minus_risk"] for t in trials])
    X = np.column_stack([np.ones_like(x), x])
    res = {"trials": trials}
    for trunc in (False, True):
        f = fit_re(y, X, v, truncate=trunc)
        res[f"H1_{'truncated' if trunc else 'primary'}"] = {**slope_test(f), "joint_0_1": joint_test(f)}
    v_obs = np.array([max(obs[t]["scale_minus_risk_sd"] ** 2, 1e-6) for t in trials])
    res["H1_observed_variance_weights"] = slope_test(fit_re(y, X, v_obs))
    # H1b: residual benefit - scale has no association with the moderator.
    yb = np.array([obs[t]["benefit_minus_scale"] for t in trials])
    res["H1b"] = slope_test(fit_re(yb, X, v))
    res["H1b"]["mean_residual"] = float(np.average(yb, weights=1 / v))
    # Leave-one-trial-out slopes and forecasts (F04, F14).
    loto = {}
    for i, t in enumerate(trials):
        m = np.arange(len(trials)) != i
        f1 = fit_re(y[m], X[m], v[m])
        f0 = fit_re(y[m], X[m][:, :1], v[m])
        mu1, var1 = predict(f1, X[i], v[i])
        mu0, var0 = predict(f0, X[i][:1], v[i])
        loto[t] = {"slope_without": slope_test(f1)["slope"], "slope_ci_without":
                   [slope_test(f1)["lo95"], slope_test(f1)["hi95"]],
                   "lpd_h1": lpd(y[i], mu1, var1, f1.df), "lpd_null": lpd(y[i], mu0, var0, f0.df),
                   "inside_pi95_h1": bool(abs(y[i] - mu1) <= stats.t.ppf(0.975, f1.df) * np.sqrt(var1))}
    res["leave_one_out"] = loto
    res["loo_lpd_gain_nats"] = float(sum(r["lpd_h1"] - r["lpd_null"] for r in loto.values()))
    # Holdout contest (F01): forecasts from the non-holdout fit.
    if holdouts:
        fit_m = np.array([t not in holdouts for t in trials])
        f1, f0 = fit_re(y[fit_m], X[fit_m], v[fit_m]), fit_re(y[fit_m], X[fit_m][:, :1], v[fit_m])
        gain = 0.0
        for t in holdouts:
            if t not in trials:
                continue
            i = trials.index(t)
            mu1, var1 = predict(f1, X[i], v[i])
            mu0, var0 = predict(f0, X[i][:1], v[i])
            gain += lpd(y[i], mu1, var1, f1.df) - lpd(y[i], mu0, var0, f0.df)
        res["holdout_lpd_gain_nats"] = gain
        res["holdout_forecast_succeeds"] = bool(gain > FORECAST_MARGIN_NATS)
    # Errors-in-variables sensitivity (F10): regression-calibration with posted moderator SEs.
    xse = np.array([post[t]["x_se"] for t in trials])
    w = 1 / v
    var_x = np.average((x - np.average(x, weights=w)) ** 2, weights=w)
    lam = max(1e-6, (var_x - np.average(xse**2, weights=w)) / var_x)
    res["EIV"] = {"reliability": float(lam), "slope_corrected": res["H1_primary"]["slope"] / lam}
    # Size placebo (F22): model-predicted ATE as the only moderator, then both.
    ate = np.array([post[t]["pred_ate"] for t in trials])
    res["size_placebo_ate_only"] = slope_test(fit_re(y, np.column_stack([np.ones_like(ate), ate]), v))
    f_both = fit_re(y, np.column_stack([np.ones_like(x), x, ate]), v)
    res["size_placebo_both"] = {"coef_moderator": float(f_both.beta[1]),
                                "se_moderator": float(np.sqrt(f_both.vcov[1, 1]))}
    # Exploratory H3: base-rate p(1 - p).
    pq = np.array([post[t]["base"] * (1 - post[t]["base"]) for t in trials])
    yd = np.array([obs[t]["benefit_minus_risk"] for t in trials])
    res["H3_exploratory"] = slope_test(fit_re(yd, np.column_stack([np.ones_like(pq), pq]), v))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["simulate", "analyze"])
    ap.add_argument("--scored", nargs="*", type=Path, default=[])
    ap.add_argument("--holdouts", default="")
    ap.add_argument("--sims", type=int, default=4000)
    a = ap.parse_args()
    if a.mode == "simulate":
        out = simulate(a.sims)
        (RES / "h1_power_simulation.json").write_text(json.dumps(out, indent=1))
        print("confirmatory:", out["confirmatory_trials"], "| excluded:", out["excluded_by_sign_rule"])
        for k, r in out["grid"].items():
            print(f"{k:32s} weak={r['p_weak_form_positive']:.3f} joint_reject={r['p_reject_joint_0_1']:.3f}")
        for t, r in sorted(out["influence"].items(), key=lambda kv: -kv[1]["slope_information_share"]):
            print(f"  {t:12s} x={r['x']:+7.2f} se={r['design_se']:7.2f} hat={r['hat']:.2f} "
                  f"info={r['slope_information_share']:.2f}")
    else:
        out = analyze(a.scored, [h for h in a.holdouts.split(",") if h])
        (RES / "h1_analysis.json").write_text(json.dumps(out, indent=1))
        print(json.dumps({k: out[k] for k in ("H1_primary", "H1b")}, indent=1))


if __name__ == "__main__":
    main()
