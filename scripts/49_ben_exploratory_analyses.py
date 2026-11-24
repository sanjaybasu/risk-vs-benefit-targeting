"""Post-unsealing exploratory analyses addressing co-author review (Ben Huynh, 2026-11-23).

All analyses use the already-hashed context-fitted models (predictions/<trial>_fitted.pkl and
predictions/<trial>.parquet) and the exact sealed test splits (../system-three-sealed/) with the
same AIPW estimator and cluster inference as score_sealed.py. Because these analyses were added
after test outcomes were scored, every output here is strictly labeled exploratory.

Computes and saves results/ben_exploratory_analyses.json:
  1. Trial-level forecast ESS, MAE/RMSE vs naive baselines (predict 0, LOO mean, expected oracle MAE),
     and unweighted mean observed vs forecast gain.
  2. Individual-level held-out test of the benefit-vs-risk mechanism across all 19 beneficial trials:
     - Pooled AIPW treatment effects across bins of risk relative to each trial's peak r* = sigma(c/2),
       compared against constant odds ratio (OR), constant risk ratio of bad outcome (RR_bad),
       constant risk difference (RD), and constant risk ratio of good outcome (RR_good).
     - Direct within-trial and pooled held-out test of the downturn above r* (peak zone vs far above peak),
       overall and by domain (voting, retail, medicine/IST, other).
     - Exact unbiased individual-level paired MSE comparisons among OR, RR_bad, RD, and RR_good using
       cluster-robust standard errors.
     - Exact power calculation for detecting IST's constant-OR downturn.
  3. Per-trial and pooled 80%-power minimum detectable differences (MDD_80 = 2.80 * SE) at q = 20%.
  4. Circular model-based (plug-in) vs held-out sealed AIPW gain overstatement for scale and DR rankings.
  5. Tighter operational budgets (q = 1% and 5%) on the hashed prediction files (exploratory).
"""

from __future__ import annotations

import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT.parent / "system-three-sealed"
PRED = ROOT / "predictions"
R = ROOT / "results"

spec_mr = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec_mr)
sys.modules["mr"] = mr
spec_mr.loader.exec_module(mr)

from system_three.estimators.dr import aipw_scores, sign  # noqa: E402
from system_three.metrics import per_1000  # noqa: E402
from system_three.splits import load as load_split  # noqa: E402

SEED = 20261112
N_BOOT = 2000


def cluster_mean_se(x: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """Mean of x and cluster-robust SE (Liang-Zeger / sandwich for a mean)."""
    x = np.asarray(x, float)
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), 0
    mu = float(x.mean())
    df = pd.DataFrame({"d": x - mu, "c": cl})
    sums = df.groupby("c", sort=False)["d"].sum().to_numpy(float)
    G = len(sums)
    if G <= 1:
        return mu, float("nan"), G
    var = (G / (G - 1.0)) * float(np.sum(sums ** 2)) / (n ** 2)
    return mu, float(np.sqrt(max(var, 0.0))), G


def cluster_diff_se(g: np.ndarray, m1: np.ndarray, m2: np.ndarray, cl: np.ndarray) -> tuple[float, float, int]:
    """Difference mean(g[m1]) - mean(g[m2]) and cluster-robust SE via influence function."""
    n1, n2 = int(m1.sum()), int(m2.sum())
    if n1 < 10 or n2 < 10:
        return float("nan"), float("nan"), 0
    mu1, mu2 = float(g[m1].mean()), float(g[m2].mean())
    n = len(g)
    p1, p2 = n1 / n, n2 / n
    psi = np.where(m1, (g - mu1) / p1, 0.0) - np.where(m2, (g - mu2) / p2, 0.0)
    df = pd.DataFrame({"psi": psi, "c": cl})
    sums = df.groupby("c", sort=False)["psi"].sum().to_numpy(float)
    G = len(sums)
    if G <= 1:
        return mu1 - mu2, float("nan"), G
    var = (G / (G - 1.0)) * float(np.sum(sums ** 2)) / (n ** 2)
    return mu1 - mu2, float(np.sqrt(max(var, 0.0))), G


def re_pool(est: list[float], se: list[float], floor: float = 0.0) -> dict:
    e = np.asarray(est, float)
    s = np.asarray(se, float)
    keep = np.isfinite(e) & np.isfinite(s) & (s > 0)
    e, s = e[keep], s[keep]
    k = len(e)
    if k == 0:
        return {"k": 0}
    if k == 1:
        return {"k": 1, "mean": float(e[0]), "se": float(s[0]),
                "lo95": float(e[0] - 1.96 * s[0]), "hi95": float(e[0] + 1.96 * s[0]), "tau": 0.0}
    v = np.maximum(s ** 2, floor)
    f = mr.fit_re(e, np.ones((k, 1)), v)
    se_m = float(np.sqrt(f.vcov[0, 0]))
    c = float(stats.t.ppf(0.975, f.df))
    p = float(2 * stats.t.sf(abs(f.beta[0] / se_m), f.df)) if se_m > 0 else 1.0
    half_pi = c * float(np.sqrt(f.tau2 + se_m ** 2))
    return {
        "k": k, "mean": float(f.beta[0]), "se": se_m,
        "lo95": float(f.beta[0] - c * se_m), "hi95": float(f.beta[0] + c * se_m),
        "p_two_sided": p, "tau": float(np.sqrt(f.tau2)),
        "pi95": [float(f.beta[0] - half_pi), float(f.beta[0] + half_pi)],
    }


def load_test_merged(t: str):
    meta = json.loads((SEALED / f"{t}_meta.json").read_text())
    sealed = pd.read_parquet(SEALED / f"{t}_test.parquet")
    pred = pd.read_parquet(PRED / f"{t}.parquet")
    two = sealed[sealed["action"].isin([meta["action"], meta["reference"]])]
    d = two.merge(pred, on="row_id", how="inner", validate="1:1").reset_index(drop=True)
    w = (d["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:
        e = np.full(len(d), meta["p_action"] / (meta["p_action"] + meta["p_reference"]))
    else:
        p = d["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float),
                    d["m1"].to_numpy(float), meta["better"])
    with open(PRED / f"{t}_fitted.pkl", "rb") as fh:
        f = pickle.load(fh)["fitted"]
    return d, meta, f, g, d["cluster_id"].to_numpy()


def context_rates(t: str, meta: dict) -> tuple[float, float]:
    """Inverse-design-weighted undesirable outcome rates under reference and action in context split."""
    ctx = load_split(t, "context")
    two = ctx[ctx["action"].isin([meta["action"], meta["reference"]])]
    w = (two["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:
        e = np.full(len(two), meta["p_action"] / (meta["p_action"] + meta["p_reference"]))
    else:
        p = two["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    y = two["outcome"].to_numpy(float)
    bad = y if meta["better"] == "lower" else (1.0 - y)
    w0 = (1 - w) / (1 - e)
    w1 = w / e
    r0 = float(np.sum(w0 * bad) / np.sum(w0))
    r1 = float(np.sum(w1 * bad) / np.sum(w1))
    return r0, r1


def _fast_topq_mean(g_sorted: np.ndarray, w_sorted: np.ndarray, q: float) -> float:
    """Weighted mean of g_sorted among the top-q share of total weight (exact for resampled bootstrap draw)."""
    n_tot = w_sorted.sum()
    k = max(1, int(round(q * n_tot)))
    csum = np.cumsum(w_sorted)
    pos = int(np.searchsorted(csum, k, side="left"))
    prev_w = csum[pos - 1] if pos > 0 else 0
    rem = k - prev_w
    num = float(np.dot(g_sorted[:pos], w_sorted[:pos]) + g_sorted[pos] * rem)
    return 1000.0 * num / k


def boot_q(g: np.ndarray, scores: dict[str, np.ndarray], cl: np.ndarray, budgets=(0.01, 0.05)):
    """Fast exact cluster bootstrap at budgets q in `budgets` (uses pre-sorted order per ranking)."""
    uniq, inv = np.unique(cl, return_inverse=True)
    G = len(uniq)
    n = len(g)
    rng = np.random.default_rng(SEED)
    # Pre-sort g and inv once per ranking (same seeded tie-break as per_1000)
    pre = {}
    for name, s in scores.items():
        jit = np.random.default_rng(SEED).random(n)
        order = np.lexsort((jit, -s))
        pre[name] = (g[order], inv[order])

    # Pre-compute point estimates
    pts = {str(q): {} for q in budgets}
    ones = np.ones(n, dtype=np.int64)
    for q in budgets:
        vals = {name: _fast_topq_mean(pre[name][0], ones, q) for name in scores}
        for a, b in (("scale", "risk"), ("benefit", "risk"), ("benefit", "scale"), ("response", "risk"), ("benefit", "response")):
            pts[str(q)][f"{a}_minus_{b}"] = vals[a] - vals[b]

    # Bootstrap cluster counts
    pairs = (("scale", "risk"), ("benefit", "risk"), ("benefit", "scale"), ("response", "risk"), ("benefit", "response"))
    draws = {str(q): {f"{a}_minus_{b}": np.empty(N_BOOT) for a, b in pairs} for q in budgets}
    for b_idx in range(N_BOOT):
        c_mult = np.bincount(rng.integers(0, G, G), minlength=G)
        # Compute csum once per ranking, reuse across all budgets q
        vals_by_q = {str(q): {} for q in budgets}
        for name, (gs, inv_s) in pre.items():
            ws = c_mult[inv_s]
            n_tot = int(ws.sum())
            csum = np.cumsum(ws)
            g_csum = np.cumsum(gs * ws)
            for q in budgets:
                k = max(1, int(round(q * n_tot)))
                pos = int(np.searchsorted(csum, k, side="left"))
                prev_w = int(csum[pos - 1]) if pos > 0 else 0
                prev_g = float(g_csum[pos - 1]) if pos > 0 else 0.0
                vals_by_q[str(q)][name] = 1000.0 * (prev_g + gs[pos] * (k - prev_w)) / k
        for q in budgets:
            q_s = str(q)
            v = vals_by_q[q_s]
            for a, b in pairs:
                draws[q_s][f"{a}_minus_{b}"][b_idx] = v[a] - v[b]

    out = {}
    for q in budgets:
        q_s = str(q)
        out[q_s] = {}
        for a, b in pairs:
            k_nm = f"{a}_minus_{b}"
            arr = draws[q_s][k_nm]
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[q_s][k_nm] = {
                "estimate": float(pts[q_s][k_nm]),
                "lo95": float(lo),
                "hi95": float(hi),
                "boot_sd": float(arr.std(ddof=1)),
            }
    return out


def main() -> None:
    fr = json.loads((R / "final_results.json").read_text())
    th = json.loads((R / "theory_checks.json").read_text())
    post = mr.posted()
    w1, w2 = fr["trials"]["wave1"], fr["trials"]["wave2"]
    all22 = w1 + w2
    elig = list(fr["Q2"]["points"].keys())
    zeros = set(fr["Q1"]["zero_gap_trials"])
    inf16 = [t for t in elig if t not in zeros]

    # ------------------------------------------------------------------
    # 1. Trial-level forecast ESS, MAE/RMSE vs baselines, unweighted means
    # ------------------------------------------------------------------
    pts = fr["Q2"]["points"]
    # Grab exact test-split bootstrap SDs for scale-minus-risk from final_results / scorer
    import glob
    boot_sd_scale = {}
    boot_sd_dr = {}
    boot_sd_dr_scale = {}
    for t in all22:
        ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"predictions__{t}.json")))
        tt = next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values()))
        for row in tt["table"]:
            if row["budget"] == 0.2:
                if row["ranking"] == "scale-minus-risk":
                    boot_sd_scale[t] = row["boot_sd"]
                elif row["ranking"] == "benefit-minus-risk":
                    boot_sd_dr[t] = row["boot_sd"]
                elif row["ranking"] == "benefit-minus-scale":
                    boot_sd_dr_scale[t] = row["boot_sd"]

    # Kish ESS of meta-regression weights
    def kish_ess(trials_list):
        x = np.array([post[t]["x"] for t in trials_list])
        y = np.array([pts[t]["y"] for t in trials_list])
        v = np.array([post[t]["v_primary"] for t in trials_list])
        f = mr.fit_re(y, np.column_stack([np.ones_like(x), x]), v)
        w = 1.0 / (v + f.tau2)
        return float((w.sum() ** 2) / np.sum(w ** 2))

    # Forecast accuracy over 16 informative trials (and 19 eligible trials) in per-1,000 units (10x = per 10,000)
    def forecast_metrics(trials_list):
        x = np.array([pts[t]["x"] for t in trials_list], float)
        y = np.array([pts[t]["y"] for t in trials_list], float)
        se = np.array([boot_sd_scale[t] for t in trials_list], float)
        k = len(trials_list)
        loo_mean = np.array([(y.sum() - y[i]) / (k - 1) for i in range(k)])
        # Precision-weighted LOO mean (for informative trials)
        iv = np.divide(1.0, se ** 2, out=np.zeros_like(se), where=se > 0)
        loo_iv = np.array([
            np.sum(np.delete(iv * y, i)) / np.sum(np.delete(iv, i)) if np.sum(np.delete(iv, i)) > 0 else 0.0
            for i in range(k)
        ])
        mae_fc = float(np.mean(np.abs(y - x)))
        mae_zero = float(np.mean(np.abs(y)))
        mae_loo = float(np.mean(np.abs(y - loo_mean)))
        mae_loo_iv = float(np.mean(np.abs(y - loo_iv)))
        rmse_fc = float(np.sqrt(np.mean((y - x) ** 2)))
        rmse_zero = float(np.sqrt(np.mean(y ** 2)))
        rmse_loo = float(np.sqrt(np.mean((y - loo_mean) ** 2)))
        # Expected MAE of a noiseless perfect forecast (E|N(0, se_i^2)| = sqrt(2/pi) * se_i)
        exp_oracle_mae = float(np.mean(math.sqrt(2.0 / math.pi) * se))
        exp_oracle_rmse = float(np.sqrt(np.mean(se ** 2)))
        # Paired bootstrap over trials for MAE(forecast) - MAE(baseline)
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, k, (10000, k))
        d_zero = np.mean(np.abs(y[idx] - x[idx]) - np.abs(y[idx]), axis=1)
        d_loo = np.mean(np.abs(y[idx] - x[idx]) - np.abs(y[idx] - loo_mean[idx]), axis=1)
        return {
            "k": k,
            "unweighted_mean_forecast": float(x.mean()),
            "unweighted_mean_observed": float(y.mean()),
            "median_test_se": float(np.median(se)),
            "mae_forecast": mae_fc,
            "mae_predict_zero": mae_zero,
            "mae_loo_mean": mae_loo,
            "mae_loo_iv_mean": mae_loo_iv,
            "expected_oracle_mae": exp_oracle_mae,
            "rmse_forecast": rmse_fc,
            "rmse_predict_zero": rmse_zero,
            "rmse_loo_mean": rmse_loo,
            "expected_oracle_rmse": exp_oracle_rmse,
            "mae_diff_vs_zero": {"est": mae_fc - mae_zero, "lo95": float(np.percentile(d_zero, 2.5)), "hi95": float(np.percentile(d_zero, 97.5))},
            "mae_diff_vs_loo_mean": {"est": mae_fc - mae_loo, "lo95": float(np.percentile(d_loo, 2.5)), "hi95": float(np.percentile(d_loo, 97.5))},
        }

    q2_power = {
        "ess_wave1_13": kish_ess([t for t in elig if not t.startswith("w2_")]),
        "ess_wave1_inf_11": kish_ess([t for t in inf16 if not t.startswith("w2_")]),
        "ess_all_19": kish_ess(elig),
        "ess_inf_16": kish_ess(inf16),
        "metrics_inf16": forecast_metrics(inf16),
        "metrics_all19": forecast_metrics(elig),
    }

    # ------------------------------------------------------------------
    # 2. Individual-level held-out mechanism test (19 beneficial trials)
    # ------------------------------------------------------------------
    DOM_MAP = {
        "gusto": "medicine", "ist": "medicine", "synbiotic": "medicine", "elaia2": "medicine",
        "voter": "voting", "sf_email": "voting", "sixcity": "voting", "newhaven": "voting",
        "w2_robocall2008": "voting", "w2_sanmateo_email": "voting", "w2_wisconsin_recall": "voting",
        "hillstrom": "retail", "lenta": "retail", "x5": "retail",
    }
    # Relative-to-peak bins: r - r*
    BIN_EDGES = [(-1.0, -0.35), (-0.35, -0.15), (-0.15, 0.05), (0.05, 0.25), (0.25, 0.38), (0.38, 1.0)]
    BIN_LABELS = ["< -0.35", "-0.35 to -0.15", "-0.15 to +0.05 (peak)", "+0.05 to +0.25", "+0.25 to +0.38", "> +0.38"]

    bin_records = {lbl: [] for lbl in BIN_LABELS}
    downturn_per_trial = {}
    mse_pairs_per_trial = {}
    ist_detail = {}

    # Also collect circular plug-in vs held-out AIPW optimism (Analysis 4) and tight budgets (Analysis 5)
    optimism = {}
    tight_budgets = {}

    for t in all22:
        print(f"processing {t}...", flush=True)
        d, meta, f, g, cl = load_test_merged(t)
        # Tight budgets (1% and 5%) on all 22 trials using already-hashed predictions
        resp_score = d["m1"].to_numpy(float) if meta["better"] == "higher" else (1.0 - d["m1"].to_numpy(float))
        scores_map = {
            "risk": d["risk"].to_numpy(float),
            "scale": d["scale"].to_numpy(float),
            "benefit": d["benefit"].to_numpy(float),
            "response": resp_score,
        }
        tight_budgets[t] = boot_q(g, scores_map, cl, budgets=(0.01, 0.05))

        # Circular plug-in gain vs held-out AIPW gain at q = 20%
        k20 = max(1, int(round(0.2 * len(d))))
        rng_t = np.random.default_rng(SEED)
        jit = rng_t.random(len(d))
        top_risk = np.lexsort((jit, -scores_map["risk"]))[:k20]
        top_dr = np.lexsort((jit, -scores_map["benefit"]))[:k20]
        top_scale = np.lexsort((jit, -scores_map["scale"]))[:k20]
        b_hat = scores_map["benefit"]
        s_hat = scores_map["scale"]
        plugin_dr_gain = float(1000.0 * (b_hat[top_dr].mean() - b_hat[top_risk].mean()))
        plugin_scale_gain = float(1000.0 * (s_hat[top_scale].mean() - s_hat[top_risk].mean()))
        obs_dr_gain = float(1000.0 * (g[top_dr].mean() - g[top_risk].mean()))
        obs_scale_gain = float(1000.0 * (g[top_scale].mean() - g[top_risk].mean()))
        optimism[t] = {
            "plugin_dr_over_risk": plugin_dr_gain,
            "heldout_dr_over_risk": obs_dr_gain,
            "overstatement_dr": plugin_dr_gain - obs_dr_gain,
            "se_dr": boot_sd_dr[t],
            "plugin_scale_over_risk": plugin_scale_gain,
            "heldout_scale_over_risk": obs_scale_gain,
            "overstatement_scale": plugin_scale_gain - obs_scale_gain,
            "se_scale": boot_sd_scale[t],
        }

        if t not in elig:
            continue

        # Exact recalibrated baseline risk of the undesirable outcome
        m0 = np.clip(d["m0"].to_numpy(float), 1e-4, 1.0 - 1e-4)
        lin = f.alpha + f.gamma * logit(m0)
        r_bad = expit(lin) if f.better == "lower" else expit(-lin)
        c = abs(float(f.beta))
        r_star = float(expit(c / 2.0))
        tau_or = expit(logit(np.clip(r_bad, 1e-6, 1 - 1e-6))) - expit(logit(np.clip(r_bad, 1e-6, 1 - 1e-6)) - c)
        # Verify exact match to hashed scale predictions
        assert float(np.max(np.abs(tau_or - scores_map["scale"]))) < 1e-4, f"scale mismatch in {t}"

        # Fit constant RR_bad, constant RD, and constant RR_good strictly on CONTEXT split
        r0_ctx, r1_ctx = context_rates(t, meta)
        rd_ctx = max(0.0, r0_ctx - r1_ctx)
        rr_bad_rel = max(0.0, (r0_ctx - r1_ctx) / max(r0_ctx, 1e-6))
        rr_good_rel = max(0.0, (r0_ctx - r1_ctx) / max(1.0 - r0_ctx, 1e-6))
        tau_rr_bad = np.clip(rr_bad_rel * r_bad, 0.0, r_bad)
        tau_rd = np.full_like(r_bad, rd_ctx)
        tau_rr_good = np.clip(rr_good_rel * (1.0 - r_bad), 0.0, r_bad)

        # Unbiased paired MSE differences: (g - tau_A)^2 - (g - tau_B)^2 (negative = A better than B)
        # Scale by 10^4 (so units are in (per 100)^2 or report in 10^-4 probability-squared units)
        pair_res = {}
        for nm, a_arr, b_arr in [
            ("or_minus_rr_bad", tau_or, tau_rr_bad),
            ("or_minus_rd", tau_or, tau_rd),
            ("or_minus_rr_good", tau_or, tau_rr_good),
            ("rr_bad_minus_rd", tau_rr_bad, tau_rd),
        ]:
            diff_sq = ((g - a_arr) ** 2 - (g - b_arr) ** 2) * 1e4  # units: 10^-4 prob^2
            est_m, se_m, G = cluster_mean_se(diff_sq, cl)
            pair_res[nm] = {"est": est_m, "se": se_m, "z": est_m / se_m if se_m > 0 else 0.0, "n_clusters": G}
        mse_pairs_per_trial[t] = {"domain": DOM_MAP.get(t, "other"), "r_star": r_star, "c": c, **pair_res}

        # Risk-binned held-out AIPW effects relative to r*
        delta = r_bad - r_star
        for (lo_e, hi_e), lbl in zip(BIN_EDGES, BIN_LABELS):
            mask = (delta >= lo_e) & (delta < hi_e)
            if mask.sum() >= 50:
                mu_g, se_g, G = cluster_mean_se(1000.0 * g[mask], cl[mask])
                bin_records[lbl].append({
                    "trial": t, "domain": DOM_MAP.get(t, "other"), "n": int(mask.sum()),
                    "obs": mu_g, "se": se_g,
                    "or": float(1000.0 * tau_or[mask].mean()),
                    "rr_bad": float(1000.0 * tau_rr_bad[mask].mean()),
                    "rd": float(1000.0 * tau_rd[mask].mean()),
                    "rr_good": float(1000.0 * tau_rr_good[mask].mean()),
                    "mean_delta": float(delta[mask].mean()),
                })

        # Direct within-trial downturn test: far above peak (r - r* > 0.25) minus peak zone (|r - r*| <= 0.15)
        m_peak = (delta >= -0.15) & (delta <= 0.15)
        m_high = delta > 0.25
        if m_peak.sum() >= 50 and m_high.sum() >= 50:
            diff_est, diff_se, G = cluster_diff_se(1000.0 * g, m_high, m_peak, cl)
            mu_pk, se_pk, _ = cluster_mean_se(1000.0 * g[m_peak], cl[m_peak])
            mu_hi, se_hi, _ = cluster_mean_se(1000.0 * g[m_high], cl[m_high])
            pred_diff_or = float(1000.0 * (tau_or[m_high].mean() - tau_or[m_peak].mean()))
            pred_diff_rr = float(1000.0 * (tau_rr_bad[m_high].mean() - tau_rr_bad[m_peak].mean()))
            downturn_per_trial[t] = {
                "domain": DOM_MAP.get(t, "other"),
                "n_peak": int(m_peak.sum()), "n_high": int(m_high.sum()),
                "obs_peak": mu_pk, "se_peak": se_pk,
                "obs_high": mu_hi, "se_high": se_hi,
                "obs_high_minus_peak": diff_est, "se_diff": diff_se,
                "pred_or_high_minus_peak": pred_diff_or,
                "pred_rr_bad_high_minus_peak": pred_diff_rr,
            }
            if t == "ist":
                # Sample size multiplier for 80% power to detect IST's predicted constant-OR downturn
                mdd_ist = 2.80 * diff_se
                mult_ist = (mdd_ist / abs(pred_diff_or)) ** 2 if abs(pred_diff_or) > 0 else float("nan")
                ist_detail = {**downturn_per_trial[t], "mdd80_downturn": mdd_ist, "n_multiplier_for_80pct_power": mult_ist}

    # Pool downturn across all eligible trials with mass in both zones, and by domain
    downturn_pooled = {}
    for grp_name, pred_fn in [
        ("all", lambda r: True),
        ("voting", lambda r: r["domain"] == "voting"),
        ("retail", lambda r: r["domain"] == "retail"),
        ("non_voting", lambda r: r["domain"] != "voting"),
        ("other_and_medicine", lambda r: r["domain"] in ("other", "medicine")),
    ]:
        sub = [v for v in downturn_per_trial.values() if pred_fn(v)]
        if sub:
            p_obs = re_pool([v["obs_high_minus_peak"] for v in sub], [v["se_diff"] for v in sub])
            p_obs["mean_pred_or_diff"] = float(np.mean([v["pred_or_high_minus_peak"] for v in sub]))
            p_obs["mean_pred_rr_bad_diff"] = float(np.mean([v["pred_rr_bad_high_minus_peak"] for v in sub]))
            p_obs["n_negative_point_est"] = int(sum(v["obs_high_minus_peak"] < 0 for v in sub))
            downturn_pooled[grp_name] = p_obs

    # Pool risk-binned curves across trials (random-effects per bin + weighted average of predicted curves using same RE weights)
    binned_pooled = {}
    for lbl, recs in bin_records.items():
        if not recs:
            continue
        p_bin = re_pool([r["obs"] for r in recs], [r["se"] for r in recs])
        v = np.array([r["se"] ** 2 for r in recs])
        w_re = 1.0 / (v + p_bin.get("tau", 0.0) ** 2)
        w_re = w_re / w_re.sum()
        p_bin["or_mean"] = float(np.sum(w_re * np.array([r["or"] for r in recs])))
        p_bin["rr_bad_mean"] = float(np.sum(w_re * np.array([r["rr_bad"] for r in recs])))
        p_bin["rd_mean"] = float(np.sum(w_re * np.array([r["rd"] for r in recs])))
        p_bin["rr_good_mean"] = float(np.sum(w_re * np.array([r["rr_good"] for r in recs])))
        p_bin["mean_delta"] = float(np.sum(w_re * np.array([r["mean_delta"] for r in recs])))
        p_bin["total_test_n"] = int(sum(r["n"] for r in recs))
        binned_pooled[lbl] = p_bin

    # Also compute same-trial binned curve over the 10 trials spanning both peak and far-above-peak zones
    span_trials = set(downturn_per_trial.keys())
    binned_span10 = {}
    for lbl, recs in bin_records.items():
        srecs = [r for r in recs if r["trial"] in span_trials]
        if not srecs:
            continue
        p_bin = re_pool([r["obs"] for r in srecs], [r["se"] for r in srecs])
        v = np.array([r["se"] ** 2 for r in srecs])
        w_re = 1.0 / (v + p_bin.get("tau", 0.0) ** 2)
        w_re = w_re / w_re.sum()
        p_bin["or_mean"] = float(np.sum(w_re * np.array([r["or"] for r in srecs])))
        p_bin["rr_bad_mean"] = float(np.sum(w_re * np.array([r["rr_bad"] for r in srecs])))
        p_bin["rd_mean"] = float(np.sum(w_re * np.array([r["rd"] for r in srecs])))
        p_bin["rr_good_mean"] = float(np.sum(w_re * np.array([r["rr_good"] for r in srecs])))
        p_bin["mean_delta"] = float(np.sum(w_re * np.array([r["mean_delta"] for r in srecs])))
        p_bin["total_test_n"] = int(sum(r["n"] for r in srecs))
        binned_span10[lbl] = p_bin

    # Also within-trial standardized (obs / trial_ate) across bins for the 10 spanning trials!
    # Wait: raw per-10,000 is cleaner and avoids ratio instability (panel dissent 6). Keep raw per-10,000!

    # Pool unbiased paired MSE differences across all 19 beneficial trials (and by domain)
    mse_pooled = {}
    for cmp_nm in ("or_minus_rr_bad", "or_minus_rd", "or_minus_rr_good", "rr_bad_minus_rd"):
        mse_pooled[cmp_nm] = {}
        for grp_name, pred_fn in [
            ("all19", lambda v: True),
            ("inf16", lambda v: v["_t"] in inf16),
            ("voting", lambda v: v["domain"] == "voting"),
            ("non_voting", lambda v: v["domain"] != "voting"),
            ("medicine", lambda v: v["domain"] == "medicine"),
        ]:
            sub = [v[cmp_nm] for t_, v in mse_pairs_per_trial.items() if pred_fn({**v, "_t": t_})]
            p_res = re_pool([s["est"] for s in sub], [s["se"] for s in sub])
            p_res["n_first_model_better"] = int(sum(s["est"] < 0 for s in sub))
            p_res["n_ci_favors_first"] = int(sum(s["est"] + 1.96 * s["se"] < 0 for s in sub))
            p_res["n_ci_favors_second"] = int(sum(s["est"] - 1.96 * s["se"] > 0 for s in sub))
            mse_pooled[cmp_nm][grp_name] = p_res

    # ------------------------------------------------------------------
    # 3. Power / Minimum Detectable Differences (MDD_80 = 2.80 * SE)
    # ------------------------------------------------------------------
    mdd_per_trial = {}
    for t in all22:
        mdd_per_trial[t] = {
            "domain": DOM_MAP.get(t, "other"),
            "n_test": int(json.loads((R / "dataset_table.json").read_text())[t]["test"]),
            "se_scale_vs_risk": float(boot_sd_scale[t]),
            "mdd80_scale_vs_risk": float(2.80 * boot_sd_scale[t]),
            "se_dr_vs_risk": float(boot_sd_dr[t]),
            "mdd80_dr_vs_risk": float(2.80 * boot_sd_dr[t]),
            "se_dr_vs_scale": float(boot_sd_dr_scale[t]),
            "mdd80_dr_vs_scale": float(2.80 * boot_sd_dr_scale[t]),
        }
    dr_ses = [mdd_per_trial[t]["se_dr_vs_risk"] for t in all22]
    med_ses = [mdd_per_trial[t]["se_dr_vs_risk"] for t in all22 if DOM_MAP.get(t) == "medicine"]
    vot_ses = [mdd_per_trial[t]["se_dr_vs_risk"] for t in all22 if DOM_MAP.get(t) == "voting"]
    # Pooled MDD_80 from exact random-effects SEs (half-width / t_crit)
    def pooled_mdd(block):
        # CI half-width = t_{0.975, k-1} * SE
        k = block["k"]
        tcrit = float(stats.t.ppf(0.975, k - 1))
        se = (block["hi95"] - block["lo95"]) / (2.0 * tcrit)
        return {"se": se, "mdd80": 2.80 * se}

    power_summary = {
        "per_trial_dr_se_median": float(np.median(dr_ses)),
        "per_trial_dr_se_min": float(np.min(dr_ses)),
        "per_trial_dr_se_max": float(np.max(dr_ses)),
        "per_trial_dr_mdd80_median": float(2.80 * np.median(dr_ses)),
        "per_trial_dr_mdd80_range": [float(2.80 * np.min(dr_ses)), float(2.80 * np.max(dr_ses))],
        "clinical_4_mdd80_range": [float(2.80 * np.min(med_ses)), float(2.80 * np.max(med_ses))],
        "voting_7_mdd80_range": [float(2.80 * np.min(vot_ses)), float(2.80 * np.max(vot_ses))],
        "pooled_dr_over_risk_22": pooled_mdd(fr["Q3"]["dr"]["all"]),
        "pooled_dr_over_scale_22": pooled_mdd(fr["Q3"]["dr_over_scale"]["all"]),
        "pooled_clinical_dr_over_risk_4": pooled_mdd(fr["Q3"]["by_domain"]["medicine"]["dr_over_risk"]),
        "per_trial": mdd_per_trial,
    }

    # ------------------------------------------------------------------
    # 4. Circular plug-in vs held-out AIPW optimism (pooled across trials)
    # ------------------------------------------------------------------
    opt_dr_pool = re_pool([optimism[t]["overstatement_dr"] for t in all22],
                          [optimism[t]["se_dr"] for t in all22], floor=mr.V_FLOOR)
    opt_dr_pool["mean_plugin"] = float(np.mean([optimism[t]["plugin_dr_over_risk"] for t in all22]))
    opt_dr_pool["mean_heldout"] = float(np.mean([optimism[t]["heldout_dr_over_risk"] for t in all22]))
    opt_dr_pool["unweighted_overstatement"] = opt_dr_pool["mean_plugin"] - opt_dr_pool["mean_heldout"]
    opt_dr_pool["n_overstated"] = int(sum(optimism[t]["overstatement_dr"] > 0 for t in all22))

    scale_17 = [t for t in all22 if boot_sd_scale[t] > 1e-9]
    opt_sc_pool = re_pool([optimism[t]["overstatement_scale"] for t in scale_17],
                          [optimism[t]["se_scale"] for t in scale_17], floor=mr.V_FLOOR)
    opt_sc_pool["mean_plugin"] = float(np.mean([optimism[t]["plugin_scale_over_risk"] for t in scale_17]))
    opt_sc_pool["mean_heldout"] = float(np.mean([optimism[t]["heldout_scale_over_risk"] for t in scale_17]))
    opt_sc_pool["unweighted_overstatement"] = opt_sc_pool["mean_plugin"] - opt_sc_pool["mean_heldout"]
    opt_sc_pool["n_overstated"] = int(sum(optimism[t]["overstatement_scale"] > 0 for t in scale_17))

    # ------------------------------------------------------------------
    # 5. Tighter budgets (1% and 5%) pooled across trials
    # ------------------------------------------------------------------
    tight_pooled = {}
    for q_str in ("0.01", "0.05"):
        tight_pooled[q_str] = {}
        for cmp_nm in ("scale_minus_risk", "benefit_minus_risk", "benefit_minus_scale", "response_minus_risk", "benefit_minus_response"):
            es = [tight_budgets[t][q_str][cmp_nm]["estimate"] for t in all22]
            sds = [tight_budgets[t][q_str][cmp_nm]["boot_sd"] for t in all22]
            keep = [i for i in range(len(all22)) if not (sds[i] < 1e-9 and abs(es[i]) < 1e-9)]
            tight_pooled[q_str][cmp_nm] = re_pool([es[i] for i in keep], [sds[i] for i in keep], floor=mr.V_FLOOR)

    # Also check correlation between context N and DR-over-scale gain & SE
    dt = json.loads((R / "dataset_table.json").read_text())
    ctx_n = np.array([dt[t]["context"] for t in all22], float)
    dr_ov_sc = np.array([fr["Q3"]["dr_over_scale"]["per_trial"][t]["estimate"] if "per_trial" in fr["Q3"]["dr_over_scale"] else 0.0 for t in all22])
    # Pull exact DR-over-scale per trial from scorer files
    dr_sc_est = {}
    for t in all22:
        ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"predictions__{t}.json")))
        tt = next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values()))
        dr_sc_est[t] = next(x["estimate"] for x in tt["table"] if x["ranking"] == "benefit-minus-scale" and x["budget"] == 0.2)
    dr_sc_arr = np.array([dr_sc_est[t] for t in all22], float)
    large_n = [t for t in all22 if dt[t]["rows"] >= 200000]
    small_n = [t for t in all22 if dt[t]["rows"] < 200000]
    size_split = {
        "spearman_ctx_n_vs_dr_over_scale": float(stats.spearmanr(ctx_n, dr_sc_arr).statistic),
        "spearman_p": float(stats.spearmanr(ctx_n, dr_sc_arr).pvalue),
        "ge_200k_trials": large_n,
        "ge_200k_dr_over_scale": re_pool([dr_sc_est[t] for t in large_n], [boot_sd_dr_scale[t] for t in large_n], floor=mr.V_FLOOR),
        "lt_200k_dr_over_scale": re_pool([dr_sc_est[t] for t in small_n], [boot_sd_dr_scale[t] for t in small_n], floor=mr.V_FLOOR),
    }

    # Also identify which trials differ between scale and risk at 1% and 5% budgets
    scale_diff_by_budget = {}
    for q_str in ("0.01", "0.05"):
        scale_diff_by_budget[q_str] = [
            t for t in all22
            if not (tight_budgets[t][q_str]["scale_minus_risk"]["boot_sd"] < 1e-9
                    and abs(tight_budgets[t][q_str]["scale_minus_risk"]["estimate"]) < 1e-9)
        ]

    out = {
        "note": "Post-unsealing exploratory analyses addressing co-author review; all values stored in per-1,000 units (multiply by 10 for per 10,000 reached) except MSE pairs (10^-4 prob^2) and dimensionless metrics.",
        "q2_power_and_baselines": q2_power,
        "individual_mechanism": {
            "binned_relative_to_peak": binned_pooled,
            "binned_spanning_10_trials": binned_span10,
            "spanning_10_trials": sorted(span_trials),
            "downturn_high_minus_peak_pooled": downturn_pooled,
            "downturn_per_trial": downturn_per_trial,
            "ist_downturn_power": ist_detail,
            "paired_mse_pooled": mse_pooled,
            "paired_mse_per_trial": mse_pairs_per_trial,
        },
        "power_mdd80": power_summary,
        "circular_evaluation_optimism": {
            "dr_over_risk_pooled": opt_dr_pool,
            "scale_over_risk_pooled": opt_sc_pool,
            "per_trial": optimism,
        },
        "tight_budgets_pooled": tight_pooled,
        "tight_budgets_scale_diff_trials": scale_diff_by_budget,
        "tight_budgets_per_trial": tight_budgets,
        "sample_size_vs_dr_over_scale": size_split,
    }
    (R / "ben_exploratory_analyses.json").write_text(json.dumps(out, indent=1))
    print("WROTE", R / "ben_exploratory_analyses.json")

    # Print crisp summary (scaled x10 to per 10,000 reached where applicable!)
    print("\n=== 1. Q2 ESS & Forecast Baselines (per 10,000 reached) ===")
    print(f"  ESS: W1(13)={q2_power['ess_wave1_13']:.2f}, W1_inf(11)={q2_power['ess_wave1_inf_11']:.2f}, All(19)={q2_power['ess_all_19']:.2f}, Inf(16)={q2_power['ess_inf_16']:.2f}")
    m16 = q2_power["metrics_inf16"]
    print(f"  Inf 16: unweighted mean obs={10*m16['unweighted_mean_observed']:.0f} vs fc={10*m16['unweighted_mean_forecast']:.0f}; median test SE={10*m16['median_test_se']:.0f}")
    print(f"  MAE (Inf 16): Forecast={10*m16['mae_forecast']:.0f}, LOO_mean={10*m16['mae_loo_mean']:.0f}, Predict_0={10*m16['mae_predict_zero']:.0f}, Expected_oracle={10*m16['expected_oracle_mae']:.0f}")
    print(f"  MAE diff vs 0: {10*m16['mae_diff_vs_zero']['est']:.0f} [{10*m16['mae_diff_vs_zero']['lo95']:.0f}, {10*m16['mae_diff_vs_zero']['hi95']:.0f}]")
    print(f"  MAE diff vs LOO mean: {10*m16['mae_diff_vs_loo_mean']['est']:.0f} [{10*m16['mae_diff_vs_loo_mean']['lo95']:.0f}, {10*m16['mae_diff_vs_loo_mean']['hi95']:.0f}]")

    print("\n=== 2A. Individual-level Held-out Curve (r - r*, per 10,000) ===")
    for lbl, b in binned_pooled.items():
        print(f"  {lbl:22s} (k={b['k']:2d}, n={b['total_test_n']:6d}): obs={10*b['obs' if 'obs' in b else 'mean']:6.0f} [{10*b['lo95']:6.0f}, {10*b['hi95']:6.0f}] | OR={10*b['or_mean']:5.0f} | RR_bad={10*b['rr_bad_mean']:5.0f} | RD={10*b['rd_mean']:5.0f}")

    print("\n=== 2B. Held-out Downturn (far above peak r-r*>0.25 minus peak |r-r*|<=0.15, per 10,000) ===")
    for grp, v in downturn_pooled.items():
        print(f"  {grp:20s} (k={v['k']:2d}): obs={10*v['mean']:6.0f} [{10*v['lo95']:6.0f}, {10*v['hi95']:6.0f}] P={v.get('p_two_sided', float('nan')):.4f} (neg={v['n_negative_point_est']}/{v['k']}) | pred_OR={10*v['mean_pred_or_diff']:5.0f} | pred_RR_bad={10*v['mean_pred_rr_bad_diff']:+5.0f}")
    if ist_detail:
        print(f"  IST specifically: obs_diff={10*ist_detail['obs_high_minus_peak']:.0f} (SE={10*ist_detail['se_diff']:.0f}), pred_OR={10*ist_detail['pred_or_high_minus_peak']:.0f}, MDD80={10*ist_detail['mdd80_downturn']:.0f}, N_mult={ist_detail['n_multiplier_for_80pct_power']:.1f}x")

    print("\n=== 2C. Unbiased Individual-Level Paired MSE (10^-4 prob^2; negative favors 1st model) ===")
    for cmp_nm, grps in mse_pooled.items():
        g = grps["all19"]
        gv = grps["voting"]
        gnv = grps["non_voting"]
        print(f"  {cmp_nm:18s}: all19={g['mean']:+7.3f} [{g['lo95']:+7.3f}, {g['hi95']:+7.3f}] P={g['p_two_sided']:.4f} (wins {g['n_first_model_better']}/19, sig {g['n_ci_favors_first']} vs {g['n_ci_favors_second']})")
        print(f"                     voting={gv['mean']:+7.3f} [{gv['lo95']:+7.3f}, {gv['hi95']:+7.3f}] | non_voting={gnv['mean']:+7.3f} [{gnv['lo95']:+7.3f}, {gnv['hi95']:+7.3f}]")

    print("\n=== 3. Power / MDD_80 (per 10,000 reached) ===")
    print(f"  Per-trial DR-vs-risk SE median={10*power_summary['per_trial_dr_se_median']:.0f} (range {10*power_summary['per_trial_dr_se_min']:.0f}-{10*power_summary['per_trial_dr_se_max']:.0f}); MDD80 median={10*power_summary['per_trial_dr_mdd80_median']:.0f}")
    print(f"  Clinical 4 trials MDD80 range: {10*power_summary['clinical_4_mdd80_range'][0]:.0f} to {10*power_summary['clinical_4_mdd80_range'][1]:.0f}; pooled clinical MDD80={10*power_summary['pooled_clinical_dr_over_risk_4']['mdd80']:.0f}")
    print(f"  Pooled 22 trials MDD80: DR-vs-risk={10*power_summary['pooled_dr_over_risk_22']['mdd80']:.0f}, DR-vs-scale={10*power_summary['pooled_dr_over_scale_22']['mdd80']:.0f}")

    print("\n=== 4. Circular Plug-in vs Held-out AIPW Optimism (per 10,000 reached) ===")
    print(f"  DR-vs-risk (22 trials): mean plugin={10*opt_dr_pool['mean_plugin']:.0f}, mean heldout={10*opt_dr_pool['mean_heldout']:.0f}, RE overstatement={10*opt_dr_pool['mean']:.0f} [{10*opt_dr_pool['lo95']:.0f}, {10*opt_dr_pool['hi95']:.0f}] (overstated in {opt_dr_pool['n_overstated']}/22)")
    print(f"  Scale-vs-risk (17 trials): mean plugin={10*opt_sc_pool['mean_plugin']:.0f}, mean heldout={10*opt_sc_pool['mean_heldout']:.0f}, RE overstatement={10*opt_sc_pool['mean']:.0f} [{10*opt_sc_pool['lo95']:.0f}, {10*opt_sc_pool['hi95']:.0f}] (overstated in {opt_sc_pool['n_overstated']}/17)")

    print("\n=== 5. Tighter Budgets 1% and 5% (per 10,000 reached) ===")
    for q_str in ("0.01", "0.05"):
        for cmp_nm in ("scale_minus_risk", "response_minus_risk", "benefit_minus_risk", "benefit_minus_scale", "benefit_minus_response"):
            v = tight_pooled[q_str][cmp_nm]
            print(f"  q={q_str} {cmp_nm:22s} (k={v['k']:2d}): {10*v['mean']:6.0f} [{10*v['lo95']:6.0f}, {10*v['hi95']:6.0f}]")

    print("\n=== 6. Trial Size vs DR-over-scale ===")
    print(f"  Spearman(ctx_N, DR-scale)={size_split['spearman_ctx_n_vs_dr_over_scale']:.2f} (P={size_split['spearman_p']:.3f})")
    print(f"  >=200k ({len(large_n)} trials: {large_n}): DR-scale={10*size_split['ge_200k_dr_over_scale']['mean']:.0f} [{10*size_split['ge_200k_dr_over_scale']['lo95']:.0f}, {10*size_split['ge_200k_dr_over_scale']['hi95']:.0f}]")
    print(f"  <200k ({len(small_n)} trials): DR-scale={10*size_split['lt_200k_dr_over_scale']['mean']:.0f} [{10*size_split['lt_200k_dr_over_scale']['lo95']:.0f}, {10*size_split['lt_200k_dr_over_scale']['hi95']:.0f}]")


if __name__ == "__main__":
    main()
