"""Primary moderator for H1: the scale-predicted gap, from context data only (v1).

For each candidate trial: cross-fit p0 on control-arm context rows (5 folds, fixed
hyperparameters), fit logit P = alpha + gamma * logit p0 + beta * a by inverse-design-weighted
maximum likelihood, and compute each context unit's implied benefit b(x). The predicted gap at
budget q is 1000 * [mean b over the top q by b - mean b over the top q by risk].

Changes from v0 (red-team): design weights in the scale fit (F67); early stopping off (F64);
frozen prep in src/system_three/analysis.py (F76); a cluster bootstrap over context that refits
everything in each draw, reporting mean and SE (F09); the beneficial-sign rule for H1 entry (F06);
out-of-fold AUC of p0 (F26).

Runs on Modal CPU (each bootstrap draw is one container call). Reads only context.parquet.
  modal run scripts/06_scale_forecast.py            # 200 draws per trial
Writes results/scale_forecast.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["gusto", "ist", "synbiotic", "elaia1", "elaia2", "voter", "sf_email", "karlan_list",
          "stepup", "hillstrom", "oregon"]
BUDGETS = (0.10, 0.20, 0.30)
N_DRAWS = 200

app = modal.App("system-three-scale-forecast")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.2.6", "pandas==2.2.3", "pyarrow==18.1.0", "scipy==1.15.3",
                 "scikit-learn==1.6.1", "pydantic==2.12.5")
    .add_local_dir(ROOT / "src", "/root/src", copy=True)
)
for _t in TRIALS:  # context split only; dev and test never leave the machine
    image = image.add_local_file(ROOT / "data" / "processed" / _t / "context.parquet",
                                 f"/root/processed/{_t}/context.parquet", copy=True)
    if (ROOT / "data" / "processed" / _t / "attrs.json").exists():
        image = image.add_local_file(ROOT / "data" / "processed" / _t / "attrs.json",
                                     f"/root/processed/{_t}/attrs.json", copy=True)


def _compute(trial: str, draw: int) -> dict:
    import sys

    import numpy as np
    import pandas as pd
    from scipy.special import expit, logit

    sys.path.insert(0, "/root/src")
    from system_three import rankers
    from system_three.analysis import ARM
    from system_three.estimators.dr import binarize, sign

    proc = Path("/root/processed") / trial
    ctx = pd.read_parquet(proc / "context.parquet")
    if (proc / "attrs.json").exists():
        ctx.attrs.update(json.loads((proc / "attrs.json").read_text()))
    ref = ctx["reference"].iat[0]
    act = ARM.get(trial) or next(a for a in sorted(ctx["action"].unique()) if a != ref)
    b = binarize(ctx, act).reset_index(drop=True)
    groups = b["cluster_id"].to_numpy()
    if draw > 0:  # cluster half-sampling without replacement: no duplicated rows (see main)
        rng = np.random.default_rng(rankers.SEED + draw)
        uniq = np.unique(groups)
        keep = set(rng.choice(uniq, size=len(uniq) // 2, replace=False).tolist())
        mask = np.array([g in keep for g in groups])
        b = b[mask].reset_index(drop=True)
        groups = groups[mask]
    X, _ = rankers.design_matrix(b)
    w, y, e = b["W"].to_numpy(), b["outcome"].to_numpy(), b["e"].to_numpy()
    better = b["better"].iat[0]
    m0, _, _, _ = rankers.cross_fit_p(X, w, y, groups)
    wt = np.where(w == 1, 1 / e, 1 / (1 - e))
    alpha, gamma, beta, se = rankers.scale_fit(logit(m0), w, y, wt)
    lin = alpha + gamma * logit(m0)
    s = sign(better)
    ben = s * (expit(lin + beta) - expit(lin))
    risk = m0 if better == "lower" else 1 - m0
    jitter = np.random.default_rng(rankers.SEED).random(len(ben))

    def top(score, q):
        k = max(1, int(round(q * len(ben))))
        return float(ben[np.lexsort((jitter, -score))[:k]].mean())

    out = {"trial": trial, "draw": draw, "beta": beta, "beta_se": se, "alpha": alpha,
           "gamma": gamma, "pred_ate_per_1000": float(1000 * ben.mean()),
           "gap": {str(q): 1000 * (top(ben, q) - top(risk, q)) for q in BUDGETS}}
    if draw == 0:
        from sklearn.metrics import roc_auc_score

        ctrl = w == 0
        out.update({
            "n_context_two_arm": int(len(y)), "contrast_arm": act, "better": better,
            "control_base_rate": float(np.average(y[ctrl])),
            "auc_p0_oof": float(roc_auc_score(y[ctrl], m0[ctrl])),
            "max_ipw_weight": float(wt.max()),
            "raw_ate_context_per_1000": float(1000 * s * (y[w == 1].mean() - y[w == 0].mean())),
        })
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=3600)
def compute(trial: str, draw: int) -> dict:
    return _compute(trial, draw)


@app.local_entrypoint()
def main(draws: int = N_DRAWS):
    import numpy as np

    jobs = [(t, d) for t in TRIALS for d in range(draws + 1)]
    res = list(compute.starmap(jobs, order_outputs=False, return_exceptions=True))
    errors = [r for r in res if isinstance(r, Exception)]
    res = [r for r in res if not isinstance(r, Exception)]
    out = {}
    for t in TRIALS:
        rs = [r for r in res if r["trial"] == t]
        plug = next(r for r in rs if r["draw"] == 0)
        boots = [r for r in rs if r["draw"] > 0]
        rec = {k: v for k, v in plug.items() if k not in ("trial", "draw")}
        rec["n_draws"] = len(boots)
        # Half-sampling: SE(full n) = SD(half-sample estimates) / sqrt(2) for root-n estimators.
        # Resampling with replacement duplicated clusters and biased cross-fitted p0 (pilot,
        # 2026-11-12), so the bootstrap is replaced by half-sampling.
        rec["se_method"] = "cluster half-sampling, SD/sqrt(2)"
        rec["gap_halfsample_mean"] = {q: float(np.mean([r["gap"][q] for r in boots])) for q in plug["gap"]}
        rec["gap_se"] = {q: float(np.std([r["gap"][q] for r in boots], ddof=1) / np.sqrt(2))
                         for q in plug["gap"]}
        bb = np.array([r["beta"] for r in boots])
        se_b = float(np.std(bb, ddof=1) / np.sqrt(2))
        rec["beta_halfsample_se"] = se_b
        rec["beta_ci95"] = [plug["beta"] - 1.96 * se_b, plug["beta"] + 1.96 * se_b]
        s = 1.0 if plug["better"] == "higher" else -1.0
        # F06(d): H1 entry requires the context beta to have the beneficial sign.
        rec["beneficial_sign"] = bool(s * plug["beta"] > 0)
        out[t] = rec
    (ROOT / "results" / "scale_forecast.json").write_text(json.dumps(out, indent=1))
    print(f"{len(errors)} failed calls")
    for t, r in out.items():
        print(f"{t:12s} base={r['control_base_rate']:.3f} beta={r['beta']:+.3f} "
              f"[{r['beta_ci95'][0]:+.3f},{r['beta_ci95'][1]:+.3f}] sandwich_se={r['beta_se']:.3f} "
              f"gap20={r['gap']['0.2']:+.2f} se={r['gap_se']['0.2']:.2f} "
              f"(half mean {r['gap_halfsample_mean']['0.2']:+.2f}) auc={r['auc_p0_oof']:.3f} "
              f"benef={r['beneficial_sign']}")
