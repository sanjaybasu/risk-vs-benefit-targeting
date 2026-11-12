"""Primary moderator for H1: the scale-predicted gap, from context data only.

For each trial, fit the risk model p0(x) (cross-fitted, control-arm context rows) and the
constant-relative-effect shift beta. Under that model each context unit's absolute benefit is
b(x) = s * (expit(logit p0 + beta) - p0). The predicted gap at budget q is
1000 * [mean b over the top q by scale score - mean b over the top q by risk score].
Also records the control-arm base rate and p(1 - p). Reads only context.parquet.

Usage: uv run python scripts/06_scale_forecast.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit
from sklearn.model_selection import KFold

from system_three import rankers
from system_three.estimators.dr import sign

sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module  # noqa: E402

prep = import_module("05_dev_baselines").prep
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "scale_forecast.json"
TRIALS = ["gusto", "ist", "synbiotic", "elaia1", "elaia2", "voter", "sf_email", "karlan_list",
          "stepup", "oregon", "hillstrom"]
BUDGETS = (0.10, 0.20, 0.30)


def top_mean(b, s, q):
    k = max(1, int(round(q * len(b))))
    jitter = np.random.default_rng(rankers.SEED).random(len(s))
    return float(b[np.lexsort((jitter, -s))[:k]].mean())


def forecast(name: str) -> dict:
    ctx = prep(name, "context")
    X, _ = rankers.design_matrix(ctx)
    w, y = ctx["W"].to_numpy(), ctx["outcome"].to_numpy()
    better = ctx["better"].iat[0]
    p0 = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=rankers.SEED).split(X):
        idx = tr[w[tr] == 0]
        p0[te] = rankers._clf().fit(X[idx], y[idx]).predict_proba(X[te])[:, 1]
    p0 = rankers._clip(p0)
    alpha, gamma, beta = rankers.scale_fit(logit(p0), w, y)
    lin = alpha + gamma * logit(p0)
    s = sign(better)
    b = s * (expit(lin + beta) - expit(lin))
    risk = p0 if better == "lower" else 1 - p0
    base = float(y[w == 0].mean())
    return {
        "trial": name, "n_context": int(len(y)), "beta_logodds": float(beta),
        "recalibration_alpha": alpha, "recalibration_gamma": gamma,
        "raw_ate_context_per_1000": float(1000 * s * (y[w == 1].mean() - y[w == 0].mean())),
        "control_base_rate": base, "p1mp": base * (1 - base),
        "predicted_gap_per_1000": {str(q): 1000 * (top_mean(b, b, q) - top_mean(b, risk, q))
                                   for q in BUDGETS},
        "predicted_ate_per_1000": float(1000 * b.mean()),
    }


def main() -> None:
    res = {t: forecast(t) for t in TRIALS}
    OUT.write_text(json.dumps(res, indent=1))
    for t, r in res.items():
        print(f"{t:12s} base={r['control_base_rate']:.3f} beta={r['beta_logodds']:+.3f} "
              f"pred_gap@20%={r['predicted_gap_per_1000']['0.2']:+.2f}/1000 "
              f"pred_ATE={r['predicted_ate_per_1000']:+.2f}/1000 raw_ATE={r['raw_ate_context_per_1000']:+.2f} "
              f"gamma={r['recalibration_gamma']:.2f}")


if __name__ == "__main__":
    main()
