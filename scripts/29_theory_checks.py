"""Checks for Supplementary Note 1 (Propositions 1-2). Context data only.

1. Simulation: in 3,000 random populations, the condition u_max + u_q > c agrees with a direct
   comparison of the risk-selected and benefit-selected sets.
2. Trials: u_max + u_q at the 20% budget under the fitted scale model, c = |beta|, r* = sigma(c/2),
   the maximum recalibrated risk, and the posted gap.
Output: results/theory_checks.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_three import rankers  # noqa: E402
from system_three.analysis import prep  # noqa: E402

TRIALS = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
          "sixcity", "newhaven", "lenta", "x5",
          "w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
          "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]


def main() -> None:
    rng = np.random.default_rng(1)
    agree = 0
    for _ in range(3000):
        c = rng.uniform(0.05, 3)
        u = rng.normal(rng.uniform(-4, 3), rng.uniform(0.2, 2), 400)
        k = int(round(rng.choice([0.1, 0.2, 0.3]) * len(u)))
        tau = expit(u) - expit(u - c)
        uq = np.sort(u)[::-1][k - 1]
        agree += (u.max() + uq > c) == (set(np.argsort(-u)[:k]) != set(np.argsort(-tau)[:k]))
    out = {"simulation_agreement": f"{agree} of 3000", "trials": {}}
    fc = json.loads((ROOT / "results" / "scale_forecast.json").read_text())
    for t in TRIALS:
        r, b = fc[t], prep(t, "context")
        X, _ = rankers.design_matrix(b)
        m0, _, _, _ = rankers.cross_fit_p(X, b["W"].to_numpy(), b["outcome"].to_numpy(), b["cluster_id"].to_numpy())
        lin = r["alpha"] + r["gamma"] * logit(m0)
        u = lin if b["better"].iat[0] == "lower" else -lin
        c = abs(r["beta"])
        uq = np.sort(u)[::-1][int(round(0.2 * len(u))) - 1]
        out["trials"][t] = {"gamma": r["gamma"], "c": c, "r_star": float(expit(c / 2)),
                            "max_recalibrated_risk": float(expit(u.max())), "umax_plus_uq": float(u.max() + uq),
                            "condition_holds": bool(u.max() + uq > c), "posted_gap": r["gap"]["0.2"]}
        print(t, out["trials"][t], flush=True)
    (ROOT / "results" / "theory_checks.json").write_text(json.dumps(out, indent=1))
    print(out["simulation_agreement"])


if __name__ == "__main__":
    main()
