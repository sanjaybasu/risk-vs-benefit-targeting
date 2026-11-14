"""Learned rankings against risk ranking across both waves (23 trials; not pre-specified as a pooled test).

For DR-learner, CausalPFN, and Rose (v0 in wave 1, v2 in wave 2; same bagged DR ranking), at the
20% budget: per-trial gain over risk ranking, random-effects pooled mean (Knapp-Hartung, registered
floor), between-trial SD, 95% prediction interval for a new trial, and a wave indicator.
Output: results/combined_learned_rankings.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)
W1 = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
      "sixcity", "newhaven", "lenta", "x5"]
W2 = ["w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
      "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
SETS = {"dr": ("predictions", "predictions"), "causalpfn": ("predictions_causalpfn", "predictions_causalpfn"),
        "rose": ("predictions_rose", "predictions_rose_v2")}


def row(set_name, t):
    p = next(iter(sorted((ROOT / "scoring").glob(f"*/{set_name}__{t}.json"))))
    tt = next(iter(json.loads(p.read_text())["trials"].values()))
    return next(x for x in tt["table"] if x["ranking"] == "benefit-minus-risk" and x["budget"] == 0.2)


def main() -> None:
    out = {}
    for k, (s1, s2) in SETS.items():
        rows = [(t, 0.0, row(s1, t)) for t in W1] + [(t, 1.0, row(s2, t)) for t in W2]
        e = np.array([r["estimate"] for _, _, r in rows])
        v = np.maximum(np.array([r["boot_sd"] for _, _, r in rows]) ** 2, mr.V_FLOOR)
        wv = np.array([w for _, w, _ in rows])
        f = mr.fit_re(e, np.ones((len(e), 1)), v)
        se = float(np.sqrt(f.vcov[0, 0]))
        c = stats.t.ppf(0.975, f.df)
        half = c * np.sqrt(f.tau2 + se**2)
        fw = mr.fit_re(e, np.column_stack([np.ones_like(wv), wv]), v)
        out[k] = {"k": len(e), "mean": float(f.beta[0]), "lo95": float(f.beta[0] - c * se), "hi95": float(f.beta[0] + c * se),
                  "tau": float(np.sqrt(f.tau2)), "pi95_new_trial": [float(f.beta[0] - half), float(f.beta[0] + half)],
                  "wave2_minus_wave1": float(fw.beta[1]), "wave2_minus_wave1_se": float(np.sqrt(fw.vcov[1, 1])),
                  "share_trials_positive_ci": float(np.mean([r["lo95"] > 0 for _, _, r in rows])),
                  "share_trials_negative_ci": float(np.mean([r["hi95"] < 0 for _, _, r in rows])),
                  "per_trial": {t: {"wave": int(w) + 1, "estimate": r["estimate"], "lo95": r["lo95"], "hi95": r["hi95"]}
                                for t, w, r in rows}}
        r = out[k]
        print(f"{k:10s} k={r['k']} {r['mean']:+.1f} ({r['lo95']:+.1f} to {r['hi95']:+.1f}) tau {r['tau']:.1f} "
              f"PI [{r['pi95_new_trial'][0]:+.1f}, {r['pi95_new_trial'][1]:+.1f}] w2-w1 {r['wave2_minus_wave1']:+.1f} "
              f"(SE {r['wave2_minus_wave1_se']:.1f}) pos {r['share_trials_positive_ci']:.2f} neg {r['share_trials_negative_ci']:.2f}")
    (ROOT / "results" / "combined_learned_rankings.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
