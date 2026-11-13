"""Registered wave-2 analyses (PREREGISTRATION_WAVE2.md Section 5), with the frozen 07 functions.

H1-W2: slope of observed scale-minus-risk on the scale-predicted gap in wave-2 trials with the
beneficial sign (confirmatory only if at least 8 such trials by the access cutoff; else descriptive).
H5a: the same slope in unflagged trials. H5b: |observed - forecast| larger in flagged trials
(one-sided Wilcoxon rank-sum; descriptive with fewer than 3 flagged).
Q2-W2: pooled gain over risk ranking at 20% for DR-learner, CausalPFN, Rose v2 (random effects,
Knapp-Hartung, floor), with between-trial SD and new-trial prediction interval.
Combined: H1 on waves 1 and 2, and with a wave indicator.
Out-of-sample: the posted wave-1-fit forecasts of the wave-2 gaps against intercept-only forecasts
(log predictive density); leave-one-trial-out within wave 2.
Output: results/w2_analysis.json
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


def fit_slope(tr, post, obs):
    x = np.array([post[t]["x"] for t in tr])
    y = np.array([obs[t]["scale_minus_risk"] for t in tr])
    v = np.array([post[t]["v_primary"] for t in tr])
    X = np.column_stack([np.ones_like(x), x])
    if len(tr) < 3:
        return None
    f = mr.fit_re(y, X, v)
    return {"k": len(tr), "trials": tr, **mr.slope_test(f), "joint_0_1": mr.joint_test(f) if f.df > 0 else None}


def pool(est, sd):
    est, v = np.asarray(est, float), np.maximum(np.asarray(sd, float) ** 2, mr.V_FLOOR)
    f = mr.fit_re(est, np.ones((len(est), 1)), v)
    se = float(np.sqrt(f.vcov[0, 0]))
    c = stats.t.ppf(0.975, f.df)
    half = c * np.sqrt(f.tau2 + se**2)
    return {"k": len(est), "mean": float(f.beta[0]), "lo95": float(f.beta[0] - c * se), "hi95": float(f.beta[0] + c * se),
            "tau": float(np.sqrt(f.tau2)), "pi95_new_trial": [float(f.beta[0] - half), float(f.beta[0] + half)]}


def main() -> None:
    post = mr.posted()
    fc = json.loads((ROOT / "results" / "w2_forecasts.json").read_text())
    w2 = list(fc["trials"])
    obs_all = mr.observed(sorted((ROOT / "scoring").glob("stage*/predictions__*.json"))
                          + sorted((ROOT / "scoring" / "wave2").glob("predictions__w2_*.json")))
    elig = [t for t in w2 if fc["trials"][t]["in_H1_W2"] and t in obs_all]
    out = {"n_eligible": len(elig), "confirmatory": len(elig) >= 8}
    out["H1_W2"] = fit_slope(elig, post, obs_all)
    unfl = [t for t in elig if not fc["trials"][t]["diagnostic_flag"]]
    fl = [t for t in elig if fc["trials"][t]["diagnostic_flag"]]
    out["H5a_unflagged"] = fit_slope(unfl, post, obs_all)
    err = {t: abs(obs_all[t]["scale_minus_risk"] - post[t]["x"]) for t in elig}
    out["H5b"] = {"flagged": {t: err[t] for t in fl}, "unflagged": {t: err[t] for t in unfl},
                  "descriptive": len(fl) < 3}
    if fl and unfl:
        out["H5b"]["wilcoxon_p_one_sided"] = float(stats.mannwhitneyu([err[t] for t in fl], [err[t] for t in unfl],
                                                                      alternative="greater").pvalue)
    # Out-of-sample: posted wave-1-fit forecasts vs intercept-only from the same wave-1 trials.
    w1 = fc["wave1_fit"]["trials"]
    x1 = np.array([post[t]["x"] for t in w1])
    y1 = np.array([obs_all[t]["scale_minus_risk"] for t in w1])
    v1 = np.array([post[t]["v_primary"] for t in w1])
    f1 = mr.fit_re(y1, np.column_stack([np.ones_like(x1), x1]), v1)
    f0 = mr.fit_re(y1, np.ones((len(y1), 1)), v1)
    per = {}
    for t in elig:
        mu1, var1 = mr.predict(f1, np.array([1.0, post[t]["x"]]), post[t]["v_primary"])
        mu0, var0 = mr.predict(f0, np.array([1.0]), post[t]["v_primary"])
        y = obs_all[t]["scale_minus_risk"]
        per[t] = {"observed": y, "forecast": mu1, "lpd_gain": mr.lpd(y, mu1, var1, f1.df) - mr.lpd(y, mu0, var0, f0.df)}
    out["wave1_forecasts_of_w2"] = {"per_trial": per, "sum_lpd_gain_nats": float(sum(r["lpd_gain"] for r in per.values()))}
    # Combined waves.
    comb = [t for t in mr.confirmatory(post) if t in obs_all]
    out["H1_combined"] = fit_slope(comb, post, obs_all)
    x = np.array([post[t]["x"] for t in comb])
    y = np.array([obs_all[t]["scale_minus_risk"] for t in comb])
    v = np.array([post[t]["v_primary"] for t in comb])
    wv = np.array([1.0 if t.startswith("w2_") else 0.0 for t in comb])
    fw = mr.fit_re(y, np.column_stack([np.ones_like(x), x, wv]), v)
    out["H1_combined_wave_indicator"] = {"slope": float(fw.beta[1]), "se": float(np.sqrt(fw.vcov[1, 1])),
                                         "wave2_shift": float(fw.beta[2]), "wave2_shift_se": float(np.sqrt(fw.vcov[2, 2]))}
    # Q2-W2 pooled learned rankings, all wave-2 trials.
    q2 = {}
    for s in ("predictions", "predictions_causalpfn", "predictions_rose_v2"):
        e, sd = [], []
        for t in w2:
            p = ROOT / "scoring" / "wave2" / f"{s}__{t}.json"
            if not p.exists():
                continue
            tt = next(iter(json.loads(p.read_text())["result"]["trials"].values())) if "result" in json.loads(p.read_text()) \
                else next(iter(json.loads(p.read_text())["trials"].values()))
            r = next(x for x in tt["table"] if x["ranking"] == "benefit-minus-risk" and x["budget"] == 0.2)
            e.append(r["estimate"])
            sd.append(r["boot_sd"])
        if len(e) >= 3:
            q2[s] = pool(e, sd)
    out["Q2_W2"] = q2
    (ROOT / "results" / "w2_analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: out[k] for k in ("n_eligible", "confirmatory")}))
    for k in ("H1_W2", "H5a_unflagged", "H1_combined"):
        r = out[k]
        if r:
            print(f"{k:16s} k={r['k']:2d} slope {r['slope']:.2f} ({r['lo95']:.2f} to {r['hi95']:.2f}) p={r['p_two_sided']:.3f}")
    print("H1 combined + wave indicator", {a: round(b, 3) for a, b in out["H1_combined_wave_indicator"].items()})
    print("wave-1 forecasts of w2: sum LPD gain", round(out["wave1_forecasts_of_w2"]["sum_lpd_gain_nats"], 2))
    for t, r in per.items():
        print(f"   {t:22s} obs {r['observed']:+7.1f} fc {r['forecast']:+6.1f} gain {r['lpd_gain']:+.2f}")
    print("H5b", out["H5b"])
    for s, r in q2.items():
        print(f"Q2 {s:24s} k={r['k']} {r['mean']:+.1f} ({r['lo95']:+.1f} to {r['hi95']:+.1f}) tau {r['tau']:.1f}")


if __name__ == "__main__":
    main()
