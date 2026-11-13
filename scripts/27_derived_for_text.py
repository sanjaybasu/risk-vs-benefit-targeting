"""Derived quantities quoted in the manuscript text, computed from results files and context/dev data only.

- Holdout contest per trial (log predictive density, H1 minus intercept-only).
- Leave-one-trial-out gain per trial.
- Random-effects tau and 95% prediction interval for a new trial, per ranking (benefit minus risk).
- Unweighted means across the 13 trials of scale-minus-risk and DR-minus-risk.
- Size of the weak-form test at slope 0 under the two weightings (from the power simulation).
- Reference Brier for Rose's probability of harm: a constant forecast at the development-split harm
  rate (Tier A dev split holds both outcomes), scored against itself as p(1 - p).
Output: results/derived_for_text.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)
R = ROOT / "results"
CONF = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
        "sixcity", "newhaven", "lenta", "x5"]


def main() -> None:
    h = json.loads((R / "h1_analysis.json").read_text())
    hf = json.loads((R / "holdout_forecasts.json").read_text())
    sm = json.loads((R / "scoring_summary.json").read_text())
    pw = json.loads((R / "h1_power_simulation.json").read_text())
    out = {}
    df1 = len(hf["fit_trials"]) - 2
    df0 = len(hf["fit_trials"]) - 1
    per = {}
    for t, r in hf["forecasts"].items():
        y = sm["per_trial"][t]["predictions"]["scale-minus-risk"]["estimate"]
        per[t] = mr.lpd(y, r["h1"]["mean"], r["h1"]["var"], df1) - mr.lpd(y, r["null"]["mean"], r["null"]["var"], df0)
    out["holdout_lpd_gain_by_trial"] = per
    out["loo_lpd_gain_by_trial"] = {t: r["lpd_h1"] - r["lpd_null"] for t, r in h["leave_one_out"].items()}
    pi = {}
    for s in ("predictions", "predictions_rose", "predictions_causalpfn", "predictions_dopfn", "predictions_llm"):
        e = np.array([sm["per_trial"][t][s]["benefit-minus-risk"]["estimate"] for t in CONF])
        v = np.maximum(np.array([sm["per_trial"][t][s]["benefit-minus-risk"]["boot_sd"] for t in CONF]) ** 2, mr.V_FLOOR)
        f = mr.fit_re(e, np.ones((len(e), 1)), v)
        se = float(np.sqrt(f.vcov[0, 0]))
        c = stats.t.ppf(0.975, f.df)
        half = c * np.sqrt(f.tau2 + se**2)
        pi[s] = {"mean": float(f.beta[0]), "tau": float(np.sqrt(f.tau2)), "pi95": [float(f.beta[0] - half), float(f.beta[0] + half)]}
    out["new_trial_prediction_interval_benefit_minus_risk"] = pi
    out["unweighted_mean"] = {k: float(np.mean([sm["per_trial"][t]["predictions"][k]["estimate"] for t in CONF]))
                              for k in ("scale-minus-risk", "benefit-minus-risk", "benefit-minus-scale")}
    g = pw["grid"]
    out["weak_form_size_slope0"] = {
        "paired_weights": [g[k]["p_weak_form_positive"] for k in g if k.startswith("noise=paired,slope=0.0") and "trunc=False" in k],
        "design_weights": [g[k]["p_weak_form_positive"] for k in g if k.startswith("weights=design")]}
    from system_three.splits import load
    ref = {}
    for n in [k for k in sm["contract"] if k.startswith("predictions_rose__routerbench")]:
        name = n.split("__", 1)[1]
        dev = load(name, "dev")
        d = dev[dev["y_act"].isin([0, 1]) & dev["y_ref"].isin([0, 1])]
        harm = float(np.mean((d["y_act"] - d["y_ref"]) < 0))
        ref[name] = {"dev_harm_rate": harm, "reference_brier": harm * (1 - harm),
                     "rose_brier": sm["contract"][n]["p_harm_brier"]}
    out["p_harm_reference_brier"] = ref
    (R / "derived_for_text.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
