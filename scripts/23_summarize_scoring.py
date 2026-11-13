"""Summarize scorer outputs (stage 1 and 2) into one table for the paper. Reads scoring/ only.

For each trial and prediction set, at the 20% budget: the set's ranking minus risk ranking
(per 1,000 actions, 95% cluster-bootstrap CI), policy values, and RATE AUTOC. The main set also
gives scale minus risk and benefit minus scale. Contract outputs give coverage and abstention.
Pooled means across the confirmatory trials are inverse-variance weighted (bootstrap SD) and
random-effects pooled with the frozen 07 functions; they are descriptive (secondary analyses).

Usage: uv run python scripts/23_summarize_scoring.py
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
CONF = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
        "sixcity", "newhaven", "lenta", "x5"]


def rows(path: Path, budget=0.2) -> dict:
    t = next(iter(json.loads(path.read_text())["trials"].values()))
    return {x["ranking"]: x for x in t["table"] if x["budget"] == budget} | {"_n": t["n"], "_rate": t.get("rate", {})}


def pool(est, sd):
    est, v = np.asarray(est, float), np.maximum(np.asarray(sd, float) ** 2, mr.V_FLOOR)  # registered floor
    f = mr.fit_re(est, np.ones((len(est), 1)), v)
    se = float(np.sqrt(f.vcov[0, 0]))
    c = stats.t.ppf(0.975, f.df)
    return {"k": len(est), "mean": float(f.beta[0]), "lo95": float(f.beta[0] - c * se),
            "hi95": float(f.beta[0] + c * se), "tau2": f.tau2}


def main() -> None:
    files = {p.stem: p for st in ("stage1", "stage2") for p in (ROOT / "scoring" / st).glob("*.json")
             if p.name != "outputs_manifest.json"}
    out = {"per_trial": {}, "pooled_benefit_minus_risk_q20": {}, "contract": {}}
    sets = sorted({k.split("__")[0] for k in files if not k.endswith("__contract")})
    for s in sets:
        ests, sds = [], []
        for t in CONF + ["elaia1", "oregon"] + [k.split("__", 1)[1] for k in files
                                                 if k.startswith("predictions_tierA__")]:
            k = f"{s}__{t}"
            if k not in files:
                continue
            r = rows(files[k])
            rec = {"n": r["_n"], **{nm: {f: r[nm][f] for f in ("estimate", "lo95", "hi95", "boot_sd")}
                                    for nm in ("risk", "scale", "benefit", "random", "benefit-minus-risk",
                                               "scale-minus-risk", "benefit-minus-scale")},
                   "benefit_autoc": r["_rate"].get("benefit_autoc"), "risk_autoc": r["_rate"].get("risk_autoc")}
            out["per_trial"].setdefault(t, {})[s] = rec
            if t in CONF:
                ests.append(rec["benefit-minus-risk"]["estimate"])
                sds.append(rec["benefit-minus-risk"]["boot_sd"])
        if len(ests) >= 3:
            out["pooled_benefit_minus_risk_q20"][s] = pool(ests, sds)
    for k, p in files.items():
        if k.endswith("__contract"):
            out["contract"][k.removesuffix("__contract")] = next(iter(json.loads(p.read_text())["trials"].values())) \
                if "trials" in json.loads(p.read_text()) else json.loads(p.read_text())
    # Main-set pooled scale-minus-risk and benefit-minus-scale.
    for nm in ("scale-minus-risk", "benefit-minus-scale"):
        e = [(out["per_trial"][t]["predictions"][nm]["estimate"], out["per_trial"][t]["predictions"][nm]["boot_sd"])
             for t in CONF if "predictions" in out["per_trial"].get(t, {})]
        if len(e) >= 3:
            out[f"pooled_main_{nm}_q20"] = pool(*zip(*e))
    (ROOT / "results" / "scoring_summary.json").write_text(json.dumps(out, indent=1, default=float))
    for s, r in out["pooled_benefit_minus_risk_q20"].items():
        print(f"{s:24s} k={r['k']:2d} benefit-minus-risk {r['mean']:+7.2f} ({r['lo95']:+.2f}, {r['hi95']:+.2f})")


if __name__ == "__main__":
    main()
