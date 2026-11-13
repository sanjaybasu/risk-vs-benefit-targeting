"""Registered secondary analyses not computed by 07 (registration Section 5 lists; reported in the SI).

- Split-context H1: observed scale-minus-risk from the split-context rankings (fit on half B of
  context) regressed on the split-context moderator (computed on half A), primary weights.
- Budgets of 10% and 30%: H1 slope with the same moderator (posted at 20%) and observed gaps at
  those budgets; pooled benefit-minus-risk per budget.
- Per-trial RATE (AUTOC) for risk, scale and benefit rankings.
- Leave-one-environment-out value in the four trials with environments: benefit minus risk.
Frozen 07 functions; reads scoring outputs only. Output: results/registered_secondary.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)
spec2 = importlib.util.spec_from_file_location("summ", ROOT / "scripts" / "23_summarize_scoring.py")
summ = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(summ)


def table(path: Path, budget: float) -> dict:
    t = next(iter(json.loads(path.read_text())["trials"].values()))
    return {x["ranking"]: x for x in t["table"] if x["budget"] == budget}, t.get("rate", {})


def h1(trials, x, y, v) -> dict:
    X = np.column_stack([np.ones_like(x), x])
    return mr.slope_test(mr.fit_re(np.asarray(y), X, np.asarray(v)))


def main() -> None:
    post = mr.posted()
    conf = [t for t in mr.confirmatory(post)]
    f = lambda s, t: next((ROOT / "scoring").glob(f"stage*/{s}__{t}.json"))  # noqa: E731
    out = {}
    sc = json.loads((ROOT / "results" / "scale_forecast_splitctx.json").read_text())
    x = np.array([sc[t]["moderator_gap_q20"] for t in conf])
    y = np.array([table(f("predictions_splitctx", t), 0.2)[0]["scale-minus-risk"]["estimate"] for t in conf])
    v = np.array([post[t]["v_primary"] for t in conf])
    out["split_context_H1"] = h1(conf, x, y, v)
    xm = np.array([post[t]["x"] for t in conf])
    for b in (0.1, 0.3):
        yb = np.array([table(f("predictions", t), b)[0]["scale-minus-risk"]["estimate"] for t in conf])
        out[f"H1_budget_{b}"] = h1(conf, xm, yb, v)
        br = [table(f("predictions", t), b)[0]["benefit-minus-risk"] for t in conf]
        out[f"pooled_benefit_minus_risk_budget_{b}"] = summ.pool([r["estimate"] for r in br], [r["boot_sd"] for r in br])
    out["rate_autoc"] = {t: {k: table(f("predictions", t), 0.2)[1].get(f"{k}_autoc") for k in ("risk", "scale", "benefit")}
                         for t in conf}
    loeo = json.loads((ROOT / "results" / "loeo_manifest.json").read_text())
    out["loeo_benefit_minus_risk_q20"] = {}
    for t in loeo:
        r = table(f("predictions_loeo", t), 0.2)[0]["benefit-minus-risk"]
        m = table(f("predictions", t), 0.2)[0]["benefit-minus-risk"]
        out["loeo_benefit_minus_risk_q20"][t] = {"loeo": {k: r[k] for k in ("estimate", "lo95", "hi95")},
                                                 "main": {k: m[k] for k in ("estimate", "lo95", "hi95")}}
    (ROOT / "results" / "registered_secondary.json").write_text(json.dumps(out, indent=1, default=float))
    for k in ("split_context_H1", "H1_budget_0.1", "H1_budget_0.3"):
        r = out[k]
        print(f"{k:22s} slope {r['slope']:.2f} ({r['lo95']:.2f} to {r['hi95']:.2f}) p={r['p_two_sided']:.3f}")
    for b in (0.1, 0.3):
        r = out[f"pooled_benefit_minus_risk_budget_{b}"]
        print(f"pooled B-R budget {b}: {r['mean']:.1f} ({r['lo95']:.1f} to {r['hi95']:.1f})")
    for t, r in out["loeo_benefit_minus_risk_q20"].items():
        print(t, "LOEO %.1f (%.1f to %.1f)" % tuple(r["loeo"].values()), "main %.1f" % r["main"]["estimate"])


if __name__ == "__main__":
    main()
