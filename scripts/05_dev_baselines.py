"""Dev-split run of the three rankings (risk, scale, benefit) on binary-outcome datasets.

Fits on context only and scores dev with AIPW (known design propensity; nuisance models from
context). Reports policy value per 1,000 actions and three paired contrasts at each budget:
benefit - risk (primary estimand), scale - risk, and benefit - scale (the scale decomposition).
Never reads the test split.

Usage: uv run python scripts/05_dev_baselines.py [dataset ...]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from system_three import rankers
from system_three.analysis import ALL_BINARY as BINARY  # frozen config (F76)
from system_three.analysis import prep
from system_three.estimators.dr import aipw_scores
from system_three.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "dev_baselines.json"
BUDGETS = (0.10, 0.20, 0.30)


def run(name: str) -> dict:
    t0 = time.time()
    ctx, dev = prep(name, "context"), prep(name, "dev")
    better = ctx["better"].iat[0]
    Xc, cols = rankers.design_matrix(ctx)
    Xd, _ = rankers.design_matrix(dev, cols)
    f = rankers.fit(Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(), better,
                    groups=ctx["cluster_id"].to_numpy())
    s = rankers.scores(f, Xd)
    g = aipw_scores(dev["outcome"].to_numpy(), dev["W"].to_numpy(), dev["e"].to_numpy(),
                    s["_m0"], s["_m1"], better)
    ranks = {k: v for k, v in s.items() if not k.startswith("_")}
    ranks["random"] = np.zeros(len(dev))
    cl = dev["cluster_id"].to_numpy()
    tables = []
    for a, b in (("benefit", "risk"), ("scale", "risk"), ("benefit", "scale")):
        t = evaluate(g, ranks, cl, budgets=BUDGETS, n_boot=300, contrast=(a, b))
        tables.append(t[t["ranking"] == f"{a}-minus-{b}"])
    base = evaluate(g, ranks, cl, budgets=BUDGETS, n_boot=300)
    res = pd.concat([base] + tables, ignore_index=True)
    return {
        "dataset": name, "n_context": len(ctx), "n_dev": len(dev), "better": better,
        "ate_dev_aipw_per_1000": float(1000 * g.mean()),
        "scale_beta_logodds": f.beta,
        "control_rate_context": float(ctx.loc[ctx["W"] == 0, "outcome"].mean()),
        "rank_corr_risk_benefit": float(pd.Series(ranks["risk"]).corr(pd.Series(ranks["benefit"]), method="spearman")),
        "seconds": round(time.time() - t0, 1),
        "table": res.to_dict(orient="records"),
    }


def main(argv: list[str]) -> int:
    names = argv or BINARY
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n in names:
        r = run(n)
        out[n] = r
        q20 = {x["ranking"]: x for x in r["table"] if x["budget"] == 0.2 and x["metric"] == "per_1000"}
        fmt = lambda k: f"{q20[k]['estimate']:+.1f} [{q20[k]['lo95']:+.1f}, {q20[k]['hi95']:+.1f}]"  # noqa: E731
        print(f"{n:10s} ATE/1000={r['ate_dev_aipw_per_1000']:+.1f}  q20 per1000: "
              f"benefit {fmt('benefit')} risk {fmt('risk')} scale {fmt('scale')} | "
              f"B-R {fmt('benefit-minus-risk')} S-R {fmt('scale-minus-risk')} "
              f"B-S {fmt('benefit-minus-scale')} | rho(risk,benefit)={r['rank_corr_risk_benefit']:.2f} "
              f"({r['seconds']}s)", flush=True)
        OUT.write_text(json.dumps(out, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
