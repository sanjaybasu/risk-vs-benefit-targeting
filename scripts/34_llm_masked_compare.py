"""Compare the masked and original language-model runs (registration updates 4-5).

Pooled gain over risk ranking at 20% (random effects, Knapp-Hartung, registered floor) for each run;
per-trial differences; Spearman correlation of the two runs' effect estimates within each trial
(computed from the hashed prediction files; no outcome needed).
Output: results/llm_masked_compare.json
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("summ", ROOT / "scripts" / "23_summarize_scoring.py")
summ = importlib.util.module_from_spec(spec)
sys.modules["summ"] = summ
spec.loader.exec_module(summ)
CONF = summ.CONF


def br(path: Path) -> dict:
    t = next(iter(json.loads(path.read_text())["trials"].values()))
    return next(x for x in t["table"] if x["ranking"] == "benefit-minus-risk" and x["budget"] == 0.2)


def main() -> None:
    out = {"per_trial": {}}
    e = {"original": [], "masked": []}
    for t in CONF:
        o = br(next((ROOT / "scoring").glob(f"stage*/predictions_llm__{t}.json")))
        m = br(ROOT / "scoring" / "masked" / f"predictions_llm_masked__{t}.json")
        po = pd.read_parquet(ROOT / "predictions_llm" / f"{t}.parquet")[["row_id", "benefit"]]
        pm = pd.read_parquet(ROOT / "predictions_llm_masked" / f"{t}.parquet")[["row_id", "benefit"]]
        d = po.merge(pm, on="row_id", suffixes=("_o", "_m"))
        rho = float(spearmanr(d["benefit_o"], d["benefit_m"]).statistic)
        out["per_trial"][t] = {"original": o["estimate"], "original_ci": [o["lo95"], o["hi95"]],
                               "masked": m["estimate"], "masked_ci": [m["lo95"], m["hi95"]], "spearman_effects": rho}
        e["original"].append((o["estimate"], o["boot_sd"]))
        e["masked"].append((m["estimate"], m["boot_sd"]))
    for k, v in e.items():
        out[f"pooled_{k}"] = summ.pool(*zip(*v))
    out["median_spearman"] = float(np.median([r["spearman_effects"] for r in out["per_trial"].values()]))
    (ROOT / "results" / "llm_masked_compare.json").write_text(json.dumps(out, indent=1))
    for k in ("pooled_original", "pooled_masked"):
        r = out[k]
        print(k, f"{r['mean']:.1f} ({r['lo95']:.1f} to {r['hi95']:.1f})")
    for t, r in out["per_trial"].items():
        print(f"{t:12s} orig {r['original']:+7.1f} masked {r['masked']:+7.1f} rho {r['spearman_effects']:.2f}")
    print("median rho", round(out["median_spearman"], 2))


if __name__ == "__main__":
    main()
