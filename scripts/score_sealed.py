"""Frozen scorer for the sealed test split. Run by the custodians, once, after the go/no-go gate.

Inputs (paths are arguments; this file names no folder):
  --sealed DIR       custodian's copy of the sealed folder: <trial>_test.parquet with row_id, action,
                     outcome, propensity, ... (written by 03_validate_and_seal.py)
  --predictions DIR  Waymark-submitted <trial>.parquet with row_id and columns risk, scale, benefit,
                     m0, m1, plus meta.json giving each trial's reference action, contrast action,
                     design probabilities, and outcome direction
  --trials a,b,c     which trials to score (two-stage unsealing: the 8 first, then the 3 holdouts)
  --out FILE         JSON results

Per trial and budget q in {0.10, 0.20, 0.30}: AIPW policy value per 1,000 actions for each
ranking, the paired contrasts benefit-risk (primary at q = 0.20), scale-risk, and benefit-scale,
with 2,000-replicate cluster bootstrap 95% CIs. Seed 20261112. Nothing is tuned here.

The script's SHA-256 is posted with the pre-registration; a changed script is a deviation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from system_three.estimators.dr import aipw_scores
from system_three.metrics import evaluate

BUDGETS = (0.10, 0.20, 0.30)
N_BOOT = 2000
SEED = 20261112


def score_trial(sealed: pd.DataFrame, pred: pd.DataFrame, meta: dict) -> dict:
    d = pred.merge(sealed, on="row_id", how="inner", validate="1:1")
    if len(d) != len(pred):
        raise ValueError(f"{len(pred) - len(d)} predicted rows have no sealed outcome")
    d = d[d["action"].isin([meta["action"], meta["reference"]])].reset_index(drop=True)
    w = (d["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:  # multi-arm trials with fixed design probabilities
        pa, pr = meta["p_action"], meta["p_reference"]
        e = np.full(len(d), pa / (pa + pr))
    else:  # two-arm trials: the sealed propensity is P(own arm), possibly stratum-specific
        p = d["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float),
                    d["m1"].to_numpy(float), meta["better"])
    ranks = {k: d[k].to_numpy(float) for k in ("risk", "scale", "benefit")}
    ranks["random"] = np.zeros(len(d))
    cl = d["cluster_id"].to_numpy() if "cluster_id" in d else d["row_id"].to_numpy()
    tabs = [evaluate(g, ranks, cl, budgets=BUDGETS, n_boot=N_BOOT, seed=SEED)]
    for a, b in (("benefit", "risk"), ("scale", "risk"), ("benefit", "scale")):
        t = evaluate(g, ranks, cl, budgets=BUDGETS, n_boot=N_BOOT, seed=SEED, contrast=(a, b))
        tabs.append(t[t["ranking"] == f"{a}-minus-{b}"])
    res = pd.concat(tabs, ignore_index=True)
    return {"n": int(len(d)), "ate_per_1000": float(1000 * g.mean()),
            "table": res.to_dict(orient="records")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    meta = json.loads((a.predictions / "meta.json").read_text())
    out = {"scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": {}}
    for t in a.trials.split(","):
        sealed = pd.read_parquet(a.sealed / f"{t}_test.parquet")
        pred = pd.read_parquet(a.predictions / f"{t}.parquet")
        out["trials"][t] = score_trial(sealed, pred, meta[t])
        print(f"scored {t}", flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
