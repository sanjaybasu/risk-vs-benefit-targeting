"""Frozen scorer for the contract fields of Rose v0 (interval_90, p_harm, support, abstain).

Run by the custodians with the sealed folder as an argument; this file names no folder.

Exact-truth datasets (Tier A, truth_exact rows): unit coverage of [lo90, hi90] for the true effect
tau = s * (y_act - y_ref); Brier score of p_harm against 1{tau < 0}; the same two among rows that
did not abstain; abstention rate.
Randomized trials: group-level coverage (panel C38) - rows binned into deciles of the predicted
mean; a bin covers if its AIPW mean effect lies inside the bin's mean interval; reported as the
share of bins covered, with the AIPW bin means and intervals. Unit coverage is not identified
there and is not reported.

Usage: python scripts/19_score_contract.py --sealed DIR --predictions DIR --manifest FILE
       --trials a,b --out FILE [--allow-subset]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from system_three.estimators.dr import aipw_scores


def score(sealed: pd.DataFrame, pred: pd.DataFrame, meta: dict, subset: bool) -> dict:
    s = 1.0 if meta["better"] == "higher" else -1.0
    two = sealed[sealed["action"].isin([meta["action"], meta["reference"]])]
    if subset:
        two = two[two["row_id"].isin(set(pred["row_id"]))]
    d = two.merge(pred, on="row_id", how="inner", validate="1:1").reset_index(drop=True)
    out = {"n": int(len(d)), "abstain_rate": float(d["abstain"].mean())}
    lo, hi = d["lo90"].to_numpy() * s, d["hi90"].to_numpy() * s  # back to the outcome scale
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    if bool(d["truth_exact"].all()):
        tau = d["y_act"].to_numpy(float) - d["y_ref"].to_numpy(float)
        cov = (tau >= lo) & (tau <= hi)
        harm = (s * tau < 0).astype(float)
        keep = ~d["abstain"].to_numpy(bool)
        out.update({
            "unit_coverage_90": float(cov.mean()),
            "unit_coverage_90_not_abstained": float(cov[keep].mean()) if keep.any() else None,
            "p_harm_brier": float(np.mean((d["p_harm"].to_numpy() - harm) ** 2)),
            "p_harm_brier_not_abstained": float(np.mean((d["p_harm"].to_numpy()[keep] - harm[keep]) ** 2))
            if keep.any() else None,
        })
        return out
    w = (d["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:
        e = np.full(len(d), meta["p_action"] / (meta["p_action"] + meta["p_reference"]))
    else:
        p = d["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float),
                    d["m1"].to_numpy(float), "higher")  # outcome scale, not sign-adjusted
    mean_pred = d["benefit"].to_numpy() * s
    bins = pd.qcut(pd.Series(mean_pred).rank(method="first"), 10, labels=False).to_numpy()
    rows, covered = [], 0
    for b in range(10):
        m = bins == b
        gb, lb, hb = float(g[m].mean()), float(lo[m].mean()), float(hi[m].mean())
        se = float(g[m].std(ddof=1) / np.sqrt(m.sum()))
        rows.append({"bin": b, "n": int(m.sum()), "aipw_mean": gb, "aipw_se": se,
                     "mean_lo90": lb, "mean_hi90": hb, "mean_pred": float(mean_pred[m].mean())})
        covered += lb <= gb <= hb
    out.update({"group_coverage_share": covered / 10, "bins": rows})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--sealed", "--predictions", "--manifest", "--out"):
        ap.add_argument(k, required=True, type=Path)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--allow-subset", action="store_true")
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    res = {"scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": {}}
    for t in a.trials.split(","):
        pf = a.predictions / f"{t}.parquet"
        if hashlib.sha256(pf.read_bytes()).hexdigest() != man[t]:
            raise ValueError(f"{t}: predictions hash does not match the posted manifest")
        res["trials"][t] = score(pd.read_parquet(a.sealed / f"{t}_test.parquet"), pd.read_parquet(pf),
                                 json.loads((a.sealed / f"{t}_meta.json").read_text()), a.allow_subset)
        print("scored", t, flush=True)
    a.out.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
