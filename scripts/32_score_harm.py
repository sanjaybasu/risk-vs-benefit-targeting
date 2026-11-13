"""Frozen scorer for individual-harm fields and a harm-aware decision rule (registration update 4).

Exact-truth data only (both potential outcomes in the sealed test split). For each pair:
- Brier score against individual harm 1{harm} for: Rose v0 p_harm, Rose v1 p_harm_ind, and a
  constant forecast at the development-split harm rate (from the prediction manifest).
- Calibration of p_harm_ind: observed harm rate by decile of p_harm_ind.
- Frechet-Hoeffding bounds: in deciles of the bound midpoint, the share of deciles whose mean
  bounds contain the decile's observed harm rate.
- Decision rule at a 20% budget: act on the top 20% by Rose benefit. Report benefit (outcomes
  improved per 1,000 actions) and harms (cases made worse per 1,000 actions). Harm-aware rules skip
  cases with p_harm_ind above a threshold (0.05, 0.10, 0.20) and fill the budget with the next
  eligible cases by Rose benefit. A unit is harmed if the action makes its outcome worse.
Usage: python scripts/32_score_harm.py --sealed DIR --predictions DIR --manifest FILE --trials a,b --out FILE
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLDS = (0.05, 0.10, 0.20)
BUDGET = 0.2


def score(sealed: pd.DataFrame, pred: pd.DataFrame, meta: dict, dev_harm_rate: float) -> dict:
    d = sealed[["row_id", "y_act", "y_ref"]].merge(pred, on="row_id", how="inner", validate="1:1")
    s = 1.0 if meta["better"] == "higher" else -1.0
    tau = s * (d["y_act"].to_numpy(float) - d["y_ref"].to_numpy(float))  # +1 helped, -1 harmed
    harm = (tau < 0).astype(float)
    out = {"n": int(len(d)), "harm_rate": float(harm.mean())}
    for name, p in (("rose_v0_p_harm", d["p_harm"].to_numpy()), ("rose_v1_p_harm_ind", d["p_harm_ind"].to_numpy()),
                    ("constant_dev_rate", np.full(len(d), dev_harm_rate))):
        out[f"brier_{name}"] = float(np.mean((p - harm) ** 2))
    q = pd.qcut(pd.Series(d["p_harm_ind"]).rank(method="first"), 10, labels=False).to_numpy()
    mid = (d["fh_lo"].to_numpy() + d["fh_hi"].to_numpy()) / 2
    qb = pd.qcut(pd.Series(mid).rank(method="first"), 10, labels=False).to_numpy()
    out["calibration_p_harm_ind"] = [{"decile": int(b), "mean_pred": float(d["p_harm_ind"].to_numpy()[q == b].mean()),
                                      "observed": float(harm[q == b].mean())} for b in range(10)]
    out["fh_bounds_contain_decile_rate"] = float(np.mean([
        d["fh_lo"].to_numpy()[qb == b].mean() <= harm[qb == b].mean() <= d["fh_hi"].to_numpy()[qb == b].mean()
        for b in range(10)]))
    k = max(1, int(round(BUDGET * len(d))))
    order = np.argsort(-d["benefit"].to_numpy(), kind="stable")
    rules = {"rose_top20": order[:k]}
    ph = d["p_harm_ind"].to_numpy()
    for t in THRESHOLDS:
        rules[f"rose_top20_skip_p_harm_above_{t}"] = order[ph[order] <= t][:k]
    out["decision"] = {r: {"n_acted": int(len(i)), "benefit_per_1000": float(1000 * tau[i].sum() / max(len(i), 1)),
                           "harms_per_1000": float(1000 * harm[i].sum() / max(len(i), 1))} for r, i in rules.items()}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    for k in ("--sealed", "--predictions", "--manifest", "--out"):
        ap.add_argument(k, required=True, type=Path)
    ap.add_argument("--trials", required=True)
    a = ap.parse_args()
    man = json.loads(a.manifest.read_text())
    res = {"scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "trials": {}}
    for t in a.trials.split(","):
        pf = a.predictions / f"{t}.parquet"
        if hashlib.sha256(pf.read_bytes()).hexdigest() != man[t]["sha256"]:
            raise ValueError(f"{t}: predictions hash does not match the posted manifest")
        res["trials"][t] = score(pd.read_parquet(a.sealed / f"{t}_test.parquet"), pd.read_parquet(pf),
                                 json.loads((a.sealed / f"{t}_meta.json").read_text()), man[t]["dev_harm_rate"])
        print("scored", t, flush=True)
    a.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
