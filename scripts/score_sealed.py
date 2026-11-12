"""Frozen scorer for the sealed test split. Run by the custodians, once per unsealing stage.

Inputs (paths are arguments; this file names no folder):
  --sealed DIR       custodian's copy of the sealed folder: <trial>_test.parquet (row_id,
                     cluster_id, stratum, action, outcome, propensity, ...) and <trial>_meta.json
                     (better, reference, action, design arm probabilities)
  --predictions DIR  Waymark-submitted <trial>.parquet with row_id, risk, scale, benefit, m0, m1
  --manifest FILE    posted prediction manifest: {trial: sha256 of <trial>.parquet}
  --trials a,b,c     trials to score in this stage
  --out FILE         JSON results

Checks before scoring (red-team F43, F55, F56, F57): each predictions file matches its posted
SHA-256; clusters, arms, direction, and design come only from the sealed folder; a predictions
file that carries cluster or outcome columns is refused; every sealed row in the two contrast
arms has exactly one prediction, and every prediction has a sealed row.

Per trial and budget q in {0.10, 0.20, 0.30}: AIPW policy value per 1,000 actions for each
ranking and the paired contrasts benefit-risk, scale-risk, benefit-scale, with 2,000-replicate
percentile cluster bootstrap 95% CIs and the bootstrap SD. Seed 20261112. Nothing is tuned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from system_three.estimators.dr import aipw_scores
from system_three.metrics import per_1000

BUDGETS = (0.10, 0.20, 0.30)
N_BOOT = 2000
SEED = 20261112
RANKS = ("risk", "scale", "benefit")
CONTRASTS = (("benefit", "risk"), ("scale", "risk"), ("benefit", "scale"))
FORBIDDEN = {"cluster_id", "stratum", "action", "outcome", "propensity", "y_ref", "y_act"}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def score_trial(sealed: pd.DataFrame, pred: pd.DataFrame, meta: dict) -> dict:
    bad = FORBIDDEN & set(pred.columns)
    if bad:
        raise ValueError(f"predictions carry sealed-side columns {sorted(bad)}")
    if "cluster_id" not in sealed:
        raise ValueError("sealed file lacks cluster_id")
    two = sealed[sealed["action"].isin([meta["action"], meta["reference"]])]
    d = two.merge(pred, on="row_id", how="left", validate="1:1", indicator=True)
    if (d["_merge"] != "both").any():
        raise ValueError(f"{int((d['_merge'] != 'both').sum())} sealed rows lack a prediction")
    if not set(pred["row_id"]) <= set(sealed["row_id"]):
        raise ValueError("predictions include row_ids not in the sealed test split")
    d = d.reset_index(drop=True)
    w = (d["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:
        e = np.full(len(d), meta["p_action"] / (meta["p_action"] + meta["p_reference"]))
    else:
        p = d["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float),
                    d["m1"].to_numpy(float), meta["better"])
    s = {k: d[k].to_numpy(float) for k in RANKS}
    s["random"] = np.zeros(len(d))
    cl = d["cluster_id"].to_numpy()
    rows = []
    for q in BUDGETS:
        stats = {k: (lambda i, k=k: per_1000(g[i], s[k][i], q, SEED)) for k in s}
        for a, b in CONTRASTS:
            stats[f"{a}-minus-{b}"] = (lambda i, a=a, b=b: per_1000(g[i], s[a][i], q, SEED)
                                       - per_1000(g[i], s[b][i], q, SEED))
        for name, fn in stats.items():
            est, lo, hi, sd = boot(fn, cl)
            rows.append({"ranking": name, "budget": q, "estimate": est, "lo95": lo, "hi95": hi,
                         "boot_sd": sd})
    return {"n": int(len(d)), "n_clusters": int(pd.Series(cl).nunique()),
            "ate_per_1000": float(1000 * g.mean()), "max_ipw_weight": float(np.max(1 / np.minimum(e, 1 - e))),
            "table": rows}


def boot(fn, cl):
    uniq, inv = np.unique(cl, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(SEED)
    draws = np.array([fn(np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))]))
                      for _ in range(N_BOOT)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return float(fn(np.arange(len(cl)))), float(lo), float(hi), float(draws.std(ddof=1))


def environment() -> dict:
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"
    lock = root / "uv.lock"
    return {"git_commit": commit, "uv_lock_sha256": sha256(lock) if lock.exists() else None,
            "scorer_sha256": sha256(Path(__file__))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=Path)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--trials", required=True)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    manifest = json.loads(a.manifest.read_text())
    out = {"environment": environment(), "trials": {}}
    for t in a.trials.split(","):
        pf = a.predictions / f"{t}.parquet"
        if sha256(pf) != manifest[t]:
            raise ValueError(f"{t}: predictions hash does not match the posted manifest")
        meta = json.loads((a.sealed / f"{t}_meta.json").read_text())
        sealed = pd.read_parquet(a.sealed / f"{t}_test.parquet")
        out["trials"][t] = score_trial(sealed, pd.read_parquet(pf), meta)
        print(f"scored {t}", flush=True)
    a.out.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
