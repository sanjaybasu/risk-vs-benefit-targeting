"""Cluster-respecting honest causal forest (econml.dml.CausalForestDML) on Modal.

Fixes the cluster-free cross-fitting limitation of scripts/40_causal_forest_predictions.py:
  - Passes explicit cluster-respecting 5-fold cross-fitting splits (`cv = cluster_folds`) to
    CausalForestDML so no household/cluster spans training and nuisance-scoring folds.
  - Raises context cap from 20,000 to 60,000 (sampled at the cluster level) so all trials
    <= 60,000 context rows use 100% of their context split.

Output: predictions_cforest_cluster/<trial>.parquet + results/cforest_cluster_manifest.json
Usage:  modal run scripts/53_cluster_causal_forest.py [--trials a,b]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIALS_W12 = [
    "gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup",
    "hillstrom", "sixcity", "newhaven", "lenta", "x5",
    "w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity",
    "w2_sa_loan", "w2_mooc_megastudy", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft",
]
CTX_CAP = 60_000
SEED = 20261114

app = modal.App("system-three-cluster-cforest")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "econml==0.16.0", "scikit-learn==1.6.1", "numpy==1.26.4", "pandas==2.2.3", "scipy==1.13.1"
)


@app.function(image=image, cpu=8, memory=32768, timeout=3600)
def fit_cluster_cf(trial: str, Xc, w, y, groups, Xt) -> dict:
    import numpy as np
    from econml.dml import CausalForestDML
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    # Build exact 5-fold cluster-respecting CV splits
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    fold_of_cl = np.empty(len(uniq), dtype=int)
    fold_of_cl[perm] = np.arange(len(uniq)) % 5
    fold = fold_of_cl[inv]
    cv_splits = [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(5)]

    est = CausalForestDML(
        model_y=HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=SEED,
        ),
        model_t=LogisticRegression(C=1.0, max_iter=1000),
        discrete_treatment=True,
        n_estimators=500, min_samples_leaf=20, honest=True,
        cv=cv_splits, random_state=SEED,
    )
    est.fit(np.asarray(y, float), np.asarray(w, int), X=np.asarray(Xc, float))
    return {"trial": trial, "cate": est.effect(np.asarray(Xt, float)).astype(float).tolist()}


def _prepare_trial(trial: str):
    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(ROOT / "src"))
    from system_three import rankers
    from system_three.analysis import prep
    from system_three.splits import load

    if trial.startswith("w3_"):
        ctx = pd.read_parquet(ROOT / "data" / "processed" / trial / "context.parquet")
        test = pd.read_parquet(ROOT / "data" / "processed" / trial / "test_state.parquet")
        base_pred = pd.read_parquet(ROOT / "predictions_w3" / f"{trial}.parquet")
    else:
        ctx = prep(trial, "context")
        test = load(trial, "test")
        base_pred = pd.read_parquet(ROOT / "predictions" / f"{trial}.parquet")

    # Cluster-level subsample ONLY if context exceeds CTX_CAP
    if len(ctx) > CTX_CAP:
        rng = np.random.default_rng(SEED)
        uniq, inv = np.unique(ctx["cluster_id"].to_numpy(), return_inverse=True)
        cl_rows = [np.flatnonzero(inv == k) for k in range(len(uniq))]
        cl_lens = np.array([len(r) for r in cl_rows])
        perm = rng.permutation(len(uniq))
        m_cl = int(np.searchsorted(np.cumsum(cl_lens[perm]), CTX_CAP, side="left")) + 1
        sub_idx = np.sort(np.concatenate([cl_rows[k] for k in perm[:m_cl]]))
        ctx = ctx.iloc[sub_idx].reset_index(drop=True)

    Xc, cols = rankers.design_matrix(ctx)
    med = np.nanmedian(Xc, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xc = np.where(np.isnan(Xc), med, Xc)
    Xt, _ = rankers.design_matrix(test, cols)
    Xt = np.where(np.isnan(Xt), med, Xt)
    return ctx, Xc, test, Xt, base_pred


@app.local_entrypoint()
def main(trials: str = ""):
    import numpy as np

    run = [t for t in trials.split(",") if t] or TRIALS_W12
    jobs, meta = [], {}
    for t in run:
        ctx, Xc, test, Xt, base_pred = _prepare_trial(t)
        jobs.append((t, Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["cluster_id"].to_numpy(), Xt))
        meta[t] = (ctx["better"].iat[0], test["row_id"].to_numpy(), base_pred)

    out = ROOT / "predictions_cforest_cluster"
    out.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "cforest_cluster_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in fit_cluster_cf.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300], flush=True)
            continue
        t = r["trial"]
        better, rid, base = meta[t]
        assert np.array_equal(base["row_id"].to_numpy(), rid), f"{t}: row_id mismatch"
        p = base[["row_id", "risk", "scale", "m0", "m1"]].copy()
        p["benefit"] = (1.0 if better == "higher" else -1.0) * np.asarray(r["cate"], float)
        f = out / f"{t}.parquet"
        p.to_parquet(f, index=False)
        man[t] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        print(f"  {t:22s} ok (n_test={len(p)})", flush=True)
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
    print("WROTE", man_path)
