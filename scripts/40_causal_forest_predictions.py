"""Causal forest comparator (exploratory; registration update 7, hashed before scoring), on Modal.

Generalized random forest (honest causal forest) via econml.dml.CausalForestDML, in an isolated
container with its own pinned versions (econml does not support the project's scikit-learn).
Context split only, the same capped and median-imputed design matrix as the CausalPFN contender
(scripts/12_causalpfn_predictions.py, prepare()). Nuisances: gradient boosting for the outcome;
the design propensity is constant within the two-arm contrast, so the treatment model is a
regularized logistic regression. 500 trees, honest splitting, cluster-free 5-fold cross-fitting,
fixed seed. Output: predictions_cforest/<trial>.parquet (benefit = sign-adjusted CATE; other
columns copied from the main predictions), results/cforest_manifest.json.
Usage: modal run scripts/40_causal_forest_predictions.py [--trials a,b]
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpfn", ROOT / "scripts" / "12_causalpfn_predictions.py")
TRIALS = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
          "sixcity", "newhaven", "lenta", "x5",
          "w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
          "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
SEED = 20261114

app = modal.App("system-three-causal-forest")
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "econml==0.16.0", "scikit-learn==1.6.1", "numpy==1.26.4", "pandas==2.2.3", "scipy==1.13.1")


@app.function(image=image, cpu=8, memory=32768, timeout=6 * 3600)
def cate(trial: str, Xc, w, y, Xt) -> dict:
    import numpy as np
    from econml.dml import CausalForestDML
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression

    est = CausalForestDML(model_y=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                                min_samples_leaf=50, l2_regularization=1.0,
                                                                early_stopping=False, random_state=SEED),
                          model_t=LogisticRegression(C=1.0, max_iter=1000), discrete_treatment=True,
                          n_estimators=500, min_samples_leaf=20, honest=True, cv=5, random_state=SEED)
    est.fit(np.asarray(y, float), np.asarray(w, int), X=Xc)
    return {"trial": trial, "cate": est.effect(Xt).astype(float).tolist()}


@app.local_entrypoint()
def main(trials: str = ""):
    import numpy as np
    import pandas as pd

    cp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cp)
    run = [t for t in trials.split(",") if t] or TRIALS
    jobs, meta = [], {}
    for t in run:
        ctx, Xc, test, Xt = cp.prepare(t)
        jobs.append((t, Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), Xt))
        meta[t] = (ctx["better"].iat[0], test["row_id"].to_numpy())
    out = ROOT / "predictions_cforest"
    out.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "cforest_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in cate.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300])
            continue
        t = r["trial"]
        better, rid = meta[t]
        base = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        if not np.array_equal(base["row_id"].to_numpy(), rid):
            raise ValueError(f"{t}: row order differs from main predictions")
        p = base.copy()
        p["benefit"] = (1.0 if better == "higher" else -1.0) * np.asarray(r["cate"])
        f = out / f"{t}.parquet"
        p.to_parquet(f, index=False)
        man[t] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        print(t, "ok", flush=True)
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
