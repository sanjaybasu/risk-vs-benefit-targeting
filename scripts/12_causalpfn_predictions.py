"""Paper 2 contender: CausalPFN (Balazadeh et al., arXiv 2506.07918), frozen and off the shelf.

For each confirmatory trial: fit CausalPFN in context on the binarized context split (at most
CTX_CAP rows, sampled within arm with a fixed seed; missing values imputed with the context
median) and estimate the CATE for every test state. Runs on a Modal L4 GPU; no training.

Output: predictions_causalpfn/<trial>.parquet in the scorer's format. The benefit column is the
sign-adjusted CausalPFN CATE (larger = act first, in outcome probability per action); risk, scale,
m0, and m1 are copied from the main predictions so the frozen scorer runs unchanged and its
benefit-minus-risk contrast scores CausalPFN against risk ranking. Hashes go to
results/causalpfn_manifest.json, posted before unsealing.

Cost: under 1 GPU-hour on L4 for all trials. License: CausalPFN code, Apache-2.0.

Usage: modal run scripts/12_causalpfn_predictions.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["gusto", "ist", "synbiotic", "elaia1", "elaia2", "voter", "sf_email", "karlan_list",
          "stepup", "oregon", "hillstrom", "sixcity", "newhaven", "lenta", "x5"]
CTX_CAP = 20_000
SEED = 20261112

app = modal.App("system-three-causalpfn")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("causalpfn==0.1.5", "torch==2.5.1", "numpy<2.3", "pandas==2.2.3", "pyarrow==18.1.0"))


@app.function(image=image, gpu="L4", timeout=3600, memory=32768)
def cate(trial: str, Xc, w, y, Xt) -> dict:
    import numpy as np
    from causalpfn import CATEEstimator

    est = CATEEstimator(device="cuda").fit(Xc.astype(np.float32), w.astype(np.float32), y.astype(np.float32))
    out = np.concatenate([est.estimate_cate(Xt[i:i + 20000].astype(np.float32))
                          for i in range(0, len(Xt), 20000)])
    return {"trial": trial, "cate": out.astype(np.float64)}


def prepare(trial: str):
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))  # modal's local entrypoint runs outside the uv venv
    from system_three import rankers
    from system_three.analysis import prep
    from system_three.splits import load

    ctx = prep(trial, "context")
    rng = np.random.default_rng(SEED)
    keep = []
    for arm in (0, 1):
        idx = np.flatnonzero(ctx["W"].to_numpy() == arm)
        n = min(len(idx), int(round(CTX_CAP * len(idx) / len(ctx))))
        keep.append(rng.choice(idx, n, replace=False) if len(idx) > n else idx)
    ctx = ctx.iloc[np.sort(np.concatenate(keep))].reset_index(drop=True)
    Xc, cols = rankers.design_matrix(ctx)
    med = np.nanmedian(Xc, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    Xc = np.where(np.isnan(Xc), med, Xc)
    test = load(trial, "test")
    Xt, _ = rankers.design_matrix(test, cols)
    Xt = np.where(np.isnan(Xt), med, Xt)
    return ctx, Xc, test, Xt


@app.local_entrypoint()
def main(trials: str = ""):
    import numpy as np
    import pandas as pd

    run = [t for t in trials.split(",") if t] or TRIALS
    jobs, meta = [], {}
    for t in run:
        ctx, Xc, test, Xt = prepare(t)
        jobs.append((t, Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), Xt))
        meta[t] = (ctx["better"].iat[0], test["row_id"].to_numpy(), len(ctx))
    out_dir = ROOT / "predictions_causalpfn"
    out_dir.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "causalpfn_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in cate.starmap(jobs):
        t = r["trial"]
        better, row_id, n_ctx = meta[t]
        main_pred = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        if not np.array_equal(main_pred["row_id"].to_numpy(), row_id):
            raise ValueError(f"{t}: row order differs from main predictions")
        s = 1.0 if better == "higher" else -1.0
        pred = main_pred.copy()
        pred["benefit"] = s * r["cate"]
        p = out_dir / f"{t}.parquet"
        pred.to_parquet(p, index=False)
        man[t] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "n_context_used": int(n_ctx)}
        print(t, "ok", man[t]["n_context_used"])
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
