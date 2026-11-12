"""Paper 2 contender: Do-PFN (Robertson et al., arXiv 2506.06039), frozen and off the shelf.

The official repository (github.com/jr2021/Do-PFN) states no license. Sanjay directed its
inclusion (2026-11-12) for research evaluation only: the code is cloned at a pinned commit inside
an isolated Modal container, its pickled weights are loaded only there, and none of its code
enters this repository.

For each confirmatory trial: fit Do-PFN in context on the binarized context split, treatment in
column 0 (the repository's convention), at most CTX_CAP rows sampled within arm (seed 20261112),
at most MAX_FEATURES covariates (highest context variance; affects Lenta only), missing values
imputed with the context median; estimate the CATE for every test state with predict_cate.
Output: predictions_dopfn/<trial>.parquet in the scorer's format (benefit = sign-adjusted CATE;
other columns from the main predictions) and results/dopfn_manifest.json.

Usage: modal run scripts/14_dopfn_predictions.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["gusto", "ist", "synbiotic", "elaia1", "elaia2", "voter", "sf_email", "karlan_list",
          "stepup", "oregon", "hillstrom", "sixcity", "newhaven", "lenta", "x5"]
COMMIT = "90d67433b43c4d52d752dc336070f525ff856e0b"
CTX_CAP = 3_000
MAX_FEATURES = 50
SEED = 20261112

app = modal.App("system-three-dopfn")
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git")
    .pip_install("torch==2.3.1", "scikit-learn==1.4.2", "einops>=0.6.0", "scipy>=1.8.0",
                 "torchmetrics==1.2.0", "pandas==2.1.4", "pyyaml>=6.0.1", "kditransform>=0.2.0",
                 "numba>=0.58.1", "tqdm>=4.66.1", "numpy<2", "requests>=2.31.0")
    .run_commands(f"git clone https://github.com/jr2021/Do-PFN /root/dopfn && "
                  f"cd /root/dopfn && git checkout {COMMIT}")
)


@app.function(image=image, gpu="L4", timeout=3600, memory=32768)
def cate(trial: str, Xc, w, y, Xt) -> dict:
    import os
    import sys

    import numpy as np

    os.chdir("/root/dopfn")
    sys.path.insert(0, "/root/dopfn")
    from scripts.transformer_prediction_interface.base import DoPFNRegressor

    import torch

    m = DoPFNRegressor()  # loads artifacts/dopfn_config.pkl; device is set from that config
    m.device = "cuda"
    # The repository's interface takes torch tensors (its datasets yield tensors).
    m.fit(torch.from_numpy(np.column_stack([w, Xc]).astype(np.float32)),
          torch.from_numpy(y.astype(np.float32)))
    out = []
    for i in range(0, len(Xt), 5000):
        xt = Xt[i:i + 5000].astype(np.float32)
        xin = torch.from_numpy(np.column_stack([np.zeros(len(xt), np.float32), xt]))
        c = m.predict_cate(xin)
        out.append(c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c))
    return {"trial": trial, "cate": np.concatenate(out).astype(np.float64).ravel()}


def prepare(trial: str):
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
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
    if Xc.shape[1] > MAX_FEATURES:
        top = np.argsort(-np.var(Xc, axis=0))[:MAX_FEATURES]
        Xc, cols, med = Xc[:, top], [cols[i] for i in top], med[top]
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
        meta[t] = (ctx["better"].iat[0], test["row_id"].to_numpy(), len(ctx), Xc.shape[1])
    out_dir = ROOT / "predictions_dopfn"
    out_dir.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "dopfn_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in cate.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300])
            continue
        t = r["trial"]
        better, row_id, n_ctx, p = meta[t]
        main_pred = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        if not np.array_equal(main_pred["row_id"].to_numpy(), row_id) or len(r["cate"]) != len(row_id):
            raise ValueError(f"{t}: row mismatch")
        pred = main_pred.copy()
        pred["benefit"] = (1.0 if better == "higher" else -1.0) * r["cate"]
        f = out_dir / f"{t}.parquet"
        pred.to_parquet(f, index=False)
        man[t] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "n_context_used": int(n_ctx),
                  "n_features": int(p)}
        print(t, "ok", n_ctx, p)
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
