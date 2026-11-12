"""Paper 2 exact-truth layer (Tier A): risk, scale, and benefit rankings on agent-routing data.

Tier A datasets: RouterBench model pairs (both models answered every prompt) and SWE-bench
Verified system pairs. The logged action is a fair coin per prompt (03_validate_and_seal.py), so
the context split looks like logged data with propensity 0.5, and the sealed test split also holds
both potential outcomes.

Rules fixed before scoring (lens decisions, 2026-11-12):
- Binary outcomes only. An eval enters if every context score in it is 0 or 1 (decided from context
  alone); other evals (for example partial-credit GSM8K) are left out. Predictions are written
  only for test rows of eligible evals; the frozen scorer scores them with --allow-subset.
- Features: TF-IDF (word 1-2 grams, 30,000 terms, sublinear tf) of the prompt text, reduced to 64
  dimensions by truncated SVD fit on context only, plus indicators for the 40 most common evals.
  SWE-bench uses repository indicators only (no problem text in the experiments repository).
- Rankers: the same rankers.fit as Paper 1 (fixed hyperparameters, cluster-level cross-fitting).

Output: predictions_tierA/<dataset>.parquet and results/tierA_manifest.json.
Usage: uv run python scripts/16_tierA_predictions.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from system_three import rankers
from system_three.splits import load

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
            "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4",
            "swebench__p25_to_p75", "swebench__p50_to_p90", "swebench__p10_to_p50"]


def parse(state: pd.Series) -> pd.DataFrame:
    recs = [json.loads(s) for s in state]
    return pd.DataFrame(recs)


def features(ctx_state: pd.DataFrame, other: pd.DataFrame, kind: str):
    if kind == "routerbench":
        tf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=3)
        svd = TruncatedSVD(64, random_state=rankers.SEED)
        zc = svd.fit_transform(tf.fit_transform(ctx_state["prompt"].astype(str)))
        zo = svd.transform(tf.transform(other["prompt"].astype(str)))
        key = "eval"
    else:
        zc, zo = np.zeros((len(ctx_state), 0)), np.zeros((len(other), 0))
        key = "repo"
    top = ctx_state[key].value_counts().index[:40]
    oc = np.column_stack([(ctx_state[key] == v).to_numpy(float) for v in top])
    oo = np.column_stack([(other[key] == v).to_numpy(float) for v in top])
    return np.hstack([zc, oc]), np.hstack([zo, oo])


def run(name: str) -> dict:
    ctx, test = load(name, "context"), load(name, "test")
    kind = "routerbench" if name.startswith("routerbench") else "swebench"
    sc, st = parse(ctx["state"]), parse(test["state"])
    key = "eval" if kind == "routerbench" else "repo"
    binary_evals = set(ctx.groupby(sc[key].to_numpy())["outcome"].apply(lambda s: s.isin([0, 1]).all())
                       .loc[lambda b: b].index)
    keep_c = sc[key].isin(binary_evals).to_numpy()
    keep_t = st[key].isin(binary_evals).to_numpy()
    ctx, sc = ctx[keep_c].reset_index(drop=True), sc[keep_c].reset_index(drop=True)
    test, st = test[keep_t].reset_index(drop=True), st[keep_t].reset_index(drop=True)
    Xc, Xt = features(sc, st, kind)
    w = (ctx["action"] != ctx["reference"]).to_numpy(int)
    e = np.full(len(ctx), 0.5)
    f = rankers.fit(Xc, w, ctx["outcome"].to_numpy(), e, ctx["better"].iat[0],
                    groups=ctx["cluster_id"].to_numpy())
    # Context-only scale forecast (same formula as 06_scale_forecast.py), recorded before scoring:
    # the exact-truth layer tests its calibration with almost no sampling noise (panel C34).
    from scipy.special import expit, logit
    m0, _, _, _ = rankers.cross_fit_p(Xc, w, ctx["outcome"].to_numpy(), ctx["cluster_id"].to_numpy())
    lin = f.alpha + f.gamma * logit(m0)
    b = expit(lin + f.beta) - expit(lin)  # higher is better for both Tier A outcomes
    jit = np.random.default_rng(rankers.SEED).random(len(b))
    k = max(1, int(round(0.2 * len(b))))
    gap = 1000 * (b[np.lexsort((jit, -b))[:k]].mean() - b[np.lexsort((jit, -(1 - m0)))[:k]].mean())
    s = rankers.scores(f, Xt)
    pred = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "risk": s["risk"], "scale": s["scale"],
                         "benefit": s["benefit"], "m0": s["_m0"], "m1": s["_m1"]})
    out = ROOT / "predictions_tierA"
    out.mkdir(exist_ok=True)
    p = out / f"{name}.parquet"
    pred.to_parquet(p, index=False)
    return {"n_context": int(len(ctx)), "n_test_scored": int(len(pred)),
            "evals_or_repos_kept": len(binary_evals), "beta": f.beta, "auc_p0_oof": f.auc_p0_oof,
            "scale_forecast_gap_q20": float(gap),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def main() -> None:
    man = {}
    for d in DATASETS:
        man[d] = run(d)
        print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in man[d].items() if k != "sha256"},
              flush=True)
    (ROOT / "results" / "tierA_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
