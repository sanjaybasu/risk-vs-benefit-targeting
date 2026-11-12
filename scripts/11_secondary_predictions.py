"""Frozen code for the two secondary analyses that the registration (r2vsu, Section 6) commits to
hash before any unsealing: leave-one-environment-out (LOEO) value and the split-context moderator.

Both produce prediction files in the same format as scripts/08_build_predictions.py, so the frozen
scorer (scripts/score_sealed.py) scores them unchanged.

LOEO. Environments: the trial's stratum. Where a trial has more than 20 strata, strata are grouped
into 5 environment groups by a salted hash. For each environment e: fit the rankers on context rows
outside e, score the test states inside e. Concatenate across e. Trials: ELAIA-2 (hospital),
synbiotic (population block), StepUp (cohort), Michigan (randomization block, grouped).
Output: predictions_loeo/<trial>.parquet.

Split context. Context clusters are split into halves A and B by a salted hash. The moderator is
computed on A under the scale model (plug-in, same formula as 06_scale_forecast.py); the rankings
and nuisances are fit on B and score the test states. Output: predictions_splitctx/<trial>.parquet
and results/scale_forecast_splitctx.json. All 13 confirmatory trials.

Usage: uv run python scripts/11_secondary_predictions.py [loeo|splitctx] [trial ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from system_three import rankers
from system_three.analysis import prep
from system_three.estimators.dr import sign
from system_three.splits import load

ROOT = Path(__file__).resolve().parents[1]
SALT = "system-three-secondary-2026-11-12"
LOEO_TRIALS = ["elaia2", "synbiotic", "stepup", "voter"]
CONFIRMATORY = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup",
                "hillstrom", "sixcity", "newhaven", "lenta", "x5"]
Q = 0.20


def h(key: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{SALT}|{key}".encode()).digest()[:8], "big") / 2**64


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def env_of(strata: pd.Series) -> pd.Series:
    uniq = strata.astype(str).unique()
    if len(uniq) <= 20:
        return strata.astype(str)
    return strata.astype(str).map(lambda s: f"group{int(h(s) * 5)}")


def predict(f, X) -> pd.DataFrame:
    s = rankers.scores(f, X)
    return pd.DataFrame({"risk": s["risk"], "scale": s["scale"], "benefit": s["benefit"],
                         "m0": s["_m0"], "m1": s["_m1"]})


def fit_on(df: pd.DataFrame, cols=None):
    X, cols = rankers.design_matrix(df, cols)
    f = rankers.fit(X, df["W"].to_numpy(), df["outcome"].to_numpy(), df["e"].to_numpy(),
                    df["better"].iat[0], groups=df["cluster_id"].to_numpy())
    return f, cols


def loeo(trial: str) -> dict:
    ctx = prep(trial, "context")
    test = load(trial, "test")
    _, cols = rankers.design_matrix(ctx)
    env_c, env_t = env_of(ctx["stratum"]), env_of(test["stratum"])
    parts = []
    for e in sorted(env_t.unique()):
        train = ctx[env_c != e].reset_index(drop=True)
        f, _ = fit_on(train, cols)
        tt = test[env_t == e].reset_index(drop=True)
        Xt, _ = rankers.design_matrix(tt, cols)
        p = predict(f, Xt)
        p.insert(0, "row_id", tt["row_id"].to_numpy())
        parts.append(p)
    out = ROOT / "predictions_loeo"
    out.mkdir(exist_ok=True)
    pp = out / f"{trial}.parquet"
    pd.concat(parts, ignore_index=True).to_parquet(pp, index=False)
    return {"environments": int(env_t.nunique()), "sha256": sha(pp)}


def splitctx(trial: str) -> dict:
    ctx = prep(trial, "context")
    half = ctx["cluster_id"].astype(str).map(lambda c: "A" if h(c) < 0.5 else "B")
    A, B = ctx[half == "A"].reset_index(drop=True), ctx[half == "B"].reset_index(drop=True)
    # Moderator on A (plug-in; same formula as 06_scale_forecast.py).
    XA, _ = rankers.design_matrix(A)
    w, y, e = A["W"].to_numpy(), A["outcome"].to_numpy(), A["e"].to_numpy()
    m0, _, _, _ = rankers.cross_fit_p(XA, w, y, A["cluster_id"].to_numpy())
    wt = np.where(w == 1, 1 / e, 1 / (1 - e))
    alpha, gamma, beta, _ = rankers.scale_fit(logit(m0), w, y, wt)
    lin = alpha + gamma * logit(m0)
    ben = sign(A["better"].iat[0]) * (expit(lin + beta) - expit(lin))
    risk = m0 if A["better"].iat[0] == "lower" else 1 - m0
    jit = np.random.default_rng(rankers.SEED).random(len(ben))
    k = max(1, int(round(Q * len(ben))))
    gap = 1000 * (ben[np.lexsort((jit, -ben))[:k]].mean() - ben[np.lexsort((jit, -risk))[:k]].mean())
    # Rankings and nuisances on B.
    f, cols = fit_on(B)
    test = load(trial, "test")
    Xt, _ = rankers.design_matrix(test, cols)
    p = predict(f, Xt)
    p.insert(0, "row_id", test["row_id"].to_numpy())
    out = ROOT / "predictions_splitctx"
    out.mkdir(exist_ok=True)
    pp = out / f"{trial}.parquet"
    p.to_parquet(pp, index=False)
    return {"moderator_gap_q20": float(gap), "beta_A": float(beta), "n_A": int(len(A)),
            "n_B": int(len(B)), "sha256": sha(pp)}


def main(argv: list[str]) -> int:
    mode, trials = argv[0], argv[1:]
    if mode == "loeo":
        trials = trials or LOEO_TRIALS
        path = ROOT / "results" / "loeo_manifest.json"
        fn = loeo
    elif mode == "splitctx":
        trials = trials or CONFIRMATORY
        path = ROOT / "results" / "scale_forecast_splitctx.json"
        fn = splitctx
    else:
        raise SystemExit("mode must be loeo or splitctx")
    res = json.loads(path.read_text()) if path.exists() else {}
    for t in trials:
        res[t] = fn(t)
        print(t, {k: v for k, v in res[t].items() if k != "sha256"}, flush=True)
        path.write_text(json.dumps(res, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
