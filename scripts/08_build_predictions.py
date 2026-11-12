"""Fit the three rankers on context and score the test states; hash everything (red-team F43).

For each trial in analysis.ALL_BINARY: fit rankers on the binarized context split (cluster-level
cross-fitting), score test_state.parquet (states only; no actions or outcomes exist here), and
write predictions/<trial>.parquet (row_id, risk, scale, benefit, m0, m1), the pickled Fitted
object, and predictions/manifest.json with SHA-256 of each file. The scorer refuses any
prediction file whose hash differs from the posted manifest.

Usage: uv run python scripts/08_build_predictions.py [trial ...]
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from system_three import rankers
from system_three.analysis import ALL_BINARY, prep
from system_three.splits import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "predictions"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build(trial: str) -> dict:
    t0 = time.time()
    ctx = prep(trial, "context")
    X, cols = rankers.design_matrix(ctx)
    f = rankers.fit(X, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(),
                    ctx["better"].iat[0], groups=ctx["cluster_id"].to_numpy())
    test = load(trial, "test")
    Xt, _ = rankers.design_matrix(test, cols)
    s = rankers.scores(f, Xt)
    pred = test[["row_id"]].copy()
    for k_out, k_in in (("risk", "risk"), ("scale", "scale"), ("benefit", "benefit"),
                        ("m0", "_m0"), ("m1", "_m1")):
        pred[k_out] = np.asarray(s[k_in], float)
    OUT.mkdir(exist_ok=True)
    pp = OUT / f"{trial}.parquet"
    pred.to_parquet(pp, index=False)
    mp = OUT / f"{trial}_fitted.pkl"
    with mp.open("wb") as fh:
        pickle.dump({"fitted": f, "columns": cols}, fh)
    return {"predictions": sha256(pp), "fitted": sha256(mp), "n_test_states": int(len(pred)),
            "beta": f.beta, "auc_p0_oof": f.auc_p0_oof, "seconds": round(time.time() - t0, 1)}


def main(argv: list[str]) -> int:
    trials = argv or ALL_BINARY
    man_path = OUT / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for t in trials:
        man[t] = build(t)
        print(f"{t:12s} n_test={man[t]['n_test_states']:>8,} beta={man[t]['beta']:+.3f} "
              f"auc={man[t]['auc_p0_oof']:.3f} ({man[t]['seconds']}s)", flush=True)
        man_path.parent.mkdir(exist_ok=True)
        man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
