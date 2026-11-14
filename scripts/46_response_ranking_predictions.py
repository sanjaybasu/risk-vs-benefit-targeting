"""Exploratory practitioner comparator (not pre-specified): the response ranking.

Campaigns and marketers usually rank people by the predicted chance of a good outcome if contacted.
This script builds that ranking from the m1 column of the main prediction files, which were hashed and
posted before any scoring, so it uses no new model and no test outcome. In the new files, the `risk`
column is replaced by the response score (P(desirable outcome | x, action)); `scale` and `benefit`
are unchanged. The frozen scorer's paired contrasts "scale-minus-risk" and "benefit-minus-risk"
then compare the scale and doubly robust rankings with the response ranking on the same units, and
the "risk" row gives the response ranking's own policy value.
Output: predictions_response/<trial>.parquet and results/response_manifest.json (hashes written
before scoring).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
          "sixcity", "newhaven", "lenta", "x5",
          "w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
          "w2_mooc_megastudy", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]


def paired_sets(dt) -> None:
    """Paired comparisons with the response ranking, all from hashed columns:
    predictions_resp_vs_risk: benefit = response score (so benefit-minus-risk = response minus risk);
    predictions_<learner>_vs_resp: risk = response score, benefit = that learner's benefit column."""
    srcs = {"resp_vs_risk": "predictions", "cforest_vs_resp": "predictions_cforest",
            "causalpfn_vs_resp": "predictions_causalpfn", "rose_vs_resp": "predictions_rose_v2"}
    for name, src in srcs.items():
        out = ROOT / f"predictions_{name}"
        out.mkdir(exist_ok=True)
        man = {}
        for t in TRIALS:
            base = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
            m1 = base["m1"].astype(float)
            resp = m1 if dt[t]["better"] == "higher" else 1.0 - m1
            p = base.copy() if name == "resp_vs_risk" else pd.read_parquet(ROOT / src / f"{t}.parquet")
            if not (p["row_id"].to_numpy() == base["row_id"].to_numpy()).all():
                raise ValueError(f"{name} {t}: row order")
            if name == "resp_vs_risk":
                p["benefit"] = resp.to_numpy()
            else:
                p["risk"] = resp.to_numpy()
            f = out / f"{t}.parquet"
            p.to_parquet(f, index=False)
            man[t] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        (ROOT / "results" / f"{name}_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
        print("wrote", name, len(man))


def main() -> None:
    dt = json.loads((ROOT / "results" / "dataset_table.json").read_text())
    out = ROOT / "predictions_response"
    out.mkdir(exist_ok=True)
    man = {}
    for t in TRIALS:
        p = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        m1 = p["m1"].astype(float)
        p["risk"] = m1 if dt[t]["better"] == "higher" else 1.0 - m1
        f = out / f"{t}.parquet"
        p.to_parquet(f, index=False)
        man[t] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (ROOT / "results" / "response_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print("wrote", len(man), "files")
    paired_sets(dt)


if __name__ == "__main__":
    main()
