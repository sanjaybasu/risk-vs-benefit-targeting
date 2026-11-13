"""Memorization check for the language-model contender (registration update 4, before scoring).

Same model, settings, sampled test rows, and covariates as scripts/13_llm_comparator.py. Only the
trial description changes: place, date, organization, and named products are removed, and the
action, comparison, and outcome are described by type. If the masked run ranks like the original
run, recall of published results does not drive the original result; a large drop suggests it does.

Usage:
  uv run python scripts/30_llm_masked.py build
  uv run python scripts/30_llm_masked.py submit [--pilot 50]
  uv run python scripts/30_llm_masked.py collect
  uv run python scripts/30_llm_masked.py predictions
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("llm", ROOT / "scripts" / "13_llm_comparator.py")
llm = importlib.util.module_from_spec(spec)
sys.modules["llm"] = llm
spec.loader.exec_module(llm)

OUT = ROOT / "llm_masked"
llm.OUT, llm.STATE = OUT, OUT / "batch_state.json"

MASKED = {
    "gusto": ("adults with an acute heart attack, treated soon after symptom onset",
              "a newer clot-dissolving drug regimen", "an older clot-dissolving drug",
              "death within 30 days", "lower"),
    "ist": ("adults with a recent ischaemic stroke", "a daily antiplatelet drug for two weeks",
            "no antiplatelet drug", "dead or dependent at 6 months", "lower"),
    "synbiotic": ("healthy newborns in a low-income setting", "a short course of an oral dietary supplement",
                  "placebo", "sepsis or death within 60 days", "lower"),
    "elaia2": ("hospitalized adults with acute kidney injury taking medicines that can harm the kidney",
               "an automated alert to clinicians", "no alert",
               "kidney injury progression, dialysis, or death within 14 days", "lower"),
    "voter": ("registered voters before an election", "a mailing about voting",
              "no mailing", "voted in the election", "higher"),
    "sf_email": ("registered voters with an email address on file, before an election",
                 "an email reminder about voting", "no email", "voted in the election", "higher"),
    "karlan_list": ("prior donors to a charity receiving a fundraising letter",
                    "a letter with a fundraising incentive", "a standard letter",
                    "made a donation", "higher"),
    "stepup": ("members of a gym who joined an exercise study", "a digital program to encourage exercise",
               "a minimal program", "visited the gym at least once during the 4 weeks", "higher"),
    "hillstrom": ("customers of an online retailer", "a marketing email", "no email",
                  "visited the website within 2 weeks", "higher"),
    "sixcity": ("registered voters before a local election", "an in-person visit encouraging them to vote",
                "no visit", "voted in the election", "higher"),
    "newhaven": ("registered voters before an election", "an in-person visit encouraging them to vote",
                 "no visit", "voted in the election", "higher"),
    "lenta": ("loyalty-card customers of a grocery chain", "a marketing text message", "no message",
              "visited a store after the campaign", "higher"),
    "x5": ("loyalty-card customers of a grocery chain", "a marketing text message", "no message",
           "made a purchase after the campaign", "higher"),
}
llm.TRIAL = MASKED  # prompt() reads TRIAL; everything else is unchanged


def write_predictions() -> None:
    req = pd.read_parquet(OUT / "requests.parquet")
    res = pd.DataFrame([json.loads(x) for x in (OUT / "results.jsonl").read_text().splitlines()])
    d = req.merge(res[["custom_id", "effect"]], on="custom_id", how="left")
    out = ROOT / "predictions_llm_masked"
    out.mkdir(exist_ok=True)
    man = {}
    for t in llm.CONFIRMATORY:
        sub = d[d["trial"] == t]
        main_p = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        main_p["row_id"] = main_p["row_id"].astype(str)
        p = main_p.merge(sub[["row_id", "effect"]], on="row_id", how="inner")
        s = 1.0 if MASKED[t][4] == "higher" else -1.0
        miss = int(p["effect"].isna().sum())
        p["benefit"] = s * p["effect"].fillna(0.0)
        p = p.drop(columns=["effect"])
        f = out / f"{t}.parquet"
        p.to_parquet(f, index=False)
        man[t] = {"n": int(len(p)), "missing_answers": miss, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (ROOT / "results" / "llm_masked_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({t: (v["n"], v["missing_answers"]) for t, v in man.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "submit", "collect", "predictions"])
    ap.add_argument("--pilot", type=int, default=None)
    a = ap.parse_args()
    {"build": llm.build, "submit": lambda: llm.submit(a.pilot), "collect": llm.collect,
     "predictions": write_predictions}[a.mode]()
