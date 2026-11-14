"""Exploratory detail on the calibrated ensemble's 90% intervals in the model-routing pairs, after scoring.

Run after every routing prediction file was hashed and scored, so it cannot affect any prediction.
For each pair: share of prompts whose effect is 0, median interval width, share of intervals that
exclude 0, and coverage of the true effect overall, among prompts with effect 0, and among prompts
with a nonzero effect. Output: results/rose_interval_detail.json
Usage: uv run python scripts/45_rose_interval_detail.py --sealed ../system-three-sealed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RB = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4", "routerbench__llama70b_to_claude2",
      "routerbench__mixtral_to_gpt4"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=Path)
    a = ap.parse_args()
    out = {}
    for t in RB:
        s = pd.read_parquet(a.sealed / f"{t}_test.parquet")[["row_id", "y_ref", "y_act"]]
        p = pd.read_parquet(ROOT / "predictions_rose_v2" / f"{t}.parquet")[["row_id", "lo90", "hi90", "abstain"]]
        d = s.merge(p, on="row_id", how="inner")
        tau = (d["y_act"] - d["y_ref"]).to_numpy(float)  # higher is better for answer quality
        cov = (d["lo90"].to_numpy() <= tau) & (tau <= d["hi90"].to_numpy())
        nz = tau != 0
        ph = pd.read_parquet(ROOT / "predictions_rose_v2" / f"{t}.parquet")[["row_id", "p_harm_ind"]]
        d = d.merge(ph, on="row_id", how="left")
        harm = (tau < 0).astype(float)
        hv = json.loads((ROOT / "scoring" / "rose_v2" / "rose_v2_harm.json").read_text())["trials"][t]
        # constant forecast c0 (development-split harm rate) recovered from its scored Brier score:
        # B = c0^2 - 2 c0 r + r, with r the test harm rate; take the root nearest r.
        r_, B_ = harm.mean(), hv["brier_constant_dev_rate"]
        roots = [r_ + sg * np.sqrt(max(r_ ** 2 - r_ + B_, 0.0)) for sg in (-1, 1)]
        c0 = min(roots, key=lambda z: abs(z - r_))
        e_m = (d["p_harm_ind"].to_numpy(float) - harm) ** 2 - (c0 - harm) ** 2
        rng = np.random.default_rng(20261114)
        bs = [e_m[rng.integers(0, len(e_m), len(e_m))].mean() for _ in range(2000)]
        brier_diff = {"estimate": float(e_m.mean()), "lo95": float(np.percentile(bs, 2.5)), "hi95": float(np.percentile(bs, 97.5)),
                      "constant": float(c0)}
        out[t] = {"brier_diff_vs_constant": brier_diff, "n": int(len(d)), "share_effect_zero": float(np.mean(~nz)),
                  "median_width": float(np.median(d["hi90"] - d["lo90"])),
                  "share_excluding_zero": float(np.mean((d["lo90"] > 0) | (d["hi90"] < 0))),
                  "coverage_all": float(cov.mean()), "coverage_effect_zero": float(cov[~nz].mean()),
                  "coverage_effect_nonzero": float(cov[nz].mean())}
        print(t, out[t]["brier_diff_vs_constant"])
    (ROOT / "results" / "rose_interval_detail.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
