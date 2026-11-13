"""Rose v1 harm fields for the exact-truth data (registration update 4, written before scoring).

Rose v0's p_harm is P(CATE(x) is harmful | data); it was scored against individual harm, which it
does not target. Rose v1 adds, from the hashed m0 = P(Y = 1 | x, reference) and m1 = P(Y = 1 | x,
action) columns:
- p_harm_ind: individual harm probability under conditional independence of the two potential
  outcomes given x. For a higher-is-better outcome, harm is Y(act) = 0 and Y(ref) = 1, so
  p = (1 - m1) * m0; for lower-is-better, p = m1 * (1 - m0).
- fh_lo, fh_hi: Frechet-Hoeffding bounds on the same probability, which need no assumption:
  lo = max(0, P(bad under act) - P(bad under ref)) and hi = min(P(bad under act), P(good under ref)).
- dev_harm_rate: the development-split individual harm rate, the constant reference forecast.
No test outcome is read. Output: predictions_rose_v1/<pair>.parquet, results/rose_v1_manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_three.splits import load  # noqa: E402

PAIRS = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
         "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4"]


def main() -> None:
    out = ROOT / "predictions_rose_v1"
    out.mkdir(exist_ok=True)
    man = {}
    for n in PAIRS:
        p = pd.read_parquet(ROOT / "predictions_rose" / f"{n}.parquet")
        better = load(n, "context")["better"].iat[0]
        m0, m1 = p["m0"].to_numpy(), p["m1"].to_numpy()
        bad_act, bad_ref = (1 - m1, 1 - m0) if better == "higher" else (m1, m0)
        p["p_harm_ind"] = bad_act * (1 - bad_ref)
        p["fh_lo"] = np.maximum(0, bad_act - bad_ref)
        p["fh_hi"] = np.minimum(bad_act, 1 - bad_ref)
        dev = load(n, "dev")
        d = dev[dev["y_act"].isin([0, 1]) & dev["y_ref"].isin([0, 1])]
        harm = (d["y_act"] < d["y_ref"]) if better == "higher" else (d["y_act"] > d["y_ref"])
        f = out / f"{n}.parquet"
        p.to_parquet(f, index=False)
        man[n] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "dev_harm_rate": float(harm.mean()),
                  "better": better, "n": int(len(p))}
        print(n, man[n])
    (ROOT / "results" / "rose_v1_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
