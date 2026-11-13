"""Wave-2 forecasts, posted before scoring (registration PREREGISTRATION_WAVE2.md, Sections 4-5).

For each wave-2 trial: the scale-predicted gap and its SE, the sign-rule status, the diagnostic flag,
the development-split primary variance, the forecast of the observed gap from the wave-1 H1 fit
(frozen 07 functions, the 13 wave-1 trials), and the SHA-256 of every test-split prediction file.
Output: results/w2_forecasts.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)
W2 = ["w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
      "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
SETS = ["predictions", "predictions_causalpfn", "predictions_rose_v2"]


def main() -> None:
    post = mr.posted()
    w1 = [t for t in mr.confirmatory(post) if not t.startswith("w2_")]
    obs = mr.observed(sorted(p for p in (ROOT / "scoring").glob("stage*/predictions__*.json")))
    w1 = [t for t in w1 if t in obs]
    x = np.array([post[t]["x"] for t in w1])
    y = np.array([obs[t]["scale_minus_risk"] for t in w1])
    v = np.array([post[t]["v_primary"] for t in w1])
    f = mr.fit_re(y, np.column_stack([np.ones_like(x), x]), v)
    diag = json.loads((ROOT / "results" / "proportionality_diagnostic_w2.json").read_text())
    out = {"wave1_fit": {"trials": w1, **mr.slope_test(f)}, "trials": {}}
    c = stats.t.ppf(0.975, f.df)
    for t in W2:
        r = post[t]
        mu, var = mr.predict(f, np.array([1.0, r["x"]]), r["v_primary"])
        out["trials"][t] = {
            "scale_predicted_gap": r["x"], "gap_se": r["x_se"], "beneficial_sign": r["beneficial"],
            "in_H1_W2": bool(r["beneficial"]), "diagnostic_flag": diag[t]["flag_not_applicable"],
            "v_primary": r["v_primary"], "wave1_forecast": {"mean": mu, "pi95": [mu - c * np.sqrt(var), mu + c * np.sqrt(var)]},
            "prediction_sha256": {s: hashlib.sha256((ROOT / s / f"{t}.parquet").read_bytes()).hexdigest() for s in SETS}}
        print(t, round(r["x"], 2), r["beneficial"], diag[t]["flag_not_applicable"], round(mu, 1))
    out["n_in_H1_W2"] = sum(v["in_H1_W2"] for v in out["trials"].values())
    (ROOT / "results" / "w2_forecasts.json").write_text(json.dumps(out, indent=1))
    print("in H1-W2:", out["n_in_H1_W2"])


if __name__ == "__main__":
    main()
