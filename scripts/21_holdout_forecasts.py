"""Holdout forecasts from the stage 1 fit (registration F01; update 3, stage order).

Fits the frozen H1 random-effects model (scripts/07_meta_regression.py, imported unchanged) on the
non-holdout confirmatory trials scored in stage 1, and the intercept-only model on the same trials.
For each holdout it writes the predictive mean, variance, df and 95% interval under both models.
This file is posted to OSF before stage 2 opens the holdouts.

Usage: uv run python scripts/21_holdout_forecasts.py --scored scoring/stage1/predictions__*.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr  # dataclasses need the module registered
spec.loader.exec_module(mr)
HOLD = json.loads((ROOT / "results" / "holdouts.json").read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", nargs="+", type=Path)
    a = ap.parse_args()
    hold = HOLD["holdouts"] if isinstance(HOLD, dict) else HOLD
    post, obs = mr.posted(), mr.observed(a.scored)
    fit_t = [t for t in mr.confirmatory(post) if t in obs and t not in hold]
    if set(hold) & set(obs):
        raise SystemExit("a holdout is already scored; forecasts must come first")
    x = np.array([post[t]["x"] for t in fit_t])
    v = np.array([post[t]["v_primary"] for t in fit_t])
    y = np.array([obs[t]["scale_minus_risk"] for t in fit_t])
    X = np.column_stack([np.ones_like(x), x])
    f1, f0 = mr.fit_re(y, X, v), mr.fit_re(y, X[:, :1], v)
    out = {"utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "fit_trials": fit_t,
           "fit_h1": mr.slope_test(f1), "fit_null_mean": float(f0.beta[0]), "fit_null_tau2": f0.tau2,
           "units": "outcomes improved per 1,000 actions, scale minus risk, 20% budget", "forecasts": {}}
    for t in hold:
        xi = np.array([1.0, post[t]["x"]])
        rec = {"moderator_x": post[t]["x"], "v_primary": post[t]["v_primary"]}
        for name, f, xv in (("h1", f1, xi), ("null", f0, xi[:1])):
            mu, var = mr.predict(f, xv, post[t]["v_primary"])
            c = stats.t.ppf(0.975, f.df)
            rec[name] = {"mean": mu, "var": var, "df": f.df, "pi95": [mu - c * np.sqrt(var), mu + c * np.sqrt(var)]}
        out["forecasts"][t] = rec
    p = ROOT / "results" / "holdout_forecasts.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
