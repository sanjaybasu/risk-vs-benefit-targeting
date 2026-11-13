"""Exploratory (not registered) H1 sensitivity, decided after scoring and labeled as such.

GUSTO-I and the synbiotic trial have a scale-predicted gap of exactly 0 and an observed gap of
exactly 0: the scale ranking orders people as the risk ranking does. With the registered variance
floor they get the largest weights and fix the intercept near 0. This refits H1 without them,
with the frozen 07 functions. Output: results/h1_exploratory_sensitivity.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)

post = mr.posted()
obs = mr.observed(sorted((ROOT / "scoring").glob("stage*/predictions__*.json")))
out = {}
for name, drop in (("without_gusto_synbiotic", {"gusto", "synbiotic"}),):
    tt = [t for t in mr.confirmatory(post) if t in obs and t not in drop]
    x = np.array([post[t]["x"] for t in tt])
    y = np.array([obs[t]["scale_minus_risk"] for t in tt])
    v = np.array([post[t]["v_primary"] for t in tt])
    f = mr.fit_re(y, np.column_stack([np.ones_like(x), x]), v)
    out[name] = {"trials": tt, **mr.slope_test(f), "joint_0_1": mr.joint_test(f)}
(ROOT / "results" / "h1_exploratory_sensitivity.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
