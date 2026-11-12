"""Checks behind statements in the pre-registration (Section 6, implementation bullet)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("meta", ROOT / "scripts" / "07_meta_regression.py")
meta = importlib.util.module_from_spec(spec)
sys.modules["meta"] = meta
spec.loader.exec_module(meta)


def test_equals_weighted_least_squares_without_heterogeneity():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 12)
    X = np.column_stack([np.ones(12), x])
    y = 2 + 0.5 * x + rng.normal(0, 1, 12)
    f = meta.fit_re(y, X, np.ones(12))
    assert np.allclose(f.beta, np.linalg.lstsq(X, y, rcond=None)[0])


def test_reml_tau2_unbiased():
    est = []
    for seed in range(40):
        rng = np.random.default_rng(seed)
        x = rng.uniform(0, 10, 400)
        X = np.column_stack([np.ones(400), x])
        v = rng.uniform(0.5, 2, 400)
        y = 1 + 0.3 * x + rng.normal(0, 2, 400) + rng.normal(0, np.sqrt(v))
        est.append(meta.fit_re(y, X, v).tau2)
    assert abs(np.mean(est) - 4.0) < 0.2, np.mean(est)  # observed mean 4.05 for true 4
