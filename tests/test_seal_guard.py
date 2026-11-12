"""Test outcomes are sealed: no development code path can read them."""

import re
from pathlib import Path

import pytest

from system_three import splits

ROOT = Path(__file__).resolve().parents[1]
SEAL_WRITER = "03_validate_and_seal.py"


def test_load_test_outcomes_raises():
    with pytest.raises(splits.SealedSplitError):
        splits.load_test_outcomes("camden")


def test_only_the_seal_writer_names_the_sealed_folder():
    pat = re.compile(r"system-three-sealed")
    offenders = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if p.name != SEAL_WRITER and pat.search(p.read_text()):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, offenders


def test_scorer_recovers_known_gap(tmp_path):
    """Frozen scorer on synthetic sealed data: benefit beats risk when they differ."""
    import importlib.util
    import json

    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    risk = 1 / (1 + np.exp(-(x - 0.5)))
    tau = 0.25 * np.exp(-(x**2))
    w = rng.binomial(1, 0.5, n)
    y0 = rng.binomial(1, risk)
    y = np.where(w == 1, np.maximum(y0 - rng.binomial(1, np.minimum(tau / risk, 1)), 0), y0)
    ids = [f"u{i}" for i in range(n)]
    sealed = pd.DataFrame({"row_id": ids, "cluster_id": ids, "action": np.where(w == 1, "act", "ref"),
                           "outcome": y, "propensity": 0.5})
    pred = pd.DataFrame({"row_id": ids, "risk": risk, "scale": risk * (1 - risk), "benefit": tau,
                         "m0": risk, "m1": np.clip(risk - tau, 0, 1)})
    meta = {"action": "act", "reference": "ref", "better": "lower"}
    spec = importlib.util.spec_from_file_location("s", ROOT / "scripts" / "score_sealed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.N_BOOT = 300
    res = mod.score_trial(sealed, pred, meta)
    row = [r for r in res["table"] if r["ranking"] == "benefit-minus-risk" and r["budget"] == 0.2][0]
    assert row["lo95"] > 0, row
    import pytest as _pt
    with _pt.raises(ValueError):  # predictions may not carry sealed-side columns
        mod.score_trial(sealed, pred.assign(cluster_id=ids), meta)
    with _pt.raises(ValueError):  # every sealed row needs a prediction
        mod.score_trial(sealed, pred.iloc[:-1], meta)
    sub = mod.score_trial(sealed, pred.iloc[:1000], meta, subset=True)  # explicit subset mode
    assert sub["n"] == 1000


def test_split_hash_is_stable_and_cluster_level():
    a = splits.hash_unit("camden", "25264")
    assert a == splits.hash_unit("camden", "25264")
    assert 0 <= a < 1
