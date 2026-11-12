"""Frozen analysis configuration shared by the forecast, prediction, and seal scripts (F76, F58).

Holds the confirmatory trial list, the pre-specified contrast arm for multi-arm trials, and the
loader of a binarized split. Scripts import from here so that no moderator or prediction depends
on unfrozen script code.
"""

from __future__ import annotations

import json

import pandas as pd

from system_three.estimators.dr import binarize
from system_three.splits import PROCESSED, load

SEED = 20261112
# Pre-specified contrast arm for multi-arm trials (the published headline arm).
ARM = {"voter": "neighbors", "hillstrom": "mens_email", "sf_email": "registrar_email"}
ALL_BINARY = ["gusto", "ist", "synbiotic", "elaia1", "elaia2", "voter", "sf_email", "karlan_list",
              "stepup", "oregon", "hillstrom", "sixcity", "newhaven", "lenta", "x5"]
# Oregon is outside the confirmatory set (F07): the published ITT shows the action raises ED use.
CANDIDATES = [t for t in ALL_BINARY if t != "oregon"]
CAP = {"criteo": (1_000_000, 500_000)}


def contrast_arm(name: str, df: pd.DataFrame) -> str:
    ref = df["reference"].iat[0]
    return ARM.get(name) or next(a for a in sorted(df["action"].unique()) if a != ref)


def prep(name: str, split: str) -> pd.DataFrame:
    """Binarized split (contrast arm vs reference) with W and design e columns."""
    df = load(name, split)
    attrs_path = PROCESSED / name / "attrs.json"
    if attrs_path.exists():
        df.attrs.update(json.loads(attrs_path.read_text()))
    if split == "test":
        raise ValueError("test split has no actions; use prep_states for test")
    b = binarize(df, contrast_arm(name, df))
    if name in CAP:
        n = CAP[name][0 if split == "context" else 1]
        b = b.sample(n=min(n, len(b)), random_state=SEED)
    return b.reset_index(drop=True)
