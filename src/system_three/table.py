"""Common table that every loader returns.

One row per unit of randomization or decision. `state` is a JSON string of fields measured
before the action. `action` is the action taken, as a string; `reference` names the
reference action for the dataset. `outcome` is numeric. `propensity` is P(action taken |
state) under the logging design; for randomized data it is the design probability.

Where both potential outcomes are observed (routing benchmarks) or known (IHDP), the columns
`y_ref` and `y_act` hold them and `truth_exact` is True; the harness then scores exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUMNS = [
    "dataset", "row_id", "cluster_id", "stratum", "state", "action", "reference",
    "outcome", "outcome_name", "better", "propensity", "y_ref", "y_act", "truth_exact",
]
# Numeric pre-assignment features may also be stored as columns named x_<field>. `state` then
# holds only text fields (or ""), and state_json() rebuilds the full JSON state on demand.
FEATURE_PREFIX = "x_"


AUX_PREFIX = "aux_"  # post-action quantities kept for scoring only (e.g., cost per call)


def features(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIX)]


def aux(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(AUX_PREFIX)]


def state_json(df: pd.DataFrame) -> pd.Series:
    import json

    feats = features(df)
    out = []
    for i, row in enumerate(df[feats].itertuples(index=False)):
        d = {k[len(FEATURE_PREFIX):]: (None if pd.isna(v) else v) for k, v in zip(feats, row)}
        s = df["state"].iat[i]
        if s:
            d.update(json.loads(s) if s.startswith("{") else {"text": s})
        out.append(json.dumps(d, default=float))
    return pd.Series(out, index=df.index)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    tier: str  # "A", "B", or "C"
    outcome_name: str
    better: str  # "higher" or "lower"
    reference: str
    actions: tuple[str, ...]
    license: str
    commercial_use: bool
    citation: str
    risk_field: str | None = None  # pre-existing risk score shipped with the data, if any


def validate(df: pd.DataFrame, info: DatasetInfo) -> pd.DataFrame:
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{info.name}: missing columns {missing}")
    extra = [c for c in ("native_split",) if c in df.columns]
    df = df[COLUMNS + features(df) + aux(df) + extra].copy()
    if df["row_id"].duplicated().any():
        raise ValueError(f"{info.name}: duplicate row_id")
    df["state"] = df["state"].fillna("")
    if not features(df) and (df["state"] == "").any():
        raise ValueError(f"{info.name}: empty state and no x_ features")
    if not set(df["action"].unique()) <= set(info.actions):
        raise ValueError(f"{info.name}: actions outside {info.actions}")
    if (df["reference"] != info.reference).any():
        raise ValueError(f"{info.name}: reference column must equal {info.reference}")
    if df["better"].ne(info.better).any():
        raise ValueError(f"{info.name}: better column must equal {info.better}")
    exact = df["truth_exact"].astype(bool)
    if exact.any():
        if df.loc[exact, ["y_ref", "y_act"]].isna().any().any():
            raise ValueError(f"{info.name}: exact rows need y_ref and y_act")
    obs = ~exact
    if obs.any():
        p = df.loc[obs, "propensity"]
        if p.isna().any() or not ((p > 0) & (p < 1)).all():
            raise ValueError(f"{info.name}: propensity must lie in (0, 1)")
        if df.loc[obs, "outcome"].isna().any():
            raise ValueError(f"{info.name}: null outcome")
    df["truth_exact"] = exact
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["stratum"] = df["stratum"].astype(str)
    for c in ("outcome", "propensity", "y_ref", "y_act"):
        df[c] = pd.to_numeric(df[c], errors="raise").astype(float)
    return df.reset_index(drop=True)


def treated(df: pd.DataFrame) -> np.ndarray:
    """1 if the row took a non-reference action, else 0 (binary-action datasets)."""
    return (df["action"] != df["reference"]).to_numpy(dtype=int)
