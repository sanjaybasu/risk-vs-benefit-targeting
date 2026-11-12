"""Shared helpers for dataset loaders.

Each loader module defines INFO (a DatasetInfo), PUBLISHED (the aggregate to reproduce before
the seal), and load() -> pd.DataFrame in the common table format.

PUBLISHED = {"source": citation, "arm": action, "metric": "mean_ref" | "mean_act" | "diff",
             "value": number in outcome units, "tol": absolute tolerance}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"


def frame(
    name: str,
    x: pd.DataFrame,
    action: pd.Series,
    outcome: pd.Series,
    info,
    propensity: pd.Series | float,
    cluster: pd.Series | None = None,
    stratum: pd.Series | None = None,
    row_id: pd.Series | None = None,
    state_text: pd.Series | None = None,
    y_ref: pd.Series | None = None,
    y_act: pd.Series | None = None,
) -> pd.DataFrame:
    n = len(x)
    rid = row_id if row_id is not None else pd.Series(np.arange(n), index=x.index)
    df = pd.DataFrame(
        {
            "dataset": name,
            "row_id": rid.astype(str).to_numpy(),
            "cluster_id": (cluster if cluster is not None else rid).astype(str).to_numpy(),
            "stratum": (stratum.astype(str).to_numpy() if stratum is not None else "all"),
            "state": state_text.to_numpy() if state_text is not None else "",
            "action": action.to_numpy(),
            "reference": info.reference,
            "outcome": outcome.astype(float).to_numpy(),
            "outcome_name": info.outcome_name,
            "better": info.better,
            "propensity": (propensity.to_numpy() if isinstance(propensity, pd.Series) else propensity),
            "y_ref": y_ref.to_numpy() if y_ref is not None else np.nan,
            "y_act": y_act.to_numpy() if y_act is not None else np.nan,
            "truth_exact": y_ref is not None,
        }
    )
    if len(x.columns):
        feats = pd.DataFrame(
            {f"x_{c}": pd.to_numeric(x[c], errors="coerce").to_numpy() for c in x.columns}
        )
        df = pd.concat([df, feats], axis=1)
    return df
