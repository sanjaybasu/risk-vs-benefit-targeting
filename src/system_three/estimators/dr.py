"""Per-unit scores for evaluating a targeting rule on randomized or exact-truth data.

Binary comparison: one non-reference action against the reference. W = 1 if the unit took the
action, e = P(W = 1 | state) under the design, Y the observed outcome.

    IPW score       G_i = W Y / e - (1 - W) Y / (1 - e)
    AIPW (DR) score G_i = m1 - m0 + W (Y - m1) / e - (1 - W) (Y - m0) / (1 - e)

The mean of G over any set chosen from the state alone is unbiased for the average effect in
that set when e is the true design propensity, whatever m0 and m1 are; good m0, m1 only reduce
variance. m0, m1 must be fit on rows that are not scored (the context split).

Exact truth (both potential outcomes known): G_i = y_act - y_ref.

All scores are signed so that larger means better (`better` = "lower" flips the sign).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sign(better: str) -> float:
    if better not in ("higher", "lower"):
        raise ValueError(better)
    return 1.0 if better == "higher" else -1.0


def binarize(df: pd.DataFrame, action: str) -> pd.DataFrame:
    """Keep rows that took `action` or the reference; renormalize the propensity to the pair.

    For multi-arm trials the per-row `propensity` is P(arm taken). With design probabilities
    p_a and p_ref, P(W = 1 | in pair) = p_a / (p_a + p_ref). `arm_probs` must be supplied by the
    loader in df.attrs["arm_probs"] as {action: probability} (constant or per-stratum dict).
    """
    ref = df["reference"].iat[0]
    sub = df[df["action"].isin([action, ref])].copy()
    probs = df.attrs.get("arm_probs")
    if probs is None:
        if df["action"].nunique() > 2:
            raise ValueError("multi-arm data need df.attrs['arm_probs']")
        e = np.where(sub["action"] == action, sub["propensity"], 1 - sub["propensity"])
    else:
        pa = sub["stratum"].map(lambda s: _p(probs, action, s))
        pr = sub["stratum"].map(lambda s: _p(probs, ref, s))
        e = (pa / (pa + pr)).to_numpy()
    sub["W"] = (sub["action"] == action).astype(int)
    sub["e"] = e
    return sub


def _p(probs: dict, action: str, stratum: str) -> float:
    v = probs[action]
    return float(v[stratum] if isinstance(v, dict) else v)


def ipw_scores(y: np.ndarray, w: np.ndarray, e: np.ndarray, better: str) -> np.ndarray:
    _check(w, e)
    return sign(better) * (w * y / e - (1 - w) * y / (1 - e))


def aipw_scores(
    y: np.ndarray, w: np.ndarray, e: np.ndarray, m0: np.ndarray, m1: np.ndarray, better: str
) -> np.ndarray:
    _check(w, e)
    g = m1 - m0 + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
    return sign(better) * g


def exact_scores(y_act: np.ndarray, y_ref: np.ndarray, better: str) -> np.ndarray:
    return sign(better) * (np.asarray(y_act, float) - np.asarray(y_ref, float))


def _check(w: np.ndarray, e: np.ndarray) -> None:
    e = np.asarray(e, float)
    if not np.all((e > 0) & (e < 1)):
        raise ValueError("propensity must lie strictly in (0, 1)")
    if not set(np.unique(w)) <= {0, 1}:
        raise ValueError("w must be 0/1")
