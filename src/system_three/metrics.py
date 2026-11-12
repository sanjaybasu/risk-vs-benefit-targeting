"""Benchmark metrics. Every function takes per-unit scores G (see estimators.dr), signed so larger
is better, and a ranking score s (larger = act first).

Headline: policy value at a fixed budget q, reported as outcomes improved per 1,000 actions,
1000 * mean(G over the top q fraction by s). Ranking quality: AUTOC and Qini forms of the rank-
weighted average treatment effect (RATE; Yadlowsky et al.). Ties in s break by a seeded random
draw so a constant score equals random targeting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from system_three.estimators.bootstrap import cluster_bootstrap

BUDGETS = (0.05, 0.10, 0.20, 0.30, 0.50)


def _order(s: np.ndarray, seed: int) -> np.ndarray:
    jitter = np.random.default_rng(seed).random(len(s))
    return np.lexsort((jitter, -np.asarray(s, float)))


def per_1000(g: np.ndarray, s: np.ndarray, q: float, seed: int = 0) -> float:
    k = max(1, int(round(q * len(g))))
    return 1000.0 * float(np.mean(np.asarray(g, float)[_order(s, seed)[:k]]))


def toc(g: np.ndarray, s: np.ndarray, seed: int = 0) -> np.ndarray:
    g = np.asarray(g, float)[_order(s, seed)]
    return np.cumsum(g) / np.arange(1, len(g) + 1) - g.mean()


def rate(g: np.ndarray, s: np.ndarray, kind: str = "autoc", seed: int = 0) -> float:
    t = toc(g, s, seed)
    if kind == "autoc":
        return float(t.mean())
    if kind == "qini":
        u = np.arange(1, len(t) + 1) / len(t)
        return float(np.mean(u * t))
    raise ValueError(kind)


def evaluate(
    g: np.ndarray,
    scores: dict[str, np.ndarray],
    clusters: np.ndarray,
    budgets: tuple[float, ...] = BUDGETS,
    n_boot: int = 1000,
    seed: int = 20261112,
    contrast: tuple[str, str] | None = None,
) -> pd.DataFrame:
    """Policy value per 1,000 actions and RATE for each ranking, with cluster-bootstrap 95% CIs.

    If `contrast` = (a, b), adds paired rows for the difference a minus b on the same resamples.
    """
    g = np.asarray(g, float)
    rows = []
    names = list(scores) + ([f"{contrast[0]}-minus-{contrast[1]}"] if contrast else [])
    for name in names:
        for metric, q in [("per_1000", q) for q in budgets] + [("rate_autoc", None), ("rate_qini", None)]:
            def stat(idx, name=name, metric=metric, q=q):
                def one(key):
                    if metric == "per_1000":
                        return per_1000(g[idx], scores[key][idx], q, seed)
                    return rate(g[idx], scores[key][idx], metric.split("_")[1], seed)
                if contrast and name == names[-1]:
                    return one(contrast[0]) - one(contrast[1])
                return one(name)

            est, lo, hi = cluster_bootstrap(stat, clusters, n_boot=n_boot, seed=seed)
            rows.append({"ranking": name, "metric": metric, "budget": q, "estimate": est, "lo95": lo, "hi95": hi})
    return pd.DataFrame(rows)


def calibration(g: np.ndarray, tau_hat: np.ndarray, n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """Mean predicted versus mean observed effect by bin of predicted effect, and the BLP slope.

    The BLP slope regresses G on the centered prediction; 1 means well calibrated, 0 means no
    detected heterogeneity signal (Chernozhukov, Demirer, Duflo, Fernandez-Val).
    """
    g, t = np.asarray(g, float), np.asarray(tau_hat, float)
    bins = pd.qcut(pd.Series(t).rank(method="first"), n_bins, labels=False)
    tab = pd.DataFrame({"bin": bins, "pred": t, "obs": g}).groupby("bin").agg(
        pred=("pred", "mean"), obs=("obs", "mean"), n=("obs", "size")
    )
    tc = t - t.mean()
    slope = float(np.dot(tc, g - g.mean()) / np.dot(tc, tc)) if np.dot(tc, tc) > 0 else float("nan")
    return tab.reset_index(), slope


def coverage_exact(truth: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Share of units whose true effect lies in its interval. Valid only where truth is exact."""
    truth = np.asarray(truth, float)
    return float(np.mean((truth >= lo) & (truth <= hi)))


def coverage_by_bin(g: np.ndarray, lo: np.ndarray, hi: np.ndarray, n_bins: int = 10) -> float:
    """Group-level check for randomized data, where unit effects are not identified.

    Bins units by interval midpoint; reports the share of bins whose observed mean G lies within
    the bin's mean interval. Weaker than unit coverage and labeled as such in reports.
    """
    mid = (np.asarray(lo) + np.asarray(hi)) / 2
    bins = pd.qcut(pd.Series(mid).rank(method="first"), n_bins, labels=False)
    df = pd.DataFrame({"b": bins, "g": g, "lo": lo, "hi": hi}).groupby("b").mean()
    return float(np.mean((df["g"] >= df["lo"]) & (df["g"] <= df["hi"])))


def abstention(abstain: np.ndarray, poor_overlap: np.ndarray) -> dict[str, float]:
    """Precision and recall of abstention against known poor-overlap units."""
    a, p = np.asarray(abstain, bool), np.asarray(poor_overlap, bool)
    tp = float(np.sum(a & p))
    return {
        "abstain_rate": float(a.mean()),
        "precision": tp / a.sum() if a.sum() else float("nan"),
        "recall": tp / p.sum() if p.sum() else float("nan"),
    }


def cost_latency(cost_usd: np.ndarray, latency_ms: np.ndarray) -> dict[str, float]:
    c, lat = np.asarray(cost_usd, float), np.asarray(latency_ms, float)
    return {
        "cost_per_1000_decisions_usd": 1000 * float(c.mean()),
        "latency_p50_ms": float(np.percentile(lat, 50)),
        "latency_p95_ms": float(np.percentile(lat, 95)),
    }
