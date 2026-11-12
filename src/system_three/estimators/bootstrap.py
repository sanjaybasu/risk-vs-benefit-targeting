"""Cluster bootstrap: resample whole clusters (households, prompts, tests) with replacement."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    clusters: np.ndarray,
    n_boot: int = 1000,
    seed: int = 20261112,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return (estimate, low, high) for stat(indices) with a percentile cluster bootstrap.

    `stat` takes an array of row indices and returns a scalar.
    """
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    est = float(stat(np.arange(len(clusters))))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[k] for k in pick])
        draws[b] = stat(idx)
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return est, float(lo), float(hi)
