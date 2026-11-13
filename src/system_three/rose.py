"""Rose v0: the reference System Three model (contract v0.2), off-the-shelf components only.

For one dataset, fit on the context split and answer the contract for each test state:
  mean        CATE(x): mean of B bagged DR-learners (cluster bootstrap of context).
  interval_90 mean +/- c * z_0.95 * sd_bag, where c is a conformal factor. Where exact truth exists
              in the development split (Tier A), c is the 90% quantile of |tau - mean| / (z * sd)
              on dev; elsewhere c = 1 and the interval is labeled uncalibrated.
  p_harm      P(CATE(x) in the worse direction | data) ~ Phi(-s * mean / (c * sd)).
  support     1 - percentile of the test point's mean distance to its k nearest context points
              among the same statistic for context points (standardized features); low = no overlap.
  abstain     support < SUPPORT_MIN.
Panel decisions: C33 (no new trained estimator), C37 (CATE semantics for p_harm), C38 (unit
coverage claimed only on exact truth), C39 (abstention via overlap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors

from system_three import rankers
from system_three.estimators.dr import sign

B = 30
K_NN = 20
SUPPORT_MIN = 0.02
Z90 = norm.ppf(0.95)


@dataclass
class RoseFit:
    bags: list
    better: str
    nn: NearestNeighbors
    mu: np.ndarray
    sd: np.ndarray
    ctx_dist: np.ndarray
    c: float = 1.0
    calibrated: bool = False


def fit(X, w, y, e, better, groups, seed=rankers.SEED) -> RoseFit:
    m0, m1, _, _ = rankers.cross_fit_p(X, w, y, groups)
    pseudo = m1 - m0 + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(uniq))]
    rng = np.random.default_rng(seed)
    bags = []
    for b in range(B):
        idx = np.concatenate([members[k] for k in rng.integers(0, len(uniq), len(uniq))])
        bags.append(HistGradientBoostingRegressor(**{**rankers.GB, "random_state": seed + b})
                    .fit(X[idx], pseudo[idx]))
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-9
    Z = np.nan_to_num((X - mu) / sd)
    sub = rng.choice(len(Z), min(len(Z), 20000), replace=False)
    nn = NearestNeighbors(n_neighbors=K_NN).fit(Z[sub])
    ctx_dist = np.sort(nn.kneighbors(Z[sub[:5000]], K_NN + 1)[0][:, 1:].mean(axis=1))
    return RoseFit(bags=bags, better=better, nn=nn, mu=mu, sd=sd, ctx_dist=ctx_dist)


def _ens(f: RoseFit, X):
    P = np.column_stack([b.predict(X) for b in f.bags])
    return P.mean(axis=1), P.std(axis=1, ddof=1) + 1e-9


def calibrate(f: RoseFit, X_dev, tau_dev) -> RoseFit:
    """Conformal factor from development rows with exact truth (Tier A only)."""
    m, s = _ens(f, X_dev)
    f.c = float(np.quantile(np.abs(tau_dev - m) / (Z90 * s), 0.90))
    f.calibrated = True
    return f


def answer(f: RoseFit, X) -> dict:
    m, s = _ens(f, X)
    half = f.c * Z90 * s
    sg = sign(f.better)
    p_harm = norm.cdf(-sg * m / (f.c * s))
    Z = np.nan_to_num((X - f.mu) / f.sd)
    d = f.nn.kneighbors(Z, K_NN)[0].mean(axis=1)
    support = 1.0 - np.searchsorted(f.ctx_dist, d) / len(f.ctx_dist)
    return {"mean": m, "lo90": m - half, "hi90": m + half, "p_harm": p_harm, "support": support,
            "abstain": support < SUPPORT_MIN}
