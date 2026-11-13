"""Rose v2: Rose v0 plus calibration on the development split (registration update 5).

Rose v0 (rose.py) is unchanged. v2 adds three steps, all fit on the development split:
1. CATE calibration (causal isotonic calibration, van der Laan et al., ICML 2023): isotonic
   regression of a target on the v0 mean. Target: doubly robust pseudo-outcomes from
   context-fitted nuisances and the design propensity (trials), or the exact effect tau where
   both potential outcomes are known (exact-truth data). Ranking order is kept: ties left by the
   isotonic fit are broken by the v0 mean.
2. Interval for CATE(x): calibrated mean +/- c * z_0.95 * sd_bag. c is cross-fitted on two halves of
   the development split (isotonic on one half, c on the other, averaged). Trials: c is the smallest value
   for which at least 90% of 20 development bins (by calibrated mean) have their doubly robust bin
   mean inside the bin's mean interval. Exact-truth data: c is the split-conformal 90% quantile of
   |tau - mean| / (z * sd) on development rows (as in v0, around the calibrated mean).
3. Harm: p_harm (CATE semantics, as in the contract) from the calibrated mean and interval scale.
   Individual harm, p_harm_ind: where both potential outcomes are known on development rows,
   isotonic calibration of the conditional-independence estimate (1 - m1) * m0 (or m1 * (1 - m0))
   against observed individual harm; elsewhere the uncalibrated estimate, labeled as such, with
   Frechet-Hoeffding bounds fh_lo, fh_hi, which need no assumption.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression

from system_three import rankers, rose
from system_three.estimators.dr import sign

Z90 = norm.ppf(0.95)
N_BINS = 20


@dataclass
class RoseV2:
    v0: rose.RoseFit
    p0_folds: list
    p1_folds: list
    iso_cate: IsotonicRegression | None = None
    c: float = 1.0
    iso_harm: IsotonicRegression | None = None
    calibrated_on: str = "none"


def fit(X, w, y, e, better, groups) -> RoseV2:
    v0 = rose.fit(X, w, y, e, better, groups)
    _, _, f0, f1 = rankers.cross_fit_p(X, w, y, groups)
    return RoseV2(v0=v0, p0_folds=f0, p1_folds=f1)


def _nuis(f: RoseV2, X):
    return rankers._avg(f.p0_folds, X), rankers._avg(f.p1_folds, X)


def _bin_c(target_mean_by_bin, pred_mean_by_bin, sd_by_bin) -> float:
    r = np.abs(target_mean_by_bin - pred_mean_by_bin) / (Z90 * sd_by_bin)
    return float(np.quantile(r, 0.90, method="higher"))


def calibrate(f: RoseV2, Xd, wd=None, yd=None, ed=None, tau_d=None, yact_d=None, yref_d=None, gd=None) -> RoseV2:
    m, s = rose._ens(f.v0, Xd)
    m0, m1 = _nuis(f, Xd)
    if tau_d is not None:  # exact truth on development rows
        target = np.asarray(tau_d, float)
        f.calibrated_on = "exact"
    else:
        target = m1 - m0 + wd * (yd - m1) / ed - (1 - wd) * (yd - m0) / (1 - ed)
        f.calibrated_on = "pseudo"
    # Cross-fit: isotonic on one half of the development rows, interval factor on the other half,
    # then swap; the two factors are averaged. The final isotonic map uses all development rows.
    gid = np.unique(gd, return_inverse=True)[1] if gd is not None else np.arange(len(m))
    half = (gid % 2).astype(bool)  # halves by cluster, so a household never spans both
    cs = []
    for a_, b_ in ((~half, half), (half, ~half)):
        iso = IsotonicRegression(out_of_bounds="clip").fit(m[a_], target[a_])
        mc_b = iso.predict(m[b_])
        if tau_d is not None:
            cs.append(float(np.quantile(np.abs(target[b_] - mc_b) / (Z90 * s[b_]), 0.90)))
        else:
            n = len(mc_b)
            k = np.minimum((np.argsort(np.argsort(mc_b, kind="stable")) * N_BINS) // n, N_BINS - 1)
            tb = np.array([target[b_][k == j].mean() for j in range(N_BINS)])
            mb = np.array([mc_b[k == j].mean() for j in range(N_BINS)])
            sb = np.array([s[b_][k == j].mean() for j in range(N_BINS)])
            cs.append(_bin_c(tb, mb, sb))
    f.c = float(np.mean(cs))
    f.iso_cate = IsotonicRegression(out_of_bounds="clip").fit(m, target)
    if tau_d is not None:
        better_high = f.v0.better == "higher"
        harm = ((yact_d < yref_d) if better_high else (yact_d > yref_d)).astype(float)
        raw = (1 - m1) * m0 if better_high else m1 * (1 - m0)
        f.iso_harm = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw, harm)
    return f


def answer(f: RoseV2, X) -> dict:
    a = rose.answer(f.v0, X)
    m, s = rose._ens(f.v0, X)
    mc = f.iso_cate.predict(m) if f.iso_cate is not None else m
    mc_rank = mc + 1e-9 * (m - m.mean()) / (m.std() + 1e-12)  # keep v0 order within isotonic ties
    sg = sign(f.v0.better)
    half = f.c * Z90 * s
    m0, m1 = _nuis(f, X)
    high = f.v0.better == "higher"
    bad_act, bad_ref = (1 - m1, 1 - m0) if high else (m1, m0)
    raw = bad_act * (1 - bad_ref)
    return {"mean": mc_rank, "lo90": mc - half, "hi90": mc + half,
            "p_harm": norm.cdf(-sg * mc / (f.c * s)),
            "p_harm_ind": f.iso_harm.predict(raw) if f.iso_harm is not None else raw,
            "p_harm_ind_calibrated": np.full(len(m), f.iso_harm is not None),
            "fh_lo": np.maximum(0, bad_act - bad_ref), "fh_hi": np.minimum(bad_act, 1 - bad_ref),
            "support": a["support"], "abstain": a["abstain"]}
