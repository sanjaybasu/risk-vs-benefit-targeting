"""The three rankings that Paper 1's primary hypothesis compares, plus nuisance models.

All models are fit on the context split only and then score any other split's states.

- risk:    predicted undesirable outcome under the reference arm (panel C21). Fit on control-arm
           context rows. Undesirable is Y = 1 if better = "lower"; Y = 0 if better = "higher".
- scale:   constant-relative-effect model (Athey-Keleher-Spiess "simple causal model"; PATH risk
           magnification): logit P(Y | x, a) = alpha + gamma * logit p0(x) + beta * a, fit by
           inverse-design-weighted maximum likelihood on cross-fitted p0. The implied benefit
           depends on x only through p0(x).
- benefit: DR-learner (Kennedy 2023): cross-fitted AIPW pseudo-outcomes with the design
           propensity, regressed on x.

Deployed scores use the average of the K fold models for p0 and p1, the same objects that
produced the cross-fitted scores the scale model was fit on (red-team F62). All boosting models
use fixed, untuned hyperparameters with early stopping off (F63, F64).
Every ranking score is signed so that larger means act first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from system_three.estimators.dr import sign

SEED = 20261112
N_FOLDS = 5
GB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
          l2_regularization=1.0, early_stopping=False, random_state=SEED)


@dataclass
class Fitted:
    p0_folds: list  # K classifiers for P(Y = 1 | x, reference)
    p1_folds: list  # K classifiers for P(Y = 1 | x, action)
    tau: object  # DR-learner regressor
    better: str
    alpha: float
    gamma: float
    beta: float
    beta_se: float
    auc_p0_oof: float  # out-of-fold AUC of p0 on control-arm context rows (F26)
    extra: dict = field(default_factory=dict)


def scale_fit(off: np.ndarray, w: np.ndarray, y: np.ndarray,
              wt: np.ndarray | None = None) -> tuple[float, float, float, float]:
    """Weighted Newton-Raphson for logit P(Y=1) = alpha + gamma * off + beta * w.

    wt are inverse-design weights (1/e for treated, 1/(1-e) for control), so a stratum-varying
    design propensity does not bias beta (F67). Returns alpha, gamma, beta, and the
    sandwich SE of beta.
    """
    wt = np.ones_like(off) if wt is None else wt
    Z = np.column_stack([np.ones_like(off), off, w])
    theta = np.array([0.0, 1.0, 0.0])

    def loglik(th):
        eta = Z @ th
        return float(np.sum(wt * (y * eta - np.logaddexp(0, eta))))

    ll = loglik(theta)
    for _ in range(200):  # Newton with step halving: the log-likelihood never decreases
        p = expit(Z @ theta)
        grad = Z.T @ (wt * (y - p))
        hess = (Z * (wt * p * (1 - p))[:, None]).T @ Z + 1e-8 * np.eye(3)
        step = np.linalg.solve(hess, grad)
        t = 1.0
        while t > 1e-6:
            cand = theta + t * step
            ll_new = loglik(cand)
            if ll_new >= ll - 1e-12:
                break
            t /= 2
        theta, gain, ll = cand, ll_new - ll, ll_new
        if np.max(np.abs(t * step)) < 1e-9 or abs(gain) < 1e-12:
            break
    p = expit(Z @ theta)
    bread = np.linalg.inv((Z * (wt * p * (1 - p))[:, None]).T @ Z + 1e-8 * np.eye(3))
    meat = (Z * (wt * (y - p))[:, None]).T @ (Z * (wt * (y - p))[:, None])
    se_beta = float(np.sqrt((bread @ meat @ bread)[2, 2]))
    return float(theta[0]), float(theta[1]), float(theta[2]), se_beta


def _clf():
    return HistGradientBoostingClassifier(**GB)


def _clip(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def folds_for(n: int, groups: np.ndarray | None):
    """K folds. With groups (cluster ids), every row of a cluster lands in one fold, so a
    household, an email address, or a bootstrap duplicate never spans training and scoring."""
    if groups is None:
        return list(KFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n)))
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    perm = np.random.default_rng(SEED).permutation(len(uniq))
    fold_of_cluster = np.empty(len(uniq), dtype=int)
    fold_of_cluster[perm] = np.arange(len(uniq)) % N_FOLDS
    fold = fold_of_cluster[inv]
    return [(np.flatnonzero(fold != k), np.flatnonzero(fold == k)) for k in range(N_FOLDS)]


def cross_fit_p(X, w, y, groups=None):
    """Out-of-fold p0 and p1 on context, and the K fold models."""
    folds = folds_for(len(y), groups)
    m0, m1 = np.zeros(len(y)), np.zeros(len(y))
    f0, f1 = [], []
    for tr, te in folds:
        for arm, store, keep in ((0, m0, f0), (1, m1, f1)):
            idx = tr[w[tr] == arm]
            mdl = _clf().fit(X[idx], y[idx])
            store[te] = mdl.predict_proba(X[te])[:, 1]
            keep.append(mdl)
    return _clip(m0), _clip(m1), f0, f1


def fit(X: np.ndarray, w: np.ndarray, y: np.ndarray, e: np.ndarray, better: str,
        groups: np.ndarray | None = None) -> Fitted:
    """Fit all models on context data. y binary 0/1; e is P(W = 1) under the design;
    groups are cluster ids for cluster-level cross-fitting."""
    m0, m1, f0, f1 = cross_fit_p(X, w, y, groups)
    wt = np.where(w == 1, 1 / e, 1 / (1 - e))
    alpha, gamma, beta, se = scale_fit(logit(m0), w, y, wt)
    pseudo = m1 - m0 + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
    tau = HistGradientBoostingRegressor(**GB).fit(X, pseudo)
    ctrl = w == 0
    auc = float(roc_auc_score(y[ctrl], m0[ctrl])) if len(np.unique(y[ctrl])) == 2 else float("nan")
    return Fitted(p0_folds=f0, p1_folds=f1, tau=tau, better=better, alpha=alpha, gamma=gamma,
                  beta=beta, beta_se=se, auc_p0_oof=auc)


def _avg(models, X):
    return _clip(np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0))


def scale_benefit(f: Fitted, p0: np.ndarray) -> np.ndarray:
    lin = f.alpha + f.gamma * logit(p0)
    return sign(f.better) * (expit(lin + f.beta) - expit(lin))


def scores(f: Fitted, X: np.ndarray) -> dict[str, np.ndarray]:
    """Ranking scores (larger = act first) and nuisance predictions for new states."""
    p0, p1 = _avg(f.p0_folds, X), _avg(f.p1_folds, X)
    undesirable = p0 if f.better == "lower" else 1 - p0
    return {"risk": undesirable, "scale": scale_benefit(f, p0),
            "benefit": sign(f.better) * f.tau.predict(X), "_m0": p0, "_m1": p1}


def design_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix. When cols is None (context split), drop all-missing and constant columns."""
    if cols is None:
        cols = [c for c in df.columns if c.startswith("x_")]
        cols = [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    return df[cols].to_numpy(dtype=float), cols
