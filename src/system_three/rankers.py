"""The three rankings that Paper 1's primary hypothesis compares, plus nuisance models.

All models are fit on the context split only and then score any other split's states.

- risk:    predicted undesirable outcome under the reference arm (panel C21). Fit on control-arm
           context rows. Undesirable is Y = 1 if better = "lower"; Y = 0 if better = "higher".
- scale:   constant-relative-effect model (Athey-Keleher-Spiess "simple causal model"; PATH risk
           magnification). logit P(Y | x, a) = logit p0(x) + beta * a, with p0 cross-fitted, so the
           implied benefit depends on x only through p0(x).
- benefit: DR-learner (Kennedy 2023): cross-fitted AIPW pseudo-outcomes with the known design
           propensity, regressed on x.
Every ranking score is signed so that larger means act first.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from system_three.estimators.dr import sign

SEED = 20261112
GB = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
          l2_regularization=1.0, random_state=SEED)


@dataclass
class Fitted:
    p0: object  # classifier for P(Y = 1 | x, reference)
    p1: object  # classifier for P(Y = 1 | x, action)
    beta: float  # scale model log-odds shift
    tau: object  # DR-learner regressor
    better: str
    alpha: float = 0.0  # recalibration intercept for logit p0
    gamma: float = 1.0  # recalibration slope for logit p0


def scale_fit(off: np.ndarray, w: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Fit logit P(Y=1) = alpha + gamma * off + beta * w by Newton-Raphson (unpenalized).

    alpha and gamma recalibrate the cross-fitted risk score, so beta carries only the
    constant relative effect of the action.
    """
    Z = np.column_stack([np.ones_like(off), off, w])
    theta = np.array([0.0, 1.0, 0.0])
    for _ in range(100):
        p = expit(Z @ theta)
        grad = Z.T @ (y - p)
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + 1e-8 * np.eye(3)
        step = np.linalg.solve(hess, grad)
        theta += step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(theta[0]), float(theta[1]), float(theta[2])


def _clf():
    return HistGradientBoostingClassifier(**GB)


def _clip(p):
    return np.clip(p, 1e-4, 1 - 1e-4)


def fit(X: np.ndarray, w: np.ndarray, y: np.ndarray, e: np.ndarray, better: str,
        n_folds: int = 5) -> Fitted:
    """Fit all models on context data. y must be binary 0/1."""
    folds = list(KFold(n_folds, shuffle=True, random_state=SEED).split(X))
    m0 = np.zeros(len(y))
    m1 = np.zeros(len(y))
    for tr, te in folds:
        for arm, store in ((0, m0), (1, m1)):
            idx = tr[w[tr] == arm]
            store[te] = _clf().fit(X[idx], y[idx]).predict_proba(X[te])[:, 1]
    m0, m1 = _clip(m0), _clip(m1)
    # Scale model: recalibrated cross-fitted logit p0 plus one log-odds shift for the action.
    alpha, gamma, beta = scale_fit(logit(m0), w, y)
    pseudo = m1 - m0 + w * (y - m1) / e - (1 - w) * (y - m0) / (1 - e)
    tau = HistGradientBoostingRegressor(**GB).fit(X, pseudo)
    p0 = _clf().fit(X[w == 0], y[w == 0])
    p1 = _clf().fit(X[w == 1], y[w == 1])
    return Fitted(p0=p0, p1=p1, beta=beta, tau=tau, better=better, alpha=alpha, gamma=gamma)


def scores(f: Fitted, X: np.ndarray) -> dict[str, np.ndarray]:
    """Ranking scores (larger = act first) and nuisance predictions for new states."""
    s = sign(f.better)
    p0 = _clip(f.p0.predict_proba(X)[:, 1])
    p1 = _clip(f.p1.predict_proba(X)[:, 1])
    undesirable = p0 if f.better == "lower" else 1 - p0
    lin = f.alpha + f.gamma * logit(p0)
    scale = s * (expit(lin + f.beta) - expit(lin))
    benefit = s * f.tau.predict(X)
    return {"risk": undesirable, "scale": scale, "benefit": benefit, "_m0": p0, "_m1": p1}


def design_matrix(df: pd.DataFrame, cols: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature matrix. When cols is None (context split), drop all-missing and constant columns."""
    if cols is None:
        cols = [c for c in df.columns if c.startswith("x_")]
        cols = [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    return df[cols].to_numpy(dtype=float), cols
