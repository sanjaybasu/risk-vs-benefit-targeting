"""The harness must recover known truth on synthetic randomized data.

Pass bar (pre-stated in the Phase 1 plan): across 200 replications, the 95% cluster-bootstrap
interval for policy value per 1,000 actions covers the in-sample exact value in 85% to 99% of
replications, under equal propensity, unequal propensity, and clustered assignment.
"""

import numpy as np
import pytest

from system_three.estimators import aipw_scores, cluster_bootstrap, exact_scores, ipw_scores
from system_three.metrics import per_1000, rate

REPS, N, Q = 200, 2000, 0.2


def simulate(rng, e, cluster_size=1):
    n_c = N // cluster_size
    cl = np.repeat(np.arange(n_c), cluster_size)
    x = rng.normal(size=N) + np.repeat(rng.normal(size=n_c), cluster_size) * (cluster_size > 1)
    risk = 1 / (1 + np.exp(-(x - 0.5)))  # baseline event probability rises with x
    tau = 0.25 * np.exp(-((x - 0.0) ** 2))  # benefit peaks at moderate x, not at highest risk
    y0 = rng.binomial(1, risk)
    y1 = np.maximum(y0 - rng.binomial(1, np.minimum(tau / np.maximum(risk, 1e-3), 1)), 0)
    w_c = rng.binomial(1, e, n_c)
    w = np.repeat(w_c, cluster_size)
    y = np.where(w == 1, y1, y0)
    return x, cl, risk, tau, y0, y1, w, y


@pytest.mark.parametrize("e,cluster_size", [(0.5, 1), (0.2, 1), (0.5, 4)])
def test_policy_value_ci_covers_exact(e, cluster_size):
    rng = np.random.default_rng(7)
    hits = 0
    for r in range(REPS):
        x, cl, risk, tau, y0, y1, w, y = simulate(rng, e, cluster_size)
        s = tau  # oracle ranking, fixed
        g_exact = exact_scores(y1, y0, "lower")
        truth = per_1000(g_exact, s, Q)
        g = ipw_scores(y, w, np.full(N, e), "lower")
        _, lo, hi = cluster_bootstrap(lambda i: per_1000(g[i], s[i], Q), cl, n_boot=300, seed=r)
        hits += lo <= truth <= hi
    cov = hits / REPS
    assert 0.85 <= cov <= 0.99, cov


def test_aipw_matches_ipw_in_expectation_and_is_tighter():
    rng = np.random.default_rng(11)
    ipw, dr = [], []
    for _ in range(300):
        x, cl, risk, tau, y0, y1, w, y = simulate(rng, 0.5)
        m0, m1 = risk, np.clip(risk - tau, 0, 1)  # good nuisance models
        ipw.append(np.mean(ipw_scores(y, w, np.full(N, 0.5), "lower")))
        dr.append(np.mean(aipw_scores(y, w, np.full(N, 0.5), m0, m1, "lower")))
    target = np.mean(0.25 * np.exp(-(rng.normal(size=200000) ** 2)))  # approx E[tau]
    assert abs(np.mean(dr) - np.mean(ipw)) < 0.01
    assert np.std(dr) < np.std(ipw)
    assert abs(np.mean(dr) - target) < 0.02


def test_benefit_ranking_beats_risk_ranking_when_they_differ():
    rng = np.random.default_rng(3)
    x, cl, risk, tau, y0, y1, w, y = simulate(rng, 0.5)
    g = exact_scores(y1, y0, "lower")
    benefit = per_1000(g, tau, Q)
    by_risk = per_1000(g, risk, Q)
    rnd = per_1000(g, np.zeros(N), Q, seed=1)
    assert benefit > by_risk
    assert benefit > rnd
    assert rate(g, tau) > rate(g, risk)


def test_constant_score_equals_random():
    rng = np.random.default_rng(5)
    g = rng.normal(size=10000)
    assert abs(per_1000(g, np.zeros_like(g), 0.3, seed=2) - 1000 * g.mean()) < 40
