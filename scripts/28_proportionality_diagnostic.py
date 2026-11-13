"""Context-only proportionality diagnostic for the scale forecast (exploratory on wave 1; to be
registered for wave 2 before any wave-2 data are loaded).

The scale model assumes a constant odds ratio: logit P = a + g*off + b*w, off = logit(cross-fitted p0).
The diagnostic adds the interaction d * w * (off - mean(off)) and tests d = 0 with a cluster-robust
sandwich SE, using context data only (inverse-design weights as in rankers.scale_fit). It also
reports the forecast gap recomputed under the interaction model, so the diagnostic can say how much
the forecast would change.

Flag rule (proposed): the forecast is "not applicable" when the 95% CI of d excludes 0 AND the
interaction-model forecast differs in sign from the scale forecast or by more than 50% of it.
Output: results/proportionality_diagnostic.json
Usage: uv run python scripts/28_proportionality_diagnostic.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from system_three import rankers  # noqa: E402
from system_three.analysis import prep  # noqa: E402
from system_three.estimators.dr import sign  # noqa: E402

TRIALS = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
          "sixcity", "newhaven", "lenta", "x5"]
TIER_A = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
          "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4"]


def glm(Z, y, wt, groups, iters=100):
    th = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = expit(Z @ th)
        g = Z.T @ (wt * (y - p))
        H = (Z * (wt * p * (1 - p))[:, None]).T @ Z + 1e-8 * np.eye(Z.shape[1])
        step = np.linalg.solve(H, g)
        th = th + step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = expit(Z @ th)
    bread = np.linalg.inv((Z * (wt * p * (1 - p))[:, None]).T @ Z + 1e-8 * np.eye(Z.shape[1]))
    u = Z * (wt * (y - p))[:, None]
    _, inv = np.unique(groups, return_inverse=True)
    U = np.zeros((inv.max() + 1, Z.shape[1]))
    np.add.at(U, inv, u)
    V = bread @ (U.T @ U) @ bread
    return th, np.sqrt(np.diag(V))


def gap(ben, risk, q=0.2):
    jitter = np.random.default_rng(rankers.SEED).random(len(ben))
    k = max(1, int(round(q * len(ben))))
    top = lambda s: float(ben[np.lexsort((jitter, -s))[:k]].mean())  # noqa: E731
    return 1000 * (top(ben) - top(risk))


def diagnose(X, w, y, e, groups, better) -> dict:
    m0, _, _, _ = rankers.cross_fit_p(X, w, y, groups)
    off = logit(m0)
    c = off - off.mean()
    wt = np.where(w == 1, 1 / e, 1 / (1 - e))
    s = sign(better)
    risk = m0 if better == "lower" else 1 - m0
    th0, _ = glm(np.column_stack([np.ones_like(off), off, w]), y, wt, groups)
    lin0 = th0[0] + th0[1] * off
    ben0 = s * (expit(lin0 + th0[2]) - expit(lin0))
    th1, se1 = glm(np.column_stack([np.ones_like(off), off, w, w * c]), y, wt, groups)
    lin1 = th1[0] + th1[1] * off
    ben1 = s * (expit(lin1 + th1[2] + th1[3] * c) - expit(lin1))
    d, sd = float(th1[3]), float(se1[3])
    g0, g1 = gap(ben0, risk), gap(ben1, risk)
    excl = abs(d / sd) > stats.norm.ppf(0.975)
    changes = (np.sign(g0) != np.sign(g1) and max(abs(g0), abs(g1)) > 1.0) or abs(g1 - g0) > 0.5 * max(abs(g0), 1.0)
    return {"delta": d, "delta_se": sd, "delta_lo95": d - 1.96 * sd, "delta_hi95": d + 1.96 * sd,
            "gap_scale": g0, "gap_interaction": g1, "flag_not_applicable": bool(excl and changes)}


def tier_a(name):
    spec = importlib.util.spec_from_file_location("ta", ROOT / "scripts" / "16_tierA_predictions.py")
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    from system_three.splits import load

    ctx = load(name, "context")
    sc = ta.parse(ctx["state"])
    ok = set(ctx.groupby(sc["eval"].to_numpy())["outcome"].apply(lambda v: v.isin([0, 1]).all()).loc[lambda b: b].index)
    k = sc["eval"].isin(ok).to_numpy()
    ctx, sc = ctx[k].reset_index(drop=True), sc[k].reset_index(drop=True)
    Xc, _ = ta.features(sc, sc, "routerbench")
    w = (ctx["action"] != ctx["reference"]).to_numpy(int)
    return Xc, w, ctx["outcome"].to_numpy(), np.full(len(w), 0.5), ctx["cluster_id"].to_numpy(), ctx["better"].iat[0]


def main() -> None:
    out = {}
    for t in TRIALS:
        b = prep(t, "context")
        X, _ = rankers.design_matrix(b)
        out[t] = diagnose(X, b["W"].to_numpy(), b["outcome"].to_numpy(), b["e"].to_numpy(),
                          b["cluster_id"].to_numpy(), b["better"].iat[0])
        print(t, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[t].items()}, flush=True)
    for n in TIER_A:
        out[n] = diagnose(*tier_a(n))
        print(n, {k: round(v, 3) if isinstance(v, float) else v for k, v in out[n].items()}, flush=True)
    (ROOT / "results" / "proportionality_diagnostic.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
