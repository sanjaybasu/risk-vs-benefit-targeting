"""Additional post-unsealing exploratory analyses addressing co-author review (Ben Huynh, 2026-11-23).

Strictly exploratory (run after sealed test outcomes were scored). Never modifies any pre-registered
prediction file or results/final_results.json. Evaluates every comparator using the exact same
sealed test splits and exact same AIPW scores `g` (built from the hashed context-split nuisances)
as score_sealed.py.

Computes and saves results/ben_additional_exploratory.json:
  1. Circular model-based (plug-in) vs held-out AIPW optimism across ALL 5 models
     (DR-learner, Causal forest, CausalPFN, Calibrated ensemble, Scale ranking), with exact
     paired cluster-bootstrap SEs for (plugin - heldout).
  2. Trial ATE distribution and exact gain over random selection (top 20% minus trial ATE) for
     risk, scale, response, and DR rankings across all 22 trials and by domain.
  3. Mid-propensity targeting comparator (-|p0 - 0.5|) in the 7 voting trials.
  4. PATH-style 4-parameter treatment x logit-risk interaction model across all 22 trials.
  5. Stronger CATE sensitivity check across all 22 trials: DR-learner given logit(p0) as an
     explicit feature, hyperparameter-tuned on context->dev (minimizing dev-split DR pseudo-outcome
     MSE), and refit on context + development combined (70% of each trial).
  6. Learning curves over context training size (5k, 15k, 50k, 100k, full) in the 4 largest
     trials (total N > 200,000: Michigan mailer, Lenta, X5, Michigan robocall).
"""

from __future__ import annotations

import importlib.util
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import expit, logit
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT.parent / "system-three-sealed"
PRED = ROOT / "predictions"
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

spec_mr = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec_mr)
sys.modules["mr"] = mr
spec_mr.loader.exec_module(mr)

spec_diag = importlib.util.spec_from_file_location("diag", ROOT / "scripts" / "28_proportionality_diagnostic.py")
diag = importlib.util.module_from_spec(spec_diag)
spec_diag.loader.exec_module(diag)

from system_three import rankers  # noqa: E402
from system_three.analysis import prep  # noqa: E402
from system_three.estimators.dr import aipw_scores, sign  # noqa: E402
from system_three.splits import load as load_split  # noqa: E402

SEED = 20261112
N_BOOT = 2000

DOM_MAP = {
    "gusto": "medicine", "ist": "medicine", "synbiotic": "medicine", "elaia2": "medicine",
    "voter": "voting", "sf_email": "voting", "sixcity": "voting", "newhaven": "voting",
    "w2_robocall2008": "voting", "w2_sanmateo_email": "voting", "w2_wisconsin_recall": "voting",
    "hillstrom": "retail", "lenta": "retail", "x5": "retail",
}

VOTING_7 = ["voter", "sf_email", "sixcity", "newhaven", "w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall"]
BIG_4 = ["voter", "lenta", "x5", "w2_robocall2008"]


def re_pool(est: list[float], se: list[float], floor: float = 0.0) -> dict:
    e = np.asarray(est, float)
    s = np.asarray(se, float)
    keep = np.isfinite(e) & np.isfinite(s) & (s > 0)
    e, s = e[keep], s[keep]
    k = len(e)
    if k == 0:
        return {"k": 0}
    if k == 1:
        return {"k": 1, "mean": float(e[0]), "se": float(s[0]),
                "lo95": float(e[0] - 1.96 * s[0]), "hi95": float(e[0] + 1.96 * s[0]), "tau": 0.0}
    v = np.maximum(s ** 2, floor)
    f = mr.fit_re(e, np.ones((k, 1)), v)
    se_m = float(np.sqrt(f.vcov[0, 0]))
    c = float(stats.t.ppf(0.975, f.df))
    p = float(2 * stats.t.sf(abs(f.beta[0] / se_m), f.df)) if se_m > 0 else 1.0
    half_pi = c * float(np.sqrt(f.tau2 + se_m ** 2))
    return {
        "k": k, "mean": float(f.beta[0]), "se": se_m,
        "lo95": float(f.beta[0] - c * se_m), "hi95": float(f.beta[0] + c * se_m),
        "p_two_sided": p, "tau": float(np.sqrt(f.tau2)),
        "pi95": [float(f.beta[0] - half_pi), float(f.beta[0] + half_pi)],
    }


def load_test_base(t: str):
    meta = json.loads((SEALED / f"{t}_meta.json").read_text())
    sealed = pd.read_parquet(SEALED / f"{t}_test.parquet")
    pred = pd.read_parquet(PRED / f"{t}.parquet")
    two = sealed[sealed["action"].isin([meta["action"], meta["reference"]])]
    d = two.merge(pred, on="row_id", how="inner", validate="1:1").reset_index(drop=True)
    w = (d["action"] == meta["action"]).to_numpy(int)
    if meta.get("p_action") is not None:
        e = np.full(len(d), meta["p_action"] / (meta["p_action"] + meta["p_reference"]))
    else:
        p = d["propensity"].to_numpy(float)
        e = np.where(w == 1, p, 1 - p)
    g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float),
                    d["m1"].to_numpy(float), meta["better"])
    return d, meta, g, d["cluster_id"].to_numpy()


def fast_boot_contrasts(g: np.ndarray, scores: dict[str, np.ndarray], cl: np.ndarray,
                        pairs: list[tuple[str, str]], q: float = 0.2,
                        include_vs_ate: list[str] | None = None,
                        plugin_scores: dict[str, np.ndarray] | None = None) -> dict:
    """Exact cluster-bootstrap inference for paired policy-value contrasts, vs-ATE, and plug-in optimism."""
    uniq, inv = np.unique(cl, return_inverse=True)
    G = len(uniq)
    n = len(g)
    pre = {}
    for name, s in scores.items():
        jit = np.random.default_rng(SEED).random(n)
        order = np.lexsort((jit, -s))
        p_ord = plugin_scores[name][order] if (plugin_scores and name in plugin_scores) else None
        pre[name] = (g[order], inv[order], p_ord)
    # Also need plugin score evaluated on risk's top-q set for optimism
    risk_order = np.lexsort((np.random.default_rng(SEED).random(n), -scores["risk"]))
    risk_inv = inv[risk_order]
    risk_plugin = {name: arr[risk_order] for name, arr in (plugin_scores or {}).items()}

    def eval_topq(vals_ord, inv_ord, c_mult):
        ws = c_mult[inv_ord]
        n_tot = int(ws.sum())
        k = max(1, int(round(q * n_tot)))
        csum = np.cumsum(ws)
        pos = int(np.searchsorted(csum, k, side="left"))
        prev_w = int(csum[pos - 1]) if pos > 0 else 0
        rem = k - prev_w
        num = float(np.dot(vals_ord[:pos], ws[:pos]) + vals_ord[pos] * rem)
        return 1000.0 * num / k

    ones = np.ones(G, dtype=np.int64)
    pt_top = {name: eval_topq(pre[name][0], pre[name][1], ones) for name in scores}
    pt_ate = float(1000.0 * g.mean())
    pt_plug = {}
    if plugin_scores:
        for name in plugin_scores:
            p_m = eval_topq(pre[name][2], pre[name][1], ones)
            p_r = eval_topq(risk_plugin[name], risk_inv, ones)
            pt_plug[name] = {
                "plugin": p_m - p_r,
                "heldout": pt_top[name] - pt_top["risk"],
                "overstatement": (p_m - p_r) - (pt_top[name] - pt_top["risk"]),
            }

    rng = np.random.default_rng(SEED)
    draws_pair = {f"{a}_minus_{b}": np.empty(N_BOOT) for a, b in pairs}
    draws_ate = {name: np.empty(N_BOOT) for name in (include_vs_ate or [])}
    draws_over = {name: np.empty(N_BOOT) for name in (plugin_scores or {})}
    ate_draws = np.empty(N_BOOT)

    # Pre-aggregate cluster sums of g and cluster sizes for O(G) ATE computation
    cl_gsum = np.bincount(inv, weights=g, minlength=G)
    cl_cnt = np.bincount(inv, minlength=G)

    for b_idx in range(N_BOOT):
        c_mult = np.bincount(rng.integers(0, G, G), minlength=G)
        ate_b = 1000.0 * float(np.dot(cl_gsum, c_mult)) / float(np.dot(cl_cnt, c_mult))
        ate_draws[b_idx] = ate_b
        vals = {}
        plug_m = {}
        for name, (gs, inv_s, ps) in pre.items():
            ws = c_mult[inv_s]
            n_tot = int(ws.sum())
            k = max(1, int(round(q * n_tot)))
            csum = np.cumsum(ws)
            pos = int(np.searchsorted(csum, k, side="left"))
            prev_w = int(csum[pos - 1]) if pos > 0 else 0
            rem = k - prev_w
            vals[name] = 1000.0 * (float(np.dot(gs[:pos], ws[:pos])) + gs[pos] * rem) / k
            if ps is not None:
                plug_m[name] = 1000.0 * (float(np.dot(ps[:pos], ws[:pos])) + ps[pos] * rem) / k
        if plugin_scores:
            ws_r = c_mult[risk_inv]
            n_tot_r = int(ws_r.sum())
            k_r = max(1, int(round(q * n_tot_r)))
            csum_r = np.cumsum(ws_r)
            pos_r = int(np.searchsorted(csum_r, k_r, side="left"))
            prev_w_r = int(csum_r[pos_r - 1]) if pos_r > 0 else 0
            rem_r = k_r - prev_w_r
            for name, p_r_ord in risk_plugin.items():
                p_r_val = 1000.0 * (float(np.dot(p_r_ord[:pos_r], ws_r[:pos_r])) + p_r_ord[pos_r] * rem_r) / k_r
                draws_over[name][b_idx] = (plug_m[name] - p_r_val) - (vals[name] - vals["risk"])
        for a, b in pairs:
            draws_pair[f"{a}_minus_{b}"][b_idx] = vals[a] - vals[b]
        for name in (include_vs_ate or []):
            draws_ate[name][b_idx] = vals[name] - ate_b

    def summarize(pt, arr):
        lo, hi = np.percentile(arr, [2.5, 97.5])
        return {"estimate": float(pt), "lo95": float(lo), "hi95": float(hi), "boot_sd": float(arr.std(ddof=1))}

    res = {
        "ate": summarize(pt_ate, ate_draws),
        "pairs": {k: summarize(pt_top[k.split("_minus_")[0]] - pt_top[k.split("_minus_")[1]], v)
                  for k, v in draws_pair.items()},
        "vs_ate": {k: summarize(pt_top[k] - pt_ate, v) for k, v in draws_ate.items()},
        "optimism": {
            k: {**pt_plug[k], **summarize(pt_plug[k]["overstatement"], v)}
            for k, v in draws_over.items()
        },
    }
    return res


def fit_path_interaction(t: str, d_test: pd.DataFrame) -> np.ndarray:
    """Fit 4-parameter PATH interaction model on context split, score test units."""
    ctx = prep(t, "context")
    Xc, _ = rankers.design_matrix(ctx)
    w = ctx["W"].to_numpy()
    y = ctx["outcome"].to_numpy()
    e = ctx["e"].to_numpy()
    groups = ctx["cluster_id"].to_numpy()
    better = ctx["better"].iat[0]
    m0_oof, _, _, _ = rankers.cross_fit_p(Xc, w, y, groups)
    off_c = logit(m0_oof)
    mean_off = float(off_c.mean())
    c_c = off_c - mean_off
    wt = np.where(w == 1, 1.0 / e, 1.0 / (1.0 - e))
    th1, _ = diag.glm(np.column_stack([np.ones_like(off_c), off_c, w, w * c_c]), y, wt, groups)
    m0_t = np.clip(d_test["m0"].to_numpy(float), 1e-4, 1.0 - 1e-4)
    off_t = logit(m0_t)
    c_t = off_t - mean_off
    lin_t = th1[0] + th1[1] * off_t
    return sign(better) * (expit(lin_t + th1[2] + th1[3] * c_t) - expit(lin_t))


def fit_stronger_dr(t: str, test_row_ids: np.ndarray) -> tuple[np.ndarray, dict]:
    """Tuned, risk-augmented DR-learner fit on context+dev (70%), tuned via context->dev DR MSE."""
    ctx = prep(t, "context")
    dev = prep(t, "dev")
    test_states = load_split(t, "test")
    Xc, cols = rankers.design_matrix(ctx)
    Xd, _ = rankers.design_matrix(dev, cols)
    Xt_all, _ = rankers.design_matrix(test_states, cols)
    # Align test_states to d_test row_ids
    pos_map = pd.Series(np.arange(len(test_states)), index=test_states["row_id"].to_numpy())
    t_idx = pos_map.loc[test_row_ids].to_numpy()
    Xt = Xt_all[t_idx]

    wc, yc, ec, gc = ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(), ctx["cluster_id"].to_numpy()
    wd, yd, ed, gd = dev["W"].to_numpy(), dev["outcome"].to_numpy(), dev["e"].to_numpy(), dev["cluster_id"].to_numpy()
    better = ctx["better"].iat[0]

    # Step 1: Cross-fit nuisances on context, augment X with logit(m0), evaluate candidate HPs on dev
    m0c, m1c, f0c, f1c = rankers.cross_fit_p(Xc, wc, yc, gc)
    pseudo_c = m1c - m0c + wc * (yc - m1c) / ec - (1 - wc) * (yc - m0c) / (1 - ec)
    m0d = rankers._avg(f0c, Xd)
    m1d = rankers._avg(f1c, Xd)
    pseudo_d = m1d - m0d + wd * (yd - m1d) / ed - (1 - wd) * (yd - m0d) / (1 - ed)

    Xc_aug = np.column_stack([Xc, logit(m0c)])
    Xd_aug = np.column_stack([Xd, logit(m0d)])

    grid = [
        {"max_leaf_nodes": 7, "min_samples_leaf": 200, "l2_regularization": 10.0, "max_iter": 150, "learning_rate": 0.03},
        {"max_leaf_nodes": 7, "min_samples_leaf": 100, "l2_regularization": 5.0, "max_iter": 200, "learning_rate": 0.05},
        {"max_leaf_nodes": 15, "min_samples_leaf": 100, "l2_regularization": 5.0, "max_iter": 200, "learning_rate": 0.05},
        {"max_leaf_nodes": 15, "min_samples_leaf": 50, "l2_regularization": 1.0, "max_iter": 300, "learning_rate": 0.05},
        {"max_leaf_nodes": 31, "min_samples_leaf": 50, "l2_regularization": 1.0, "max_iter": 300, "learning_rate": 0.05},
        {"max_leaf_nodes": 31, "min_samples_leaf": 20, "l2_regularization": 0.1, "max_iter": 300, "learning_rate": 0.05},
    ]
    best_mse = float("inf")
    best_hp = grid[0]
    for hp in grid:
        mdl = HistGradientBoostingRegressor(**hp, early_stopping=False, random_state=SEED).fit(Xc_aug, pseudo_c)
        pred_d = mdl.predict(Xd_aug)
        mse = float(np.mean((pseudo_d - pred_d) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_hp = hp

    # Step 2: Refit nuisances + tuned risk-augmented DR on context + dev combined (70%)
    X_cd = np.vstack([Xc, Xd])
    w_cd = np.concatenate([wc, wd])
    y_cd = np.concatenate([yc, yd])
    e_cd = np.concatenate([ec, ed])
    g_cd = np.concatenate([gc, gd])
    m0_cd, m1_cd, f0_cd, f1_cd = rankers.cross_fit_p(X_cd, w_cd, y_cd, g_cd)
    pseudo_cd = m1_cd - m0_cd + w_cd * (y_cd - m1_cd) / e_cd - (1 - w_cd) * (y_cd - m0_cd) / (1 - e_cd)
    X_cd_aug = np.column_stack([X_cd, logit(m0_cd)])
    m0_t = rankers._avg(f0_cd, Xt)
    Xt_aug = np.column_stack([Xt, logit(m0_t)])
    mdl_final = HistGradientBoostingRegressor(**best_hp, early_stopping=False, random_state=SEED).fit(X_cd_aug, pseudo_cd)
    score_t = sign(better) * mdl_final.predict(Xt_aug)
    return score_t, best_hp


def run_learning_curves() -> dict:
    """Learning curves over training size in the 4 largest trials (total N > 200,000)."""
    sizes = [5000, 15000, 50000, 100000]
    seeds = [101, 202, 303]
    out = {}
    for t in BIG_4:
        t0 = time.time()
        ctx = prep(t, "context")
        Xc, cols = rankers.design_matrix(ctx)
        wc, yc, ec = ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy()
        gc = ctx["cluster_id"].to_numpy()
        better = ctx["better"].iat[0]

        d_test, meta, g_test, cl_test = load_test_base(t)
        test_states = load_split(t, "test")
        Xt_all, _ = rankers.design_matrix(test_states, cols)
        pos_map = pd.Series(np.arange(len(test_states)), index=test_states["row_id"].to_numpy())
        Xt = Xt_all[pos_map.loc[d_test["row_id"].to_numpy()].to_numpy()]

        # Cluster index mapping for cluster-level subsampling
        uniq_c, inv_c = np.unique(gc, return_inverse=True)
        cl_rows = [np.flatnonzero(inv_c == k) for k in range(len(uniq_c))]
        cl_lens = np.array([len(r) for r in cl_rows])

        def eval_point(score_vec):
            jit = np.random.default_rng(SEED).random(len(g_test))
            k20 = max(1, int(round(0.2 * len(g_test))))
            top = np.lexsort((jit, -score_vec))[:k20]
            return float(1000.0 * g_test[top].mean())

        risk_val = eval_point(d_test["risk"].to_numpy(float))
        full_scale_val = eval_point(d_test["scale"].to_numpy(float))
        full_dr_val = eval_point(d_test["benefit"].to_numpy(float))

        curve = {}
        for sz in sizes:
            if sz >= len(ctx) - 1000:
                continue
            sc_gains, dr_gains, dr_sc_gains = [], [], []
            for sd in seeds:
                rng = np.random.default_rng(sd)
                perm = rng.permutation(len(uniq_c))
                cs = np.cumsum(cl_lens[perm])
                m_cl = int(np.searchsorted(cs, sz, side="left")) + 1
                sub_idx = np.concatenate([cl_rows[k] for k in perm[:m_cl]])
                f_sub = rankers.fit(Xc[sub_idx], wc[sub_idx], yc[sub_idx], ec[sub_idx],
                                    better, groups=gc[sub_idx])
                s_sub = rankers.scores(f_sub, Xt)
                v_r = eval_point(s_sub["risk"])
                v_s = eval_point(s_sub["scale"])
                v_b = eval_point(s_sub["benefit"])
                # Compare against the same fixed full-sample risk ranking AND subsample risk ranking
                sc_gains.append(v_s - risk_val)
                dr_gains.append(v_b - risk_val)
                dr_sc_gains.append(v_b - v_s)
            curve[str(sz)] = {
                "n_ctx": sz,
                "scale_minus_fullrisk": float(np.mean(sc_gains)),
                "dr_minus_fullrisk": float(np.mean(dr_gains)),
                "dr_minus_scale": float(np.mean(dr_sc_gains)),
                "dr_minus_scale_sd_across_seeds": float(np.std(dr_sc_gains, ddof=1)),
            }
        curve["full"] = {
            "n_ctx": int(len(ctx)),
            "scale_minus_fullrisk": full_scale_val - risk_val,
            "dr_minus_fullrisk": full_dr_val - risk_val,
            "dr_minus_scale": full_dr_val - full_scale_val,
            "dr_minus_scale_sd_across_seeds": 0.0,
        }
        out[t] = curve
        print(f"  learning curve {t} done in {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> None:
    fr = json.loads((R / "final_results.json").read_text())
    all22 = fr["trials"]["wave1"] + fr["trials"]["wave2"]

    per_trial = {}
    t_start = time.time()
    for t in all22:
        t0 = time.time()
        d, meta, g, cl = load_test_base(t)
        resp_score = d["m1"].to_numpy(float) if meta["better"] == "higher" else (1.0 - d["m1"].to_numpy(float))

        # Load other pre-registered learners (already hashed)
        cf = pd.read_parquet(ROOT / "predictions_cforest" / f"{t}.parquet")[["row_id", "benefit"]].rename(columns={"benefit": "cf"})
        cpfn = pd.read_parquet(ROOT / "predictions_causalpfn" / f"{t}.parquet")[["row_id", "benefit"]].rename(columns={"benefit": "cpfn"})
        rose = pd.read_parquet(ROOT / "predictions_rose_v2" / f"{t}.parquet")[["row_id", "benefit"]].rename(columns={"benefit": "rose"})
        dm = d.merge(cf, on="row_id", validate="1:1").merge(cpfn, on="row_id", validate="1:1").merge(rose, on="row_id", validate="1:1")

        # Exploratory PATH interaction & stronger tuned risk-augmented context+dev DR
        path_score = fit_path_interaction(t, dm)
        strong_dr_score, best_hp = fit_stronger_dr(t, dm["row_id"].to_numpy())

        scores = {
            "risk": dm["risk"].to_numpy(float),
            "scale": dm["scale"].to_numpy(float),
            "response": resp_score,
            "dr": dm["benefit"].to_numpy(float),
            "cf": dm["cf"].to_numpy(float),
            "cpfn": dm["cpfn"].to_numpy(float),
            "rose": dm["rose"].to_numpy(float),
            "path_int": path_score,
            "dr_strong": strong_dr_score,
        }
        pairs = [
            ("path_int", "risk"), ("path_int", "scale"), ("dr", "path_int"),
            ("dr_strong", "risk"), ("dr_strong", "scale"), ("dr_strong", "response"), ("dr_strong", "dr"),
        ]
        if t in VOTING_7:
            scores["midprop"] = -np.abs(dm["m0"].to_numpy(float) - 0.5)
            pairs += [("midprop", "risk"), ("scale", "midprop"), ("response", "midprop"), ("dr", "midprop")]

        plugin_map = {
            "scale": scores["scale"],
            "dr": scores["dr"],
            "cf": scores["cf"],
            "cpfn": scores["cpfn"],
            "rose": scores["rose"],
        }
        res_t = fast_boot_contrasts(
            g, scores, cl, pairs=pairs, q=0.2,
            include_vs_ate=["risk", "scale", "response", "dr", "path_int", "dr_strong"],
            plugin_scores=plugin_map,
        )
        res_t["domain"] = DOM_MAP.get(t, "other")
        res_t["best_hp_dr_strong"] = best_hp
        per_trial[t] = res_t
        print(f"[{t:20s}] {time.time()-t0:4.1f}s | ATE={10*res_t['ate']['estimate']:+5.0f} | "
              f"PATH-sc={10*res_t['pairs']['path_int_minus_scale']['estimate']:+5.0f} | "
              f"DRstr-sc={10*res_t['pairs']['dr_strong_minus_scale']['estimate']:+5.0f} | "
              f"DRstr-DR={10*res_t['pairs']['dr_strong_minus_dr']['estimate']:+5.0f}", flush=True)

    # Pool everything cleanly using exact same REML + Knapp-Hartung + floor
    def pool_key(trials_sub, getter, skip_zero=False):
        es, sds = [], []
        for t in trials_sub:
            r = getter(per_trial[t])
            if skip_zero and r["boot_sd"] < 1e-9 and abs(r["estimate"]) < 1e-9:
                continue
            es.append(r["estimate"])
            sds.append(r["boot_sd"])
        return re_pool(es, sds, floor=mr.V_FLOOR)

    # 1. Optimism pooled across all 5 models
    optimism_pooled = {}
    for nm in ("dr", "cf", "cpfn", "rose", "scale"):
        sub_t = [t for t in all22 if not (nm == "scale" and per_trial[t]["optimism"][nm]["boot_sd"] < 1e-9 and abs(per_trial[t]["optimism"][nm]["overstatement"]) < 1e-9)]
        p_opt = pool_key(sub_t, lambda x, n_=nm: x["optimism"][n_])
        p_opt["mean_plugin"] = float(np.mean([per_trial[t]["optimism"][nm]["plugin"] for t in sub_t]))
        p_opt["mean_heldout"] = float(np.mean([per_trial[t]["optimism"][nm]["heldout"] for t in sub_t]))
        p_opt["n_overstated"] = int(sum(per_trial[t]["optimism"][nm]["overstatement"] > 0 for t in sub_t))
        p_opt["n_trials"] = len(sub_t)
        optimism_pooled[nm] = p_opt

    # 2. Gain over random selection (ATE)
    vs_ate_pooled = {}
    for nm in ("risk", "scale", "response", "dr", "path_int", "dr_strong"):
        vs_ate_pooled[nm] = {
            "all22": pool_key(all22, lambda x, n_=nm: x["vs_ate"][n_]),
            "voting7": pool_key(VOTING_7, lambda x, n_=nm: x["vs_ate"][n_]),
            "medicine4": pool_key(["gusto", "ist", "synbiotic", "elaia2"], lambda x, n_=nm: x["vs_ate"][n_]),
            "non_voting15": pool_key([t for t in all22 if t not in VOTING_7], lambda x, n_=nm: x["vs_ate"][n_]),
        }

    # 3. Mid-propensity in voting
    midprop_voting = {}
    for k_pair in ("midprop_minus_risk", "scale_minus_midprop", "response_minus_midprop", "dr_minus_midprop"):
        midprop_voting[k_pair] = pool_key(VOTING_7, lambda x, kp=k_pair: x["pairs"][kp])

    # 4 & 5. PATH interaction and stronger DR pooled
    comparators_pooled = {}
    for k_pair in ("path_int_minus_risk", "path_int_minus_scale", "dr_minus_path_int",
                   "dr_strong_minus_risk", "dr_strong_minus_scale", "dr_strong_minus_response", "dr_strong_minus_dr"):
        comparators_pooled[k_pair] = {
            "all": pool_key(all22, lambda x, kp=k_pair: x["pairs"][kp], skip_zero=True),
            "voting7": pool_key(VOTING_7, lambda x, kp=k_pair: x["pairs"][kp], skip_zero=True),
            "non_voting15": pool_key([t for t in all22 if t not in VOTING_7], lambda x, kp=k_pair: x["pairs"][kp], skip_zero=True),
            "medicine4": pool_key(["gusto", "ist", "synbiotic", "elaia2"], lambda x, kp=k_pair: x["pairs"][kp], skip_zero=True),
            "big4": pool_key(BIG_4, lambda x, kp=k_pair: x["pairs"][kp], skip_zero=True),
        }

    # 6. Learning curves in the 4 largest trials
    print("\nRunning learning curves on 4 largest trials...", flush=True)
    lc = run_learning_curves()

    ates = {t: per_trial[t]["ate"]["estimate"] for t in all22}
    out = {
        "note": "Post-unsealing exploratory analyses; values in per-1,000 units (multiply by 10 for per 10,000 reached).",
        "ate_summary": {
            "per_trial": ates,
            "min": float(min(ates.values())),
            "max": float(max(ates.values())),
            "median": float(np.median(list(ates.values()))),
            "beneficial_19_min": float(min(ates[t] for t in fr["Q2"]["points"])),
            "beneficial_19_max": float(max(ates[t] for t in fr["Q2"]["points"])),
            "beneficial_19_median": float(np.median([ates[t] for t in fr["Q2"]["points"]])),
        },
        "optimism_all_models": optimism_pooled,
        "gain_over_random_ate": vs_ate_pooled,
        "midprop_voting7": midprop_voting,
        "path_and_stronger_dr_pooled": comparators_pooled,
        "learning_curves_big4": lc,
        "per_trial": per_trial,
    }
    (R / "ben_additional_exploratory.json").write_text(json.dumps(out, indent=1))
    print(f"\nWROTE {R / 'ben_additional_exploratory.json'} in {time.time() - t_start:.1f}s")

    # Crisp summary scaled x10 to per 10,000 reached
    print("\n=== ATE range (per 10,000 reached) ===")
    a_s = out["ate_summary"]
    print(f"  All 22: {10*a_s['min']:.0f} to {10*a_s['max']:.0f} (median {10*a_s['median']:.0f}); 19 beneficial: {10*a_s['beneficial_19_min']:.0f} to {10*a_s['beneficial_19_max']:.0f} (median {10*a_s['beneficial_19_median']:.0f})")

    print("\n=== 1. Circular plug-in vs held-out optimism (per 10,000 reached) ===")
    for nm, v in optimism_pooled.items():
        print(f"  {nm:6s} (k={v['n_trials']:2d}): plugin={10*v['mean_plugin']:+6.0f} heldout={10*v['mean_heldout']:+5.0f} | RE over={10*v['mean']:+5.0f} [{10*v['lo95']:+5.0f}, {10*v['hi95']:+5.0f}] (over in {v['n_overstated']}/{v['n_trials']})")

    print("\n=== 2. Gain over random selection (top 20% minus trial ATE, per 10,000) ===")
    for nm in ("risk", "scale", "response", "dr", "path_int", "dr_strong"):
        v = vs_ate_pooled[nm]["all22"]
        vv = vs_ate_pooled[nm]["voting7"]
        vm = vs_ate_pooled[nm]["medicine4"]
        print(f"  {nm:10s}: all22={10*v['mean']:+5.0f} [{10*v['lo95']:+5.0f}, {10*v['hi95']:+5.0f}] | voting={10*vv['mean']:+5.0f} [{10*vv['lo95']:+5.0f}, {10*vv['hi95']:+5.0f}] | med4={10*vm['mean']:+5.0f} [{10*vm['lo95']:+5.0f}, {10*vm['hi95']:+5.0f}]")

    print("\n=== 3. Mid-propensity targeting in 7 voting trials (per 10,000) ===")
    for kp, v in midprop_voting.items():
        print(f"  {kp:25s}: {10*v['mean']:+5.0f} [{10*v['lo95']:+5.0f}, {10*v['hi95']:+5.0f}]")

    print("\n=== 4 & 5. PATH interaction & Stronger DR (tuned + logit-risk + ctx+dev 70%, per 10,000) ===")
    for kp, grp in comparators_pooled.items():
        va, vv, vnv, vb4 = grp["all"], grp["voting7"], grp["non_voting15"], grp["big4"]
        print(f"  {kp:26s} (k={va['k']:2d}): all={10*va['mean']:+5.0f} [{10*va['lo95']:+5.0f}, {10*va['hi95']:+5.0f}] | "
              f"vot={10*vv['mean']:+5.0f} | nonvot={10*vnv['mean']:+5.0f} | big4={10*vb4['mean']:+5.0f}")

    print("\n=== 6. Learning curves in 4 largest trials (DR minus Scale, per 10,000) ===")
    for t, curve in lc.items():
        s = ", ".join(f"{k}:{10*v['dr_minus_scale']:+.0f} (DR={10*v['dr_minus_fullrisk']:+.0f}, Sc={10*v['scale_minus_fullrisk']:+.0f})"
                      for k, v in curve.items())
        print(f"  {t:18s}: {s}")


if __name__ == "__main__":
    main()
