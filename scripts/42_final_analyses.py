"""Final analyses for the paper: one script, one output (results/final_results.json).

Trial set: 22 trials (13 wave 1; 9 wave 2 after applying the registered size rule, which excludes
the Washington bonus trial, n = 4,617 < 5,000). Forecast questions use the trials whose context
effect is beneficial (sign rule). All fits use the frozen 07 functions (REML, Knapp-Hartung,
development-split weights floored at (2 per 1,000)^2).

Q1  Theorem check: the Proposition 2 condition against the posted gap (results/theory_checks.json).
Q2  Forecast slope: all eligible; informative trials only (gap not 0 by construction); slope x wave
    interaction; leave-one-trial-out ("driven by") and LPD; size placebo; error in variables;
    budgets 10% and 30%; design and bootstrap weights; split-context (wave 1, where it exists).
Q3  Learned rankings (DR, causal forest, CausalPFN, Rose v2) and the language model: gain over risk
    ranking at 20%, by wave and pooled, with between-trial SD, new-trial 95% PI, and wave
    difference; scale over risk and DR over scale, pooled.
Q4  Rose v2 uncertainty: group coverage (trials), unit coverage and harm Brier (routing).
"""

from __future__ import annotations

import glob
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec)
sys.modules["mr"] = mr
spec.loader.exec_module(mr)
R = ROOT / "results"
W1 = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
      "sixcity", "newhaven", "lenta", "x5"]
W2 = ["w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
      "w2_mooc_megastudy", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
ALL = W1 + W2
RB = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4", "routerbench__llama70b_to_claude2",
      "routerbench__mixtral_to_gpt4"]
SETS = {"dr": "predictions", "causal_forest": "predictions_cforest", "causalpfn": "predictions_causalpfn",
        "rose": "predictions_rose_v2"}


def table(set_name, t):
    ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"{set_name}__{t}.json")))
    if not ps:
        return None
    tt = next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values()))
    return tt


def cell(tt, rk, q=0.2):
    return next(x for x in tt["table"] if x["ranking"] == rk and x["budget"] == q)


def slope(x, y, v, X=None):
    X = np.column_stack([np.ones_like(x), x]) if X is None else X
    f = mr.fit_re(np.asarray(y, float), X, np.asarray(v, float))
    return f, mr.slope_test(f)


def pool(e, sd):
    """Random-effects pool. Trials where the two rankings select the same units (estimate 0 with zero
    bootstrap SD) carry no sampling information and are left out; k counts the trials used."""
    e, sd = np.asarray(e, float), np.asarray(sd, float)
    keep = ~((sd < 1e-9) & (np.abs(e) < 1e-9))
    e, sd = e[keep], sd[keep]
    v = np.maximum(sd ** 2, mr.V_FLOOR)
    f = mr.fit_re(e, np.ones((len(e), 1)), v)
    se = float(np.sqrt(f.vcov[0, 0]))
    c = stats.t.ppf(0.975, f.df)
    half = c * np.sqrt(f.tau2 + se**2)
    return {"k": len(e), "mean": float(f.beta[0]), "lo95": float(f.beta[0] - c * se), "hi95": float(f.beta[0] + c * se),
            "tau": float(np.sqrt(f.tau2)), "pi95": [float(f.beta[0] - half), float(f.beta[0] + half)]}


def main() -> None:
    post = mr.posted()
    out = {"trials": {"wave1": W1, "wave2": W2, "excluded": {"w2_wa_bonus": "registered size rule (n = 4,617 < 5,000)",
                                                            "elaia1": "sign rule", "oregon": "sign rule"}}}
    dt = json.loads((R / "dataset_table.json").read_text())
    out["n_test_units"] = {"wave1": int(sum(table("predictions", t)["n"] for t in W1)),
                           "wave2": int(sum(table("predictions", t)["n"] for t in W2))}
    # Q1
    th = json.loads((R / "theory_checks.json").read_text())
    elig = [t for t in ALL if post[t]["beneficial"]]
    zeros = [t for t in elig if abs(post[t]["x"]) < 1e-9]
    out["Q1"] = {"simulation": th["simulation_agreement"], "eligible": len(elig), "zero_gap_trials": zeros,
                 "condition_matches_posted_gap": int(sum((not th["trials"][t]["condition_holds"]) == (t in zeros) for t in elig))}
    # Q2
    obs = {t: cell(table("predictions", t), "scale-minus-risk") for t in ALL}
    x = np.array([post[t]["x"] for t in elig])
    y = np.array([obs[t]["estimate"] for t in elig])
    v = np.array([post[t]["v_primary"] for t in elig])
    wv = np.array([1.0 if t.startswith("w2_") else 0.0 for t in elig])
    q2 = {"k": len(elig), "k_wave1": int((wv == 0).sum()), "k_wave2": int((wv == 1).sum())}
    f, q2["primary"] = slope(x, y, v)
    q2["primary"]["joint_0_1"] = mr.joint_test(f)
    q2["primary_fit"] = {"beta": f.beta.tolist(), "vcov": f.vcov.tolist(), "df": float(f.df), "tau2": float(f.tau2)}
    q2["points"] = {t: {"x": float(post[t]["x"]), "y": obs[t]["estimate"], "lo95": obs[t]["lo95"], "hi95": obs[t]["hi95"]}
                    for t in elig}
    inf = np.array([t not in zeros for t in elig])
    _, q2["informative_only"] = slope(x[inf], y[inf], v[inf])
    q2["informative_only"]["k"] = int(inf.sum())
    # Proposition 2 fixes the line through the origin (gap 0 forecast implies gain 0); fit without intercept.
    fo = mr.fit_re(y[inf], x[inf][:, None], v[inf])
    se_o = float(np.sqrt(fo.vcov[0, 0]))
    c_o = stats.t.ppf(0.975, fo.df)
    q2["informative_through_origin"] = {"k": int(inf.sum()), "slope": float(fo.beta[0]), "lo95": float(fo.beta[0] - c_o * se_o),
                                        "hi95": float(fo.beta[0] + c_o * se_o),
                                        "p_two_sided": float(2 * stats.t.sf(abs(fo.beta[0] / se_o), fo.df))}
    m1i = (wv == 0) & inf
    _, q2["wave1_informative_only"] = slope(x[m1i], y[m1i], v[m1i])
    q2["wave1_informative_only"]["k"] = int(m1i.sum())
    f_w1 = mr.fit_re(y[wv == 0], np.column_stack([np.ones(int((wv == 0).sum())), x[wv == 0]]), v[wv == 0])
    w1w = 1 / (v[wv == 0] + f_w1.tau2)
    q2["weight_share_zero_gap_trials_wave1"] = float(w1w[~inf[wv == 0]].sum() / w1w.sum())
    wts = 1 / (v + f.tau2)
    q2["weight_share_zero_gap_trials"] = float(wts[~inf].sum() / wts.sum())
    # Truncated Knapp-Hartung (registered sensitivity) for the main fits.
    q2["truncated_kh"] = {}
    for name, m in (("primary", np.ones(len(x), bool)), ("informative_only", inf), ("wave1_only", wv == 0), ("wave2_only", wv == 1)):
        ft = mr.fit_re(y[m], np.column_stack([np.ones(m.sum()), x[m]]), v[m], truncate=True)
        q2["truncated_kh"][name] = mr.slope_test(ft)
    fi = mr.fit_re(y, np.column_stack([np.ones_like(x), x, wv, x * wv]), v)
    _sei = float(np.sqrt(fi.vcov[3, 3]))
    q2["slope_by_wave_interaction"] = {"coef": float(fi.beta[3]), "se": _sei,
                                      "lo95": float(fi.beta[3] - stats.t.ppf(0.975, fi.df) * _sei),
                                      "hi95": float(fi.beta[3] + stats.t.ppf(0.975, fi.df) * _sei),
                                      "p": float(2 * stats.t.sf(abs(fi.beta[3] / _sei), fi.df))}
    for name, m in (("wave1_only", wv == 0), ("wave2_only", wv == 1)):
        _, q2[name] = slope(x[m], y[m], v[m])
    loo, lpd = {}, 0.0
    for i, t in enumerate(elig):
        m = np.arange(len(elig)) != i
        X = np.column_stack([np.ones_like(x), x])
        f1, f0 = mr.fit_re(y[m], X[m], v[m]), mr.fit_re(y[m], X[m][:, :1], v[m])
        s1 = mr.slope_test(f1)
        mu1, var1 = mr.predict(f1, X[i], v[i])
        mu0, var0 = mr.predict(f0, X[i][:1], v[i])
        g = mr.lpd(y[i], mu1, var1, f1.df) - mr.lpd(y[i], mu0, var0, f0.df)
        lpd += g
        loo[t] = {"slope": s1["slope"], "lo95": s1["lo95"], "hi95": s1["hi95"], "lpd_gain": g}
    q2["leave_one_out"] = loo
    q2["driven_by"] = [t for t, r in loo.items() if r["lo95"] <= 0]
    q2["loo_lpd_gain_nats"] = lpd
    q2["loo_lpd_gain_informative_only"] = float(sum(r["lpd_gain"] for t, r in loo.items() if t not in zeros))
    ate = np.array([post[t]["pred_ate"] for t in elig])
    fb = mr.fit_re(y, np.column_stack([np.ones_like(x), x, ate]), v)
    q2["size_placebo_both"] = {"coef_gap": float(fb.beta[1]), "se_gap": float(np.sqrt(fb.vcov[1, 1])),
                               "lo95": float(fb.beta[1] - stats.t.ppf(0.975, fb.df) * np.sqrt(fb.vcov[1, 1])),
                               "hi95": float(fb.beta[1] + stats.t.ppf(0.975, fb.df) * np.sqrt(fb.vcov[1, 1]))}
    xse = np.array([post[t]["x_se"] for t in elig])
    w = 1 / v
    vx = np.average((x - np.average(x, weights=w)) ** 2, weights=w)
    lam = max(1e-6, (vx - np.average(xse**2, weights=w)) / vx)
    q2["eiv"] = {"reliability": float(lam), "slope": q2["primary"]["slope"] / lam}
    for b in (0.1, 0.3):
        yb = np.array([cell(table("predictions", t), "scale-minus-risk", b)["estimate"] for t in elig])
        _, q2[f"budget_{b}"] = slope(x, yb, v)
    vobs = np.array([max(obs[t]["boot_sd"] ** 2, mr.V_FLOOR) for t in elig])
    vdes = np.array([post[t]["v_design"] for t in elig])
    _, q2["bootstrap_weights"] = slope(x, y, vobs)
    _, q2["design_weights"] = slope(x, y, vdes)
    X2 = np.column_stack([np.ones_like(x), x])
    q2["truncated_kh"]["design_weights"] = mr.slope_test(mr.fit_re(y, X2, vdes, truncate=True))
    yb3 = np.array([cell(table("predictions", t), "scale-minus-risk", 0.3)["estimate"] for t in elig])
    q2["truncated_kh"]["budget_0.3"] = mr.slope_test(mr.fit_re(yb3, X2, v, truncate=True))
    # Wave 1 fit forecasts each eligible wave-2 trial (out of sample); gain over intercept-only.
    m1 = wv == 0
    X = np.column_stack([np.ones_like(x), x])
    f1, f0 = mr.fit_re(y[m1], X[m1], v[m1]), mr.fit_re(y[m1], X[m1][:, :1], v[m1])
    oos, g_oos, inside = {}, 0.0, 0
    for i in np.where(~m1)[0]:
        mu1, var1 = mr.predict(f1, X[i], v[i])
        mu0, var0 = mr.predict(f0, X[i][:1], v[i])
        half = stats.t.ppf(0.975, f1.df) * np.sqrt(var1)
        g = mr.lpd(y[i], mu1, var1, f1.df) - mr.lpd(y[i], mu0, var0, f0.df)
        g_oos += g
        inside += int(mu1 - half <= y[i] <= mu1 + half)
        oos[elig[i]] = {"forecast": float(mu1), "pi95": [float(mu1 - half), float(mu1 + half)], "observed": float(y[i]),
                        "lpd_gain": float(g)}
    q2["wave2_out_of_sample"] = {"per_trial": oos, "lpd_gain_nats": float(g_oos), "n_inside_pi95": inside, "k": len(oos)}
    # Data with known individual effects (model routing): forecast, observed, DR, diagnostic flag.
    tc = json.loads((R / "tierA_manifest.json").read_text())
    d1 = json.loads((R / "proportionality_diagnostic.json").read_text())
    d2 = json.loads((R / "proportionality_diagnostic_w2.json").read_text())
    rb = {}
    for t in RB:
        tt = table("predictions_tierA", t)
        rb[t] = {"n_test": int(tt["n"]), "forecast": float(tc[t]["scale_forecast_gap_q20"]),
                 "scale_minus_risk": {k: cell(tt, "scale-minus-risk")[k] for k in ("estimate", "lo95", "hi95")},
                 "dr_minus_risk": {k: cell(tt, "benefit-minus-risk")[k] for k in ("estimate", "lo95", "hi95")},
                 "flag": bool(d1[t]["flag_not_applicable"]),
                 "interaction": {k: d1[t][k] for k in ("delta", "delta_lo95", "delta_hi95")}}
        rb[t]["forecast_failed"] = bool(rb[t]["scale_minus_risk"]["hi95"] < 0 or
                                        (rb[t]["forecast"] > 0 and rb[t]["scale_minus_risk"]["estimate"] < 0))
    for t in ["swebench__p10_to_p50", "swebench__p25_to_p75", "swebench__p50_to_p90"]:
        tt = table("predictions_tierA", t)
        rb[t] = {"n_test": int(tt["n"]), "forecast": float(tc[t]["scale_forecast_gap_q20"]),
                 "scale_minus_risk": {k: cell(tt, "scale-minus-risk")[k] for k in ("estimate", "lo95", "hi95")},
                 "dr_minus_risk": {k: cell(tt, "benefit-minus-risk")[k] for k in ("estimate", "lo95", "hi95")},
                 "flag": None, "interaction": None, "forecast_failed": None, "uninformative": True}
    q2["routing"] = rb
    h1 = json.loads((R / "h1_analysis.json").read_text())
    q2["registered_falsifier_wave1"] = {"weak_slope_ci_excludes_0": bool(h1["H1_primary"]["lo95"] > 0),
                                        "strong_joint_test_p": float(h1["H1_primary"]["joint_0_1"]["p"]),
                                        "holdouts": json.loads((R / "holdouts.json").read_text())["holdouts"],
                                        "holdout_lpd_gain_nats": float(h1["holdout_lpd_gain_nats"]),
                                        "margin_nats": float(mr.FORECAST_MARGIN_NATS)}
    q2["diagnostic_flagged_trials"] = [t for t in ALL if {**d1, **d2}[t]["flag_not_applicable"]]
    out["Q2"] = q2
    # Q3
    q3 = {}
    for k, s in SETS.items():
        rows = {t: cell(table(s, t), "benefit-minus-risk") for t in ALL if table(s, t) is not None}
        e = {t: (r["estimate"], r["boot_sd"]) for t, r in rows.items()}
        res = {"per_trial": {t: {"estimate": r["estimate"], "lo95": r["lo95"], "hi95": r["hi95"]} for t, r in rows.items()}}
        for name, tl in (("wave1", W1), ("wave2", W2), ("all", ALL)):
            tl = [t for t in tl if t in e]
            if len(tl) >= 3:
                res[name] = pool(*zip(*[e[t] for t in tl]))
        tl = [t for t in ALL if t in e]
        ee = np.array([e[t][0] for t in tl])
        vv = np.maximum(np.array([e[t][1] for t in tl]) ** 2, mr.V_FLOOR)
        wv2 = np.array([1.0 if t.startswith("w2_") else 0.0 for t in tl])
        fw = mr.fit_re(ee, np.column_stack([np.ones_like(wv2), wv2]), vv)
        _se = float(np.sqrt(fw.vcov[1, 1]))
        _c = stats.t.ppf(0.975, fw.df)
        res["wave2_minus_wave1"] = {"est": float(fw.beta[1]), "se": _se, "lo95": float(fw.beta[1] - _c * _se),
                                    "hi95": float(fw.beta[1] + _c * _se),
                                    "p": float(2 * stats.t.sf(abs(fw.beta[1] / _se), fw.df))}
        res["n_ci_above_0"] = int(sum(r["lo95"] > 0 for r in rows.values()))
        res["n_ci_below_0"] = int(sum(r["hi95"] < 0 for r in rows.values()))
        q3[k] = res
    def wave_diff(set_name, rk, q=0.2):
        rows = [(t, cell(table(set_name, t), rk, q)) for t in ALL]
        rows = [(t, r) for t, r in rows if not (r["boot_sd"] < 1e-9 and abs(r["estimate"]) < 1e-9)]
        ee = np.array([r["estimate"] for _, r in rows])
        vv = np.maximum(np.array([r["boot_sd"] for _, r in rows]) ** 2, mr.V_FLOOR)
        w2v = np.array([1.0 if t.startswith("w2_") else 0.0 for t, _ in rows])
        fw = mr.fit_re(ee, np.column_stack([np.ones_like(w2v), w2v]), vv)
        se, c = float(np.sqrt(fw.vcov[1, 1])), stats.t.ppf(0.975, fw.df)
        return {"est": float(fw.beta[1]), "se": se, "lo95": float(fw.beta[1] - c * se), "hi95": float(fw.beta[1] + c * se),
                "p": float(2 * stats.t.sf(abs(fw.beta[1] / se), fw.df)), "df": float(fw.df)}

    for rk, name in (("scale-minus-risk", "scale_over_risk"), ("benefit-minus-scale", "dr_over_scale")):
        q3[name] = {w: pool(*zip(*[(cell(table("predictions", t), rk)["estimate"], cell(table("predictions", t), rk)["boot_sd"])
                                   for t in tl])) for w, tl in (("wave1", W1), ("wave2", W2), ("all", ALL))}
        q3[name]["wave2_minus_wave1"] = wave_diff("predictions", rk)
    # Learned rankings on the same trials as the scale-over-risk pool (where scale and risk differ).
    diff17 = [t for t in ALL if not (cell(table("predictions", t), "scale-minus-risk")["boot_sd"] < 1e-9
                                     and abs(cell(table("predictions", t), "scale-minus-risk")["estimate"]) < 1e-9)]
    q3["trials_scale_differs"] = diff17
    q3["learned_on_scale_trials"] = {k: pool(*zip(*[(cell(table(s_, t), "benefit-minus-risk")["estimate"],
                                                     cell(table(s_, t), "benefit-minus-risk")["boot_sd"]) for t in diff17]))
                                     for k, s_ in SETS.items()}
    # Budget sensitivity for the three main contrasts (all trials).
    q3["budgets"] = {}
    for b in (0.1, 0.2, 0.3):
        q3["budgets"][str(b)] = {name: pool(*zip(*[(cell(table("predictions", t), rk, b)["estimate"],
                                                    cell(table("predictions", t), rk, b)["boot_sd"]) for t in ALL]))
                                 for rk, name in (("benefit-minus-risk", "dr_over_risk"), ("scale-minus-risk", "scale_over_risk"),
                                                  ("benefit-minus-scale", "dr_over_scale"))}
    # Sensitivity: wave-1 trials excluded by the sign rule at enrollment (ELAIA-1, Oregon), doubly robust only.
    extra = {t: cell(table("predictions", t), "benefit-minus-risk") for t in ("elaia1", "oregon")}
    base = {t: cell(table("predictions", t), "benefit-minus-risk") for t in ALL}
    q3["dr_with_sign_rule_trials"] = {
        "wave1": pool(*zip(*[(r["estimate"], r["boot_sd"]) for r in [base[t] for t in W1] + list(extra.values())])),
        "all": pool(*zip(*[(r["estimate"], r["boot_sd"]) for r in list(base.values()) + list(extra.values())]))}
    # Rank-weighted average treatment effect (AUTOC, per 1,000) by ranking, pooled; risk AUTOC below 0 count.
    au = {}
    for rk in ("risk", "scale", "benefit"):
        rows = [table("predictions", t)["rate"][f"{rk}_autoc"] for t in ALL]
        au[rk] = pool([1000 * r["estimate"] for r in rows], [1000 * r["boot_sd"] for r in rows])
    for k_, s_ in SETS.items():
        if k_ == "dr":
            continue
        rows_ = [table(s_, t)["rate"]["benefit_autoc"] for t in ALL]
        au[k_] = pool([1000 * r["estimate"] for r in rows_], [1000 * r["boot_sd"] for r in rows_])
    for rk in ("risk", "scale", "benefit"):
        rows_ = [table("predictions", t)["rate"][f"{rk}_qini"] for t in ALL]
        au[f"{rk}_qini"] = pool([1000 * r["estimate"] for r in rows_], [1000 * r["boot_sd"] for r in rows_])
    au["n_risk_autoc_ci_below_0"] = int(sum(table("predictions", t)["rate"]["risk_autoc"]["hi95"] < 0 for t in ALL))
    au["n_risk_below_random_q20"] = int(sum(cell(table("predictions", t), "risk")["estimate"] < cell(table("predictions", t), "random")["estimate"]
                                            for t in ALL))
    q3["autoc"] = au
    # Exploratory: is the doubly robust gain larger where predicted effects vary more? (prediction files only)
    import pandas as pd
    het = np.array([float(pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")["benefit"].std()) * 1000 for t in ALL])
    ed = np.array([base[t]["estimate"] for t in ALL])
    vd = np.maximum(np.array([base[t]["boot_sd"] for t in ALL]) ** 2, mr.V_FLOOR)
    wvv = np.array([1.0 if t.startswith("w2_") else 0.0 for t in ALL])
    fh = mr.fit_re(ed, np.column_stack([np.ones_like(het), het, wvv]), vd)
    _c = stats.t.ppf(0.975, fh.df)
    q3["heterogeneity_moderator"] = {"sd_predicted_effect_per_1000": dict(zip(ALL, het.tolist())),
                                     "median_sd_wave1": float(np.median(het[wvv == 0])), "median_sd_wave2": float(np.median(het[wvv == 1])),
                                     "coef_per_1000_sd": float(fh.beta[1]),
                                     "lo95": float(fh.beta[1] - _c * np.sqrt(fh.vcov[1, 1])), "hi95": float(fh.beta[1] + _c * np.sqrt(fh.vcov[1, 1])),
                                     "wave2_adjusted": float(fh.beta[2]), "wave2_adjusted_lo95": float(fh.beta[2] - _c * np.sqrt(fh.vcov[2, 2])),
                                     "wave2_adjusted_hi95": float(fh.beta[2] + _c * np.sqrt(fh.vcov[2, 2]))}
    lc = [np.corrcoef([q3[a]["per_trial"][t]["estimate"] for t in ALL], [q3[b]["per_trial"][t]["estimate"] for t in ALL])[0, 1]
          for i, a in enumerate(SETS) for b in list(SETS)[i + 1:]]
    q3["between_learner_correlation_range"] = [float(min(lc)), float(max(lc))]
    llm = [cell(table("predictions_llm", t), "benefit-minus-risk") for t in W1]
    q3["language_model_wave1"] = pool([r["estimate"] for r in llm], [r["boot_sd"] for r in llm])
    q3["language_model_masked_wave1"] = json.loads((R / "llm_masked_compare.json").read_text())["pooled_masked"]
    dp = [cell(table("predictions_dopfn", t), "benefit-minus-risk") for t in W1]
    q3["dopfn_wave1"] = pool([r["estimate"] for r in dp], [r["boot_sd"] for r in dp])
    # Exploratory practitioner comparator: response ranking (P(good outcome | action)); see script 46.
    if all(table("predictions_response", t) is not None for t in ALL):
        rsp = {}
        for rk, name in (("risk", "response_value"), ("scale-minus-risk", "scale_over_response"),
                         ("benefit-minus-risk", "dr_over_response")):
            if rk == "risk":
                continue
            rsp[name] = {w: pool(*zip(*[(cell(table("predictions_response", t), rk)["estimate"],
                                         cell(table("predictions_response", t), rk)["boot_sd"]) for t in tl]))
                         for w, tl in (("wave1", W1), ("wave2", W2), ("all", ALL))}
        # response over risk: same units; unpaired difference of values (bootstrap SDs combined), descriptive
        ror = []
        for t in ALL:
            a_, b_ = cell(table("predictions_response", t), "risk"), cell(table("predictions", t), "risk")
            ror.append((a_["estimate"] - b_["estimate"], float(np.sqrt(a_["boot_sd"] ** 2 + b_["boot_sd"] ** 2))))
        rsp["response_over_risk_unpaired"] = pool(*zip(*ror))
        rsp["per_trial"] = {t: {k2: {kk: cell(table("predictions_response", t), rk)[kk] for kk in ("estimate", "lo95", "hi95")}
                                for rk, k2 in (("scale-minus-risk", "scale_over_response"), ("benefit-minus-risk", "dr_over_response"))}
                            for t in ALL}
        q3["response"] = rsp
    DOMAIN = {"gusto": "medicine", "ist": "medicine", "synbiotic": "medicine", "elaia2": "medicine",
              "voter": "voting", "sf_email": "voting", "sixcity": "voting", "newhaven": "voting", "w2_robocall2008": "voting",
              "w2_sanmateo_email": "voting", "w2_wisconsin_recall": "voting", "hillstrom": "retail", "lenta": "retail",
              "x5": "retail"}
    def pool_set(set_name, rk, trials):
        rows = [cell(table(set_name, t), rk) for t in trials if table(set_name, t) is not None]
        return pool([r["estimate"] for r in rows], [r["boot_sd"] for r in rows]) if len(rows) >= 2 else None
    # Each learned ranking over the scale ranking (paired, from its own scorer output).
    q3["learners_over_scale"] = {k: pool_set(s_, "benefit-minus-scale", ALL) for k, s_ in SETS.items()}
    # Paired comparisons with the response ranking.
    vs = {"dr": "predictions_response", "causal_forest": "predictions_cforest_vs_resp",
          "causalpfn": "predictions_causalpfn_vs_resp", "rose": "predictions_rose_vs_resp"}
    if all(table(s_, t) is not None for s_ in vs.values() for t in ALL) and all(table("predictions_resp_vs_risk", t) for t in ALL):
        q3["learners_over_response"] = {k: pool_set(s_, "benefit-minus-risk", ALL) for k, s_ in vs.items()}
        rr = {"all": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", ALL),
              "wave1": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", W1),
              "wave2": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", W2),
              "higher_is_better": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", [t for t in ALL if dt[t]["better"] == "higher"]),
              "lower_is_better": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", [t for t in ALL if dt[t]["better"] == "lower"])}
        rr["wave2_minus_wave1"] = wave_diff("predictions_resp_vs_risk", "benefit-minus-risk")
        rr["per_trial"] = {t: {kk: cell(table("predictions_resp_vs_risk", t), "benefit-minus-risk")[kk] for kk in ("estimate", "lo95", "hi95")} for t in ALL}
        q3["response_over_risk_paired"] = rr
        q3["dr_over_response_wave_diff"] = wave_diff("predictions_response", "benefit-minus-risk")
        # domain-stratified pools
        dom = {}
        for d_ in ("medicine", "voting", "retail", "other"):
            tl = [t for t in ALL if DOMAIN.get(t, "other") == d_]
            dom[d_] = {"trials": tl, "dr_over_risk": pool_set("predictions", "benefit-minus-risk", tl),
                       "scale_over_risk": pool_set("predictions", "scale-minus-risk", tl),
                       "response_over_risk": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", tl),
                       "dr_over_scale": pool_set("predictions", "benefit-minus-scale", tl),
                       "dr_over_response": pool_set("predictions_response", "benefit-minus-risk", tl)}
        q3["by_domain"] = dom
        # Lenta's weight in the pools against the response ranking, and pools without it
        nl = [t for t in ALL if t != "lenta"]
        def wshare(set_name, rk):
            rows = {t: cell(table(set_name, t), rk) for t in ALL}
            e = np.array([r["estimate"] for r in rows.values()]); sd = np.array([r["boot_sd"] for r in rows.values()])
            keep = ~((sd < 1e-9) & (np.abs(e) < 1e-9))
            v_ = np.maximum(sd ** 2, mr.V_FLOOR)
            f_ = mr.fit_re(e[keep], np.ones((keep.sum(), 1)), v_[keep])
            w_ = 1 / (v_ + f_.tau2)
            idx = list(rows).index("lenta")
            return float(w_[idx] / w_[keep].sum())
        q3["lenta_check"] = {"scale_over_response": {"weight": wshare("predictions_response", "scale-minus-risk"),
                                                     "without": pool_set("predictions_response", "scale-minus-risk", nl)},
                             "dr_over_response": {"weight": wshare("predictions_response", "benefit-minus-risk"),
                                                  "without": pool_set("predictions_response", "benefit-minus-risk", nl)}}
    # --- Final-panel additions (all from posted scorer outputs; nothing reads the sealed split) ---
    def pool_keep(e, sd):
        """Random-effects pool that keeps zero-variance trials at the registered floor (as the Q2 fits do)."""
        e, sd = np.asarray(e, float), np.asarray(sd, float)
        v = np.maximum(sd ** 2, mr.V_FLOOR)
        f = mr.fit_re(e, np.ones((len(e), 1)), v)
        se = float(np.sqrt(f.vcov[0, 0])); c = stats.t.ppf(0.975, f.df)
        return {"k": len(e), "mean": float(f.beta[0]), "lo95": float(f.beta[0] - c * se), "hi95": float(f.beta[0] + c * se),
                "tau": float(np.sqrt(f.tau2))}
    sr_rows = [cell(table("predictions", t), "scale-minus-risk") for t in ALL]
    q3["scale_over_risk"]["all_22_zero_gap_at_floor"] = pool_keep([r["estimate"] for r in sr_rows], [r["boot_sd"] for r in sr_rows])
    ew_m = float(np.mean([r["estimate"] for r in sr_rows]))
    ew_se = float(np.sqrt(sum(r["boot_sd"] ** 2 for r in sr_rows)) / len(sr_rows))
    ew_c = stats.t.ppf(0.975, len(sr_rows) - 1)
    q3["scale_over_risk"]["equal_weight_mean_22"] = {"k": len(sr_rows), "mean": ew_m, "se": ew_se,
                                                     "lo95": ew_m - ew_c * ew_se, "hi95": ew_m + ew_c * ew_se}
    # Each learned ranking over the scale and response rankings at every budget (same test units).
    q3["budgets_learners"] = {}
    for b in (0.1, 0.2, 0.3):
        bb = {"over_scale": {k: pool(*zip(*[(cell(table(s_, t), "benefit-minus-scale", b)["estimate"],
                                            cell(table(s_, t), "benefit-minus-scale", b)["boot_sd"]) for t in ALL]))
                             for k, s_ in SETS.items()}}
        if "learners_over_response" in q3:
            bb["over_response"] = {k: pool(*zip(*[(cell(table(s_, t), "benefit-minus-risk", b)["estimate"],
                                                   cell(table(s_, t), "benefit-minus-risk", b)["boot_sd"]) for t in ALL]))
                                   for k, s_ in vs.items()}
            bb["response_over_risk"] = pool(*zip(*[(cell(table("predictions_resp_vs_risk", t), "benefit-minus-risk", b)["estimate"],
                                                    cell(table("predictions_resp_vs_risk", t), "benefit-minus-risk", b)["boot_sd"]) for t in ALL]))
            bb["scale_over_response"] = pool(*zip(*[(cell(table("predictions_response", t), "scale-minus-risk", b)["estimate"],
                                                     cell(table("predictions_response", t), "scale-minus-risk", b)["boot_sd"]) for t in ALL]))
        q3["budgets_learners"][str(b)] = bb
    # Wave-1 prediction interval as a forecast of each learned ranking's wave-2 per-trial gains.
    q3["wave1_pi_forecast_wave2"] = {}
    for k in SETS:
        pi = q3[k]["wave1"]["pi95"]
        w2v = {t: v["estimate"] for t, v in q3[k]["per_trial"].items() if t.startswith("w2_")}
        q3["wave1_pi_forecast_wave2"][k] = {"pi95_wave1": pi, "n_wave2": len(w2v),
                                            "n_inside": int(sum(pi[0] <= v <= pi[1] for v in w2v.values())),
                                            "outside": {t: v for t, v in w2v.items() if not pi[0] <= v <= pi[1]},
                                            "wave2_mean_minus_wave1_center": float(q3[k]["wave2"]["mean"] - q3[k]["wave1"]["mean"])}
    # Wave difference within the voting trials only (the one domain present in both waves with >= 3 trials each).
    def wave_diff_on(set_name, rk, trials):
        rows = [(t, cell(table(set_name, t), rk)) for t in trials]
        rows = [(t, r) for t, r in rows if not (r["boot_sd"] < 1e-9 and abs(r["estimate"]) < 1e-9)]
        ee = np.array([r["estimate"] for _, r in rows]); vv = np.maximum(np.array([r["boot_sd"] for _, r in rows]) ** 2, mr.V_FLOOR)
        w2v = np.array([1.0 if t.startswith("w2_") else 0.0 for t, _ in rows])
        fw = mr.fit_re(ee, np.column_stack([np.ones_like(w2v), w2v]), vv)
        se, c = float(np.sqrt(fw.vcov[1, 1])), stats.t.ppf(0.975, fw.df)
        return {"k": len(rows), "est": float(fw.beta[1]), "lo95": float(fw.beta[1] - c * se), "hi95": float(fw.beta[1] + c * se)}
    voting = [t for t in ALL if DOMAIN.get(t) == "voting"]
    q3["voting_only_wave_diff"] = {"dr_over_risk": wave_diff_on("predictions", "benefit-minus-risk", voting),
                                   "scale_over_risk": wave_diff_on("predictions", "scale-minus-risk", voting)}
    if "by_domain" in q3:
        nv = [t for t in ALL if DOMAIN.get(t, "other") != "voting"]
        q3["by_domain"]["non_voting"] = {"trials": nv, "dr_over_risk": pool_set("predictions", "benefit-minus-risk", nv),
                                         "scale_over_risk": pool_set("predictions", "scale-minus-risk", nv),
                                         "response_over_risk": pool_set("predictions_resp_vs_risk", "benefit-minus-risk", nv),
                                         "dr_over_scale": pool_set("predictions", "benefit-minus-scale", nv),
                                         "dr_over_response": pool_set("predictions_response", "benefit-minus-risk", nv)}
    # Risk ranking's 20% policy value against the trial's average effect (TOC(0.2) < 0), per trial.
    q3["autoc"]["n_risk_q20_below_ate"] = int(sum(cell(table("predictions", t), "risk")["estimate"] < table("predictions", t)["ate_per_1000"]
                                                 for t in ALL))
    # Holm-adjusted P values for the family of four abstract-level claims (Knapp–Hartung t, df = k - 1).
    def p_from(r):
        c = stats.t.ppf(0.975, r["k"] - 1); se = (r["hi95"] - r["lo95"]) / (2 * c)
        return float(2 * stats.t.sf(abs(r["mean"] / se), r["k"] - 1))
    fam = {"risk_autoc": q3["autoc"]["risk"], "scale_over_risk_17": q3["scale_over_risk"]["all"],
           "dr_over_risk_22": q3["dr"]["all"]}
    if "response_over_risk_paired" in q3:
        fam["response_over_risk_22"] = q3["response_over_risk_paired"]["all"]
    raw = {k: p_from(r) for k, r in fam.items()}
    order = sorted(raw, key=raw.get); m = len(raw); holm = {}; run = 0.0
    for i, k in enumerate(order):
        run = max(run, (m - i) * raw[k]); holm[k] = min(1.0, run)
    q3["holm_family"] = {k: {"p": raw[k], "p_holm": holm[k]} for k in raw}
    out["Q3"] = q3
    # Q4
    gc = {}
    for t in ALL:
        p = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"predictions_rose_v2__{t}__contract.json")))
        if p:
            gc[t] = next(iter(json.loads(Path(p[-1]).read_text())["trials"].values()))["group_coverage_share"]
    hv = json.loads((ROOT / "scoring" / "rose_v2" / "rose_v2_harm.json").read_text())["trials"]
    uc = {}
    for t in RB:
        p = sorted(glob.glob(str(ROOT / "scoring" / "rose_v2" / f"predictions_rose_v2__{t}__contract.json")))[-1]
        uc[t] = next(iter(json.loads(Path(p).read_text())["trials"].values()))["unit_coverage_90"]
    detail = json.loads((R / "rose_interval_detail.json").read_text())
    out["Q4"] = {"interval_detail": detail,"group_coverage_by_trial": gc, "n_trials_full_coverage": int(sum(v == 1.0 for v in gc.values())),
                 "unit_coverage": uc,
                 "harm_brier": {t: {"rose": hv[t]["brier_rose_v1_p_harm_ind"], "constant": hv[t]["brier_constant_dev_rate"],
                                    "harm_rate": hv[t]["harm_rate"]} for t in RB}}
    (R / "final_results.json").write_text(json.dumps(out, indent=1, default=float))
    p = q2["primary"]
    print(f"Q1 {out['Q1']['condition_matches_posted_gap']}/{out['Q1']['eligible']} zeros={zeros}")
    print(f"Q2 k={q2['k']} slope {p['slope']:.2f} ({p['lo95']:.2f},{p['hi95']:.2f}) p={p['p_two_sided']:.3f} | informative k={q2['informative_only']['k']} "
          f"{q2['informative_only']['slope']:.2f} ({q2['informative_only']['lo95']:.2f},{q2['informative_only']['hi95']:.2f}) | "
          f"interaction {q2['slope_by_wave_interaction']['coef']:.2f} (SE {q2['slope_by_wave_interaction']['se']:.2f}) | driven_by {q2['driven_by']} | "
          f"LOO {q2['loo_lpd_gain_nats']:.2f} (inf {q2['loo_lpd_gain_informative_only']:.2f}) | placebo gap coef {q2['size_placebo_both']['coef_gap']:.2f} "
          f"({q2['size_placebo_both']['lo95']:.2f},{q2['size_placebo_both']['hi95']:.2f})")
    for k in SETS:
        r = q3[k]
        print(k, {w: f"{r[w]['mean']:.1f} ({r[w]['lo95']:.1f},{r[w]['hi95']:.1f})" for w in ("wave1", "wave2", "all") if w in r},
              "tau", round(r["all"]["tau"], 1), "PI", [round(z, 1) for z in r["all"]["pi95"]], "w2-w1", round(r["wave2_minus_wave1"]["est"], 1))
    print("scale_over_risk", {w: round(v["mean"], 1) for w, v in q3["scale_over_risk"].items() if "mean" in v}, "dr_over_scale",
          {w: f"{v['mean']:.1f} ({v['lo95']:.1f},{v['hi95']:.1f})" for w, v in q3["dr_over_scale"].items() if "mean" in v})
    print("Q4 full coverage", out["Q4"]["n_trials_full_coverage"], "of", len(gc))


if __name__ == "__main__":
    main()
