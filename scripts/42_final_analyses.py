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
    e, v = np.asarray(e, float), np.maximum(np.asarray(sd, float) ** 2, mr.V_FLOOR)
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
    inf = np.array([t not in zeros for t in elig])
    _, q2["informative_only"] = slope(x[inf], y[inf], v[inf])
    q2["informative_only"]["k"] = int(inf.sum())
    fi = mr.fit_re(y, np.column_stack([np.ones_like(x), x, wv, x * wv]), v)
    q2["slope_by_wave_interaction"] = {"coef": float(fi.beta[3]), "se": float(np.sqrt(fi.vcov[3, 3])),
                                      "p": float(2 * stats.t.sf(abs(fi.beta[3] / np.sqrt(fi.vcov[3, 3])), fi.df))}
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
        res["wave2_minus_wave1"] = {"est": float(fw.beta[1]), "se": float(np.sqrt(fw.vcov[1, 1]))}
        res["n_ci_above_0"] = int(sum(r["lo95"] > 0 for r in rows.values()))
        res["n_ci_below_0"] = int(sum(r["hi95"] < 0 for r in rows.values()))
        q3[k] = res
    for rk, name in (("scale-minus-risk", "scale_over_risk"), ("benefit-minus-scale", "dr_over_scale")):
        q3[name] = {w: pool(*zip(*[(cell(table("predictions", t), rk)["estimate"], cell(table("predictions", t), rk)["boot_sd"])
                                   for t in tl])) for w, tl in (("wave1", W1), ("wave2", W2), ("all", ALL))}
    llm = [cell(table("predictions_llm", t), "benefit-minus-risk") for t in W1]
    q3["language_model_wave1"] = pool([r["estimate"] for r in llm], [r["boot_sd"] for r in llm])
    q3["language_model_masked_wave1"] = json.loads((R / "llm_masked_compare.json").read_text())["pooled_masked"]
    dp = [cell(table("predictions_dopfn", t), "benefit-minus-risk") for t in W1]
    q3["dopfn_wave1"] = pool([r["estimate"] for r in dp], [r["boot_sd"] for r in dp])
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
    out["Q4"] = {"group_coverage_by_trial": gc, "n_trials_full_coverage": int(sum(v == 1.0 for v in gc.values())),
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
    print("scale_over_risk", {w: round(v["mean"], 1) for w, v in q3["scale_over_risk"].items()}, "dr_over_scale",
          {w: f"{v['mean']:.1f} ({v['lo95']:.1f},{v['hi95']:.1f})" for w, v in q3["dr_over_scale"].items()})
    print("Q4 full coverage", out["Q4"]["n_trials_full_coverage"], "of", len(gc))


if __name__ == "__main__":
    main()
