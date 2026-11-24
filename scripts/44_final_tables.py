"""Main Tables 1-3 and SI Tables S1-S8, written from results files only.

Output: notebooks/system-three/paper/tables/main_tables.md and si_tables.md.
Descriptive labels (trial names, domains, actions, outcomes) are fixed text; every number is read
from results/final_results.json, results/dataset_table.json, results/theory_checks.json,
results/llm_masked_compare.json, results/proportionality_diagnostic*.json, or the scorer outputs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT.parents[1] / "notebooks" / "system-three" / "paper" / "tables"
FR = json.loads((R / "final_results.json").read_text())
DT = json.loads((R / "dataset_table.json").read_text())
W1, W2 = FR["trials"]["wave1"], FR["trials"]["wave2"]
SCALE_PT = {t: {"estimate": p["y"], "lo95": p["lo95"], "hi95": p["hi95"]} for t, p in FR["Q2"]["points"].items()}

# name, domain, action versus reference, outcome (direction)
META = {
    "gusto": ("GUSTO-I", "Medicine", "Accelerated alteplase vs streptokinase", "Death by 30 days (lower)"),
    "ist": ("International Stroke Trial", "Medicine", "Aspirin vs no aspirin", "Death or dependency at 6 months (lower)"),
    "synbiotic": ("Neonatal synbiotic trial", "Medicine", "Oral synbiotic vs placebo", "Sepsis or death by 60 days (lower)"),
    "elaia2": ("ELAIA-2", "Medicine", "Electronic kidney-injury alert vs usual care",
               "Progression, dialysis, or death by 14 days (lower)"),
    "voter": ("Michigan social-pressure mailer", "Voting", "Neighbors mailer vs no mail", "Voted (higher)"),
    "sf_email": ("San Francisco registrar email", "Voting", "Email vs no email", "Voted (higher)"),
    "karlan_list": ("Karlan–List matching grant", "Charity", "Matching-grant letter vs standard letter", "Gave (higher)"),
    "stepup": ("StepUp exercise megastudy", "Exercise", "Behavioral program vs control", "Any gym visit, weeks 1–4 (higher)"),
    "hillstrom": ("Hillstrom email", "Retail", "Email vs no email", "Website visit in 2 weeks (higher)"),
    "sixcity": ("Six-city canvass", "Voting", "Door-to-door canvass vs none", "Voted (higher)"),
    "newhaven": ("New Haven canvass", "Voting", "Door-to-door canvass vs none", "Voted (higher)"),
    "lenta": ("Lenta text message", "Retail", "Text message vs none", "Store visit (higher)"),
    "x5": ("X5 text message", "Retail", "Text message vs none", "Purchase (higher)"),
    "w2_robocall2008": ("Michigan robocall", "Voting", "Recorded call vs none", "Voted (higher)"),
    "w2_sanmateo_email": ("San Mateo registrar email", "Voting", "Email vs no email", "Voted (higher)"),
    "w2_wisconsin_recall": ("Wisconsin ballot-secrecy letter", "Voting", "Letter vs none", "Voted (higher)"),
    "w2_ffh_charity": ("Freedom from Hunger mailing", "Charity", "Research insert vs standard mailing", "Gave (higher)"),
    "w2_sa_loan": ("South Africa loan offer", "Consumer credit", "One example loan vs several", "Took up loan (higher)"),
    "w2_mooc_megastudy": ("edX plan-making", "Education", "Plan-making prompt vs control", "Earned certificate (higher)"),
    "w2_jsa_florida": ("Florida job-search assistance", "Employment", "Structured job-search assistance vs usual services",
                       "Exhausted benefits in year 1 (lower)"),
    "w2_jtpa": ("Job Training Partnership Act", "Employment", "Access to training vs none", "Any earnings by 30 months (higher)"),
    "w2_turkey_overdraft": ("Turkey overdraft message", "Consumer finance", "Interest-discount message vs none",
                            "Used overdraft (lower)"),
}
RBL = {"routerbench__mistral7b_to_gpt4": "Mistral-7B to GPT-4", "routerbench__gpt35_to_gpt4": "GPT-3.5 to GPT-4",
       "routerbench__llama70b_to_claude2": "Llama-2-70B to Claude 2", "routerbench__mixtral_to_gpt4": "Mixtral to GPT-4"}
LEARNER = {"scale_over_risk": "Scale ranking", "dr": "Doubly robust learner", "causal_forest": "Causal forest",
           "causalpfn": "CausalPFN", "rose": "Calibrated ensemble"}


def ci(e, lo, hi, d=1):
    e, lo, hi = (0.0 if round(v, d) == 0 else v for v in (e, lo, hi))
    return f"{e:.{d}f} ({lo:.{d}f} to {hi:.{d}f})"


def v10(x):
    v = int(round(10.0 * x))
    return f"{0 if v == 0 else v:,}"


def ci10(e, lo, hi):
    return f"{v10(e)} ({v10(lo)} to {v10(hi)})"


def n(x):
    return f"{int(round(x)):,}"


def pooled(r, d=1):
    return ci(r["mean"], r["lo95"], r["hi95"], d)


def pooled10(r):
    return ci10(r["mean"], r["lo95"], r["hi95"])


def trial_table() -> list:
    q2 = FR["Q2"]
    elig = set(q2["points"])
    gap = {t: p["x"] for t, p in q2["points"].items()}
    o = ["## Table S1. The 22 randomized trials\n",
         "| Trial | Wave | Domain | Action vs reference | Outcome (better direction) | Individuals in the trial file (all arms) | Test units | Scale-predicted gap, per 10,000 reached | In forecast test |",
         "|---|---|---|---|---|---|---|---|---|"]
    for t in W1 + W2:
        nm, dom, act, out = META[t]
        g = v10(gap[t]) if t in elig else "–"
        o.append(f"| {nm} | {2 if t.startswith('w2_') else 1} | {dom} | {act} | {out} | {n(DT[t]['rows'])} | {n(DT[t]['test'])} | "
                 f"{g} | {'Yes' if t in elig else 'No (effect not beneficial in context data)'} |")
    o.append(f"\nTest units total {n(FR['n_test_units']['wave1'] + FR['n_test_units']['wave2'])} "
             f"(wave 1, {n(FR['n_test_units']['wave1'])}; wave 2, {n(FR['n_test_units']['wave2'])}). GUSTO-I compares accelerated "
             "alteplase with the two streptokinase arms combined; the combination arm is not used. The International Stroke Trial "
             "compares aspirin with no aspirin; heparin allocation, randomized independently, is kept as a covariate. StepUp pools "
             "all 53 behavioral program arms against placebo control using the within-cohort randomization share; Karlan–List pools "
             "the 1:1, 2:1, and 3:1 matching-grant arms (two-thirds of letters by design) against the standard no-match letter. The "
             "scale-predicted gap is in people helped per 10,000 reached (meaning per 10,000 people assigned to the action under the "
             "budget) and is defined only where the context effect is beneficial. Figure labels use short forms of these names.\n")
    return o


def main_tables() -> str:
    q2, q3 = FR["Q2"], FR["Q3"]
    o = []
    p = q2["primary"]
    io, o16 = q2["informative_only"], q2["informative_through_origin"]
    rows = [("Wave 1 (registered primary)", q2["k_wave1"], q2["wave1_only"]),
            ("Wave 1, without the 2 trials whose forecast is 0 by construction", q2["wave1_informative_only"]["k"], q2["wave1_informative_only"]),
            ("Wave 2 (registered primary; descriptive, fewer than 8 trials)", q2["k_wave2"], q2["wave2_only"]),
            ("Both waves (registered secondary)", q2["k"], p),
            ("Both waves, without the 3 trials whose forecast is 0 by construction", io["k"], io),
            ("Same 16 trials, line through the origin (not pre-specified)", o16["k"], o16)]
    o += ["## Table 1. Does the scale-predicted gap forecast the observed gain of the scale ranking over the risk ranking?\n",
          "| Analysis | Trials | Slope (95% CI) | P | Between-trial SD |", "|---|---|---|---|---|"]
    for name, k, r in rows:
        tau = "–" if r.get("tau2") is None else v10(r["tau2"] ** 0.5)
        o.append(f"| {name} | {k} | {ci(r['slope'], r['lo95'], r['hi95'], 2)} | {r['p_two_sided']:.3f} | {tau} |")
    b1 = json.loads((R / "ben_exploratory_analyses.json").read_text())
    ess_w1 = b1["q2_power_and_baselines"]["ess_wave1_13"]
    ess_all = b1["q2_power_and_baselines"]["ess_all_19"]
    o.append(f"\nSlope 1 means that observed gains equal forecast gains; slope 0 means no relation. The between-trial SD is the square root "
             "of the restricted maximum likelihood estimate of the between-trial variance, in people helped per 10,000 reached; the trials whose forecast "
             "is 0 by construction have zero residual and the floor variance, so the fits that include them estimate this SD as 0 and "
             "behave as fixed-effect fits. In the 19-trial fit, the 3 trials whose "
             f"forecast is 0 by construction carry {100 * q2['weight_share_zero_gap_trials']:.1f}% of the weight (Kish effective sample size {ess_all:.1f}); "
             f"in the wave-1 fit, the 2 such trials carry {100 * q2['weight_share_zero_gap_trials_wave1']:.1f}% (effective sample size {ess_w1:.1f}). "
             "In each of these trials the two rankings selected the same test units, "
             "so the observed gain was 0 with no sampling variance. SI Appendix, Table S9 gives the registered "
             "sensitivity analyses, and Table S17 gives forecast accuracy against naive baselines.\n")
    o += ["\n## Table 2. Do the scale ranking, the response ranking, and learned benefit rankings select people with larger effects than the risk ranking?\n",
          "*A. Gain over the risk ranking*\n",
          "| Ranking | Wave 1 | Wave 2 | All 22 trials | 17 trials in which scale and risk rankings differ | Wave 2 minus wave 1 | 95% prediction interval, new trial |",
          "|---|---|---|---|---|---|---|"]
    def wd(r):
        w_ = r.get("wave2_minus_wave1") if isinstance(r, dict) else None
        return ci10(w_["est"], w_["lo95"], w_["hi95"]) if w_ and "lo95" in w_ else "–"
    pi_ = lambda r: f"{v10(r['pi95'][0])} to {v10(r['pi95'][1])}"  # noqa: E731
    sr = q3["scale_over_risk"]
    ew = sr["equal_weight_mean_22"]
    o.append(f"| Scale ranking | {pooled10(sr['wave1'])} [{sr['wave1']['k']}] | {pooled10(sr['wave2'])} [{sr['wave2']['k']}] | "
             f"{pooled10(ew)}^a^ | {pooled10(sr['all'])} | {wd(sr)} | {pi_(sr['all'])}^b^ |")
    rr = q3.get("response_over_risk_paired")
    if rr:
        o.append(f"| Response ranking | {pooled10(rr['wave1'])} [{rr['wave1']['k']}] | {pooled10(rr['wave2'])} [{rr['wave2']['k']}] | "
                 f"{pooled10(rr['all'])} | – | {wd(rr)} | {pi_(rr['all'])} |")
    for k, name in LEARNER.items():
        if k == "scale_over_risk":
            continue
        r, r17 = q3[k], q3["learned_on_scale_trials"][k]
        o.append(f"| {name} | {pooled10(r['wave1'])} [{r['wave1']['k']}] | {pooled10(r['wave2'])} [{r['wave2']['k']}] | "
                 f"{pooled10(r['all'])} | {pooled10(r17)} | {wd(r)} | {pi_(r['all'])} |")
    dp = q3["dopfn_wave1"]
    o.append(f"| Do-PFN | {pooled10(dp)} [{dp['k']}] | – | – | – | – | – |")
    o += ["\n*B. Gain of each learned ranking over the two simple alternatives, all 22 trials, same test units*\n",
          "| Learned ranking | Over the scale ranking | 95% prediction interval, new trial | Over the response ranking | 95% prediction interval, new trial |",
          "|---|---|---|---|---|"]
    lr_ = q3.get("learners_over_response", {})
    for k, name in LEARNER.items():
        if k == "scale_over_risk":
            continue
        a_ = q3["learners_over_scale"][k]
        b_ = lr_.get(k)
        o.append(f"| {name} | {pooled10(a_)} | {pi_(a_)} | {pooled10(b_) if b_ else '–'} | {pi_(b_) if b_ else '–'} |")
    fl = sr["all_22_zero_gap_at_floor"]
    hf = q3["holm_family"]
    ho = ["risk_autoc", "scale_over_risk_17", "response_over_risk_22", "dr_over_risk_22"]
    holm_s = ", ".join(f"{hf[k]['p_holm']:.3f}" for k in ho)
    raw_s = ", ".join(f"{hf[k]['p']:.3f}" for k in ho)
    o.append("\nValues are gains at a 20% budget, in people helped per 10,000 reached (95% CI) [number of trials], pooled by random "
             "effects; panel B compares each learned ranking with the scale or response ranking on the same units. Pools leave out trials in "
             "which the two compared rankings select the same units. The wave difference is the wave coefficient of one random-effects "
             "meta-regression over all trials, with a common between-trial variance, so it is not the difference of the two wave cells. "
             "Do-PFN was run in wave 1 only. Intervals are not adjusted for multiple comparisons; the doubly robust learner is the "
             "registered learned ranking in both waves. Pooling over both waves was not pre-specified. The prediction interval applies to a new trial "
             "exchangeable with these trials, and the pooled values average absolute differences in outcomes that differ between trials. "
             f"^a^Equal-weighted mean over all 22 trials with the 5 trials in which the two rankings selected the same units entered as 0 "
             f"(SE {v10(ew['se'])} from the per-trial bootstrap variances; t interval with 21 degrees of freedom). An inverse-variance pool that "
             f"enters those 5 trials at the registered variance floor gives {pooled10(fl)}, because they then carry nearly all the weight and set "
             f"the between-trial variance to 0; the 17-trial pool is conditional on the two rankings differing. ^b^From the 17-trial pool. "
             f"Where the estimated between-trial SD is 0 (the wave-2 pools of the doubly robust learner, the causal forest, and CausalPFN, "
             f"and the causal forest and CausalPFN over the scale ranking), the pool is a fixed-effect pool and the prediction interval equals "
             f"the confidence interval. Holm-adjusted P values for the four pooled claims against the risk ranking in Results, question 3 (its "
             f"rank-weighted average treatment effect, and the gains of the scale, response, and doubly robust rankings over it, in that order) were "
             f"{holm_s} (unadjusted {raw_s}).\n")
    return "\n".join(o)


def scored(set_name, t):
    ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"{set_name}__{t}.json")))
    return next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values())) if ps else None


def si_tables() -> str:
    q2, q3, q4 = FR["Q2"], FR["Q3"], FR["Q4"]
    o = ["## Table S1. Reproduction of each trial's published estimate before the test split was sealed\n",
         "| Trial | Published quantity | Published | Reproduced | Prespecified tolerance | Result | Terms of use |",
         "|---|---|---|---|---|---|---|"]
    for t in W1 + W2:
        r = DT[t]["reproduction"]
        fmt = (lambda v: f"{int(round(v)):,}") if r["metric"] == "n_units" else (lambda v: f"{v:.4f}".rstrip("0").rstrip("."))
        pub = "–" if r["published"] is None else fmt(r["published"])
        tol = "–" if r.get("tol") is None else f"{r['tol']:.4g}"
        res = "Within tolerance" if r["status"] == "match" else "No published target in the public file"
        metric = {"diff": "Difference in proportions", "n_units": "Analysis sample size", "relative_change": "Relative change",
                  "diff_strata": "Difference in proportions", "treated_share": "Share assigned to action"}.get(r["metric"], r["metric"])
        o.append(f"| {META[t][0]} | {metric} | {pub} | {fmt(r['reproduced'])} | {tol} | {res} | {DT[t]['license']} |")
    o.append("\nTolerances were fixed before loading, from the precision of the published value. For edX plan-making, Freedom from Hunger, "
             "and Turkey overdraft, the tolerance exceeds the published effect, so the check confirms the sample and the order of magnitude "
             "only.\n")
    o += ["\n## Table S2. Leave-one-trial-out slope and forecast gain\n",
          "| Trial left out | Slope without it (95% CI) | Gain in log predictive density, nats |", "|---|---|---|"]
    for t, r in q2["leave_one_out"].items():
        o.append(f"| {META[t][0]} | {ci(r['slope'], r['lo95'], r['hi95'], 2)} | {r['lpd_gain'] + 0.0 if abs(r['lpd_gain']) >= 0.005 else 0.0:.2f} |")
    oos = q2["wave2_out_of_sample"]
    o += ["\n## Table S3. Wave-1 fit as a predictor of the eligible wave-2 trials (people helped per 10,000 reached)\n",
          "| Trial | Forecast (95% prediction interval) | Observed gain | Gain in log predictive density, nats |", "|---|---|---|---|"]
    for t, r in oos["per_trial"].items():
        o.append(f"| {META[t][0]} | {ci10(r['forecast'], *r['pi95'])} | {v10(r['observed'])} | {r['lpd_gain'] + 0.0 if abs(r['lpd_gain']) >= 0.005 else 0.0:.2f} |")
    o.append(f"\nTotal gain {oos['lpd_gain_nats']:.2f} nats; {oos['n_inside_pi95']} of {oos['k']} observed gains inside the 95% prediction interval.\n")
    o += ["\n## Table S4. Model routing, where each prompt's effect is known (prompts improved per 10,000 routed)\n",
          "| Model pair (smaller to larger) | Test prompts | Scale-predicted gap | Scale over risk (95% CI) | Doubly robust over risk (95% CI) | Interaction of action with logit risk (95% CI) | Diagnostic flag |",
          "|---|---|---|---|---|---|---|"]
    for t, r in q2["routing"].items():
        a, b, it = r["scale_minus_risk"], r["dr_minus_risk"], r["interaction"]
        nm = RBL.get(t, "SWE-bench, " + t.split("__")[1].replace("_to_", " to ").replace("p", "agent percentile "))
        its = ci(it["delta"], it["delta_lo95"], it["delta_hi95"], 2) if it else "–"
        fl = "–" if r["flag"] is None else ("Yes" if r["flag"] else "No")
        o.append(f"| {nm} | {n(r['n_test'])} | {v10(r['forecast'])} | {ci10(a['estimate'], a['lo95'], a['hi95'])} | "
                 f"{ci10(b['estimate'], b['lo95'], b['hi95'])} | {its} | {fl} |")
    o.append("\nThe three SWE-bench pairs (coding agents at three skill percentiles, 148 test tasks each) were scored but are too small to "
             "inform any comparison; in each, the scale and risk rankings selected the same test tasks. In two of them the context-split forecast was positive, but "
             "on 148 test tasks with few distinct risk values the practical check (Note 1, Proposition 2) did not hold, so the selected sets coincided.\n")
    o += ["\n## Table S5. Gain of each ranking over the risk ranking, by trial (people helped per 10,000 reached, 95% CI)\n",
          "| Trial | " + " | ".join(LEARNER.values()) + " |", "|---|" + "---|" * len(LEARNER)]
    for t in W1 + W2:
        cells = []
        for k in LEARNER:
            if k == "scale_over_risk":
                c_ = next(x for x in scored("predictions", t)["table"] if x["ranking"] == "scale-minus-risk" and x["budget"] == 0.2)
                r = {"estimate": c_["estimate"] + 0.0, "lo95": c_["lo95"] + 0.0, "hi95": c_["hi95"] + 0.0}
            else:
                r = q3[k]["per_trial"].get(t)
            cells.append(ci10(r["estimate"], r["lo95"], r["hi95"]) if r else "–")
        o.append(f"| {META[t][0]} | " + " | ".join(cells) + " |")
    o += ["\n## Table S6. Uncertainty statements of the calibrated ensemble\n",
          "| Setting | Measure | Calibrated ensemble | Comparator |", "|---|---|---|---|"]
    for t in RB_ORDER:
        u = q4["unit_coverage"][t]
        h = q4["harm_brier"][t]
        o.append(f"| {RBL[t]} | Coverage of each prompt's effect by the 90% interval | {u:.3f} | Nominal 0.900 |")
        bd = q4["interval_detail"][t]["brier_diff_vs_constant"]
        o.append(f"| {RBL[t]} | Brier score, probability that the action harms the prompt | {h['rose']:.4f} | "
                 f"{h['constant']:.4f} (development-split harm rate) |")
        o.append(f"| {RBL[t]} | Brier difference, ensemble minus constant (95% bootstrap CI) | "
                 f"{ci(bd['estimate'], bd['lo95'], bd['hi95'], 4)} | 0 |")
    o.append(f"| 22 trials | Share of 10 predicted-effect bins whose doubly robust mean lies inside the 90% interval | "
             f"1.0 in {q4['n_trials_full_coverage']} of {len(q4['group_coverage_by_trial'])} trials | Nominal 0.9 |")
    m = json.loads((R / "llm_masked_compare.json").read_text())
    o += ["\n## Table S7. Language model, original and masked trial descriptions (wave 1; gain over risk ranking, people helped per 10,000 reached)\n",
          "| Trial | Original (95% CI) | Masked (95% CI) | Rank correlation of the two runs' effect estimates |",
          "|---|---|---|---|"]
    for t, r in m["per_trial"].items():
        o.append(f"| {META[t][0]} | {ci10(r['original'], *r['original_ci'])} | {ci10(r['masked'], *r['masked_ci'])} | "
                 f"{r['spearman_effects']:.2f} |")
    lm, lmm = q3["language_model_wave1"], q3["language_model_masked_wave1"]
    o.append(f"\nPooled: original {pooled10(lm)}; masked {ci10(lmm['mean'], lmm['lo95'], lmm['hi95'])}.\n")
    d = {**json.loads((R / "proportionality_diagnostic.json").read_text()),
         **json.loads((R / "proportionality_diagnostic_w2.json").read_text())}
    o += ["\n## Table S8. Context-only diagnostic of a constant odds ratio (gaps in people helped per 10,000 reached)\n",
          "| Trial | Interaction of action with logit risk (95% CI) | Scale-predicted gap | Gap predicted with the interaction | Flag |",
          "|---|---|---|---|---|"]
    for t in W1 + W2:
        r = d[t]
        ok = t in q2["points"]
        g1 = v10(r["gap_scale"]) if ok else "–"
        g2 = v10(r["gap_interaction"]) if ok else "–"
        o.append(f"| {META[t][0]} | {ci(r['delta'], r['delta_lo95'], r['delta_hi95'], 3)} | {g1} | {g2} | "
                 f"{('Yes' if r['flag_not_applicable'] else 'No') if ok else '–'} |")
    o += ["\n## Table S9. Registered sensitivity analyses of the forecast slope\n",
          "| Analysis | Trials | Slope (95% CI) | P |", "|---|---|---|---|"]
    rows = [("Budget 10%", q2["k"], q2["budget_0.1"]), ("Budget 30%", q2["k"], q2["budget_0.3"]),
            ("Weights from test-split bootstrap variances", q2["k"], q2["bootstrap_weights"]),
            ("Weights from design variances", q2["k"], q2["design_weights"])]
    kk = {"primary": (q2["k"], "both waves"), "informative_only": (q2["informative_only"]["k"], "16 informative trials"),
          "wave1_only": (q2["k_wave1"], "wave 1"), "wave2_only": (q2["k_wave2"], "wave 2"),
          "design_weights": (q2["k"], "weights from design variances"), "budget_0.3": (q2["k"], "budget 30%")}
    for nm_, r in q2["truncated_kh"].items():
        rows.append((f"Truncated Knapp–Hartung, {kk[nm_][1]}", kk[nm_][0], r))
    for name, k, r in rows:
        o.append(f"| {name} | {k if k else '–'} | {ci(r['slope'], r['lo95'], r['hi95'], 2)} | {r['p_two_sided']:.3f} |")
    sp, ia = q2["size_placebo_both"], q2["slope_by_wave_interaction"]
    o.append(f"| Adjusted for the predicted average effect | {q2['k']} | {ci(sp['coef_gap'], sp['lo95'], sp['hi95'], 2)} | – |")
    o.append(f"| Difference in slope, wave 2 minus wave 1 (common between-trial variance) | {q2['k']} | {ci(ia['coef'], ia['lo95'], ia['hi95'], 2)} | {ia['p']:.3f} |")
    o.append(f"| Regression-calibration correction for error in the forecast (reliability {q2['eiv']['reliability']:.2f}) | {q2['k']} | "
             f"{q2['eiv']['slope']:.2f} | – |")
    fz = q2["registered_falsifier_wave1"]
    o.append(f"\nJoint test of intercept 0 and slope 1 in 19 trials, P = {q2['primary']['joint_0_1']['p']:.3f}. Leave-one-trial-out "
             f"forecasts that used the gap improved on forecasts that ignored it by {q2['loo_lpd_gain_nats']:.2f} nats of log predictive density "
             f"over 19 trials and {q2['loo_lpd_gain_informative_only']:.2f} nats over the 16 informative trials. Registered wave-1 falsifier: "
             f"the slope CI excluded 0; the joint test gave P = {fz['strong_joint_test_p']:.3f}; forecasts of the 3 holdout trials, posted "
             f"before they were scored, gained {fz['holdout_lpd_gain_nats']:.2f} nats against a margin of {fz['margin_nats']:.0f} nat. The primary "
             "fit uses the untruncated Knapp–Hartung correction; the truncated version is the registered sensitivity analysis.\n")
    au = q3["autoc"]
    o += ["\n## Table S10. Rank-weighted average treatment effects (AUTOC and Qini) and comparison with random selection\n",
          "| Ranking | Pooled AUTOC per 10,000 reached (95% CI) | Pooled Qini per 10,000 reached (95% CI) |", "|---|---|---|"]
    for k_, nm_ in (("risk", "Risk ranking"), ("scale", "Scale ranking"), ("benefit", "Doubly robust learner"),
                    ("causal_forest", "Causal forest"), ("causalpfn", "CausalPFN"), ("rose", "Calibrated ensemble")):
        qn = au.get(f"{k_}_qini")
        o.append(f"| {nm_} | {pooled10(au[k_])} | {pooled10(qn) if qn else '–'} |")
    from math import comb
    n15 = au["n_risk_q20_below_ate"]
    p_sign = min(1.0, 2 * sum(comb(22, i) for i in range(n15, 23)) / 2 ** 22)
    o.append(f"\nAll values pool 22 trials. A negative value means that the ranking puts people with smaller-than-average effects first. "
             f"The risk ranking's AUTOC CI lay below 0 in {au['n_risk_autoc_ci_below_0']} of 22 trials. At a 20% budget, the risk ranking's "
             f"estimated policy value was below the trial's average effect, which is the expected value of random selection, in {n15} of 22 "
             f"trials (point estimates; two-sided sign test P = {p_sign:.2f}). Qini was computed only for the risk, scale, and doubly robust "
             f"rankings. The frozen scorer reports each ranking's AUTOC with its own bootstrap interval; paired AUTOC differences between "
             f"rankings would need the sealed outcomes to be read outside the frozen scorer, so they were not computed.\n")
    sr = q3["dr_with_sign_rule_trials"]
    o += ["\n## Table S11. Sensitivity of the Question 3 comparisons (people helped per 10,000 reached, 95% CI; not pre-specified)\n",
          "| Analysis | Doubly robust over risk | Scale over risk | Doubly robust over scale |", "|---|---|---|---|"]
    for b_ in ("0.1", "0.2", "0.3"):
        bd = q3["budgets"][b_]
        o.append(f"| Budget {int(float(b_) * 100)}% | {pooled10(bd['dr_over_risk'])} | {pooled10(bd['scale_over_risk'])} [{bd['scale_over_risk']['k']}] | "
                 f"{pooled10(bd['dr_over_scale'])} |")
    o.append(f"| Wave 1 with ELAIA-1 and Oregon added | {pooled10(sr['wave1'])} [15] | – | – |")
    o.append(f"| All trials with ELAIA-1 and Oregon added | {pooled10(sr['all'])} [24] | – | – |")
    lr = q3["between_learner_correlation_range"]
    o.append(f"\nBracketed numbers are trials pooled where fewer than 22. Across trials, the per-trial gains of the four learned rankings "
             f"correlated at {lr[0]:.2f} to {lr[1]:.2f}.\n")
    bl = q3["budgets_learners"]
    o += ["\n*Each learned ranking over the two simple alternatives, by budget (all 22 trials, same test units)*\n",
          "| Comparison | Budget 10% | Budget 20% | Budget 30% |", "|---|---|---|---|"]
    o.append(f"| Response ranking over the risk ranking | " + " | ".join(pooled10(bl[b]["response_over_risk"]) for b in ("0.1", "0.2", "0.3")) + " |")
    o.append(f"| Scale ranking over the response ranking | " + " | ".join(pooled10(bl[b]["scale_over_response"]) for b in ("0.1", "0.2", "0.3")) + " |")
    for k_, nm_ in LEARNER.items():
        if k_ == "scale_over_risk":
            continue
        for src, lab in (("over_scale", "over the scale ranking"), ("over_response", "over the response ranking")):
            o.append(f"| {nm_} {lab} | " + " | ".join(pooled10(bl[b][src][k_]) for b in ("0.1", "0.2", "0.3")) + " |")
    pf = q3["wave1_pi_forecast_wave2"]
    o.append("\nThe wave-1 95% prediction interval of each learned ranking's gain over the risk ranking, taken as a forecast of the 9 wave-2 "
             "per-trial gains, contained " + "; ".join(
                 f"{v['n_inside']} of {v['n_wave2']} for the {LEARNER[k_].lower()} ({v10(v['pi95_wave1'][0])} to {v10(v['pi95_wave1'][1])})"
                 for k_, v in pf.items()) + ". The gains outside the intervals were Florida job-search assistance for three learners and "
             "Freedom from Hunger for the causal forest.\n")
    o += ["\n## Table S12. Detail on the calibrated ensemble's 90% intervals in model routing (not pre-specified)\n",
          "| Model pair | Test prompts | Prompts with effect 0 | Median interval width | Intervals excluding 0 | Coverage, all | Coverage, effect 0 | Coverage, effect not 0 |",
          "|---|---|---|---|---|---|---|---|"]
    for t in RB_ORDER:
        r = q4["interval_detail"][t]
        o.append(f"| {RBL[t]} | {n(r['n'])} | {r['share_effect_zero']:.3f} | {r['median_width']:.2f} | {r['share_excluding_zero']:.3f} | "
                 f"{r['coverage_all']:.3f} | {r['coverage_effect_zero']:.3f} | {r['coverage_effect_nonzero']:.3f} |")
    o.append("\nThe effect of routing a prompt to the larger model ranges from −1 to 1.\n")
    rsp, rr = q3.get("response"), q3.get("response_over_risk_paired")
    o.append("{{SPLIT}}")
    if rsp and rr:
        o += ["\n## Table S15. Response ranking as a practitioner comparator (people helped per 10,000 reached, 95% CI; not pre-specified)\n",
              "| Comparison | Wave 1 | Wave 2 | All trials |", "|---|---|---|---|"]
        o.append(f"| Response ranking over the risk ranking | {pooled10(rr['wave1'])} | {pooled10(rr['wave2'])} | {pooled10(rr['all'])} |")
        o.append(f"| Response ranking over the risk ranking, trials where higher is better | – | – | {pooled10(rr['higher_is_better'])} [{rr['higher_is_better']['k']}] |")
        o.append(f"| Response ranking over the risk ranking, trials where lower is better | – | – | {pooled10(rr['lower_is_better'])} [{rr['lower_is_better']['k']}] |")
        for k_, nm_ in (("scale_over_response", "Scale ranking over the response ranking"),
                        ("dr_over_response", "Doubly robust learner over the response ranking")):
            r = rsp[k_]
            o.append(f"| {nm_} | {pooled10(r['wave1'])} | {pooled10(r['wave2'])} | {pooled10(r['all'])} [{r['all']['k']}] |")
        lc = q3["lenta_check"]
        o += ["\n| Trial | Response over risk | Scale over response | Doubly robust over response |", "|---|---|---|---|"]
        for t in W1 + W2:
            r = rsp["per_trial"][t]; q_ = rr["per_trial"][t]
            a_, b_ = r["scale_over_response"], r["dr_over_response"]
            o.append(f"| {META[t][0]} | {ci10(q_['estimate'], q_['lo95'], q_['hi95'])} | {ci10(a_['estimate'], a_['lo95'], a_['hi95'])} | "
                     f"{ci10(b_['estimate'], b_['lo95'], b_['hi95'])} |")
        o.append("\nThe response ranking orders people by the predicted probability of the desirable outcome with the action, from the "
                 "outcome model fit to the action arm of the context split. Where the lower outcome is better, it serves the people with the "
                 "lowest predicted risk under the action first. All contrasts are paired on the same test units. In the pools against the "
                 f"response ranking, Lenta carries {100 * lc['scale_over_response']['weight']:.1f}% (scale) and "
                 f"{100 * lc['dr_over_response']['weight']:.1f}% (doubly robust) of the weight; without Lenta, the scale ranking over the "
                 f"response ranking was {pooled10(lc['scale_over_response']['without'])} and the doubly robust learner over the response ranking "
                 f"was {pooled10(lc['dr_over_response']['without'])}.\n")
        o += ["\n## Table S16. Gains by domain (people helped per 10,000 reached, 95% CI; not pre-specified)\n",
              "| Domain | Trials | Scale over risk | Response over risk | Doubly robust over risk | Doubly robust over scale | Doubly robust over response |",
              "|---|---|---|---|---|---|---|"]
        for d_, v_ in q3["by_domain"].items():
            cells = [pooled10(v_[c]) + (f" [{v_[c]['k']}]" if v_[c]["k"] != len(v_["trials"]) else "") if v_[c] else "–"
                     for c in ("scale_over_risk", "response_over_risk", "dr_over_risk", "dr_over_scale", "dr_over_response")]
            o.append(f"| {d_.replace('_', '-').capitalize()} | {len(v_['trials'])} | " + " | ".join(cells) + " |")
        o.append("\nOther: charity, exercise, consumer credit, education, employment, and consumer finance. Non-voting pools the medicine, "
                 "retail, and other trials. Pools leave out trials in which the two compared rankings select the same units; bracketed "
                 "numbers give the trials pooled where fewer than the domain's trials. Where the estimated between-trial SD is 0, the pool is a fixed-effect pool. The split by domain was not pre-specified.\n")

    # Exploratory post-unsealing tables S18-S23 (written as S17-S22 before shift_si adds +1)
    b1 = json.loads((R / "ben_exploratory_analyses.json").read_text())
    b2 = json.loads((R / "ben_additional_exploratory.json").read_text())

    # Table S17 -> S18: Trial-level forecast ESS and accuracy vs naive baselines
    qp = b1["q2_power_and_baselines"]
    m16, m19 = qp["metrics_inf16"], qp["metrics_all19"]
    o += ["\n## Table S17. Trial-level forecast effective sample size and accuracy against naive baselines (exploratory, post-unsealing)\n",
          "| Predictor | 16 informative trials MAE | 16 informative trials RMSE | All 19 eligible trials MAE | All 19 eligible trials RMSE |",
          "|---|---|---|---|---|"]
    o.append(f"| Scale-predicted gap (posted forecast) | {v10(m16['mae_forecast'])} | {v10(m16['rmse_forecast'])} | {v10(m19['mae_forecast'])} | {v10(m19['rmse_forecast'])} |")
    o.append(f"| Cross-trial mean (leave one trial out) | {v10(m16['mae_loo_mean'])} | {v10(m16['rmse_loo_mean'])} | {v10(m19['mae_loo_mean'])} | {v10(m19['rmse_loo_mean'])} |")
    o.append(f"| Predict 0 for every trial | {v10(m16['mae_predict_zero'])} | {v10(m16['rmse_predict_zero'])} | {v10(m19['mae_predict_zero'])} | {v10(m19['rmse_predict_zero'])} |")
    o.append(f"| Perfect noiseless forecast with observed sampling error (expected) | {v10(m16['expected_oracle_mae'])} | {v10(m16['expected_oracle_rmse'])} | {v10(m19['expected_oracle_mae'])} | {v10(m19['expected_oracle_rmse'])} |")
    o.append(f"\nValues are in people helped per 10,000 reached at a 20% budget. Kish effective sample size of the meta-regression weights "
             f"is {qp['ess_wave1_13']:.1f} in wave 1 (13 trials; {qp['ess_wave1_inf_11']:.1f} without the 2 zero-gap trials) and "
             f"{qp['ess_all_19']:.1f} across both waves (19 trials; {qp['ess_inf_16']:.1f} without the 3 zero-gap trials). Over the "
             f"16 informative trials, the unweighted mean observed gain was {v10(m16['unweighted_mean_observed'])} vs unweighted mean "
             f"forecast {v10(m16['unweighted_mean_forecast'])}, with median test-split SE {v10(m16['median_test_se'])}. Paired trial-bootstrap "
             f"MAE difference for the forecast was {ci10(m16['mae_diff_vs_zero']['est'], m16['mae_diff_vs_zero']['lo95'], m16['mae_diff_vs_zero']['hi95'])} "
             f"vs predicting 0 and {ci10(m16['mae_diff_vs_loo_mean']['est'], m16['mae_diff_vs_loo_mean']['lo95'], m16['mae_diff_vs_loo_mean']['hi95'])} "
             "vs the leave-one-trial-out cross-trial mean.\n")

    # Table S18 -> S19: Individual-level held-out test of benefit-vs-risk curve, downturn, and paired MSE
    im = b1["individual_mechanism"]
    o += ["\n## Table S18. Individual-level held-out test of the benefit-versus-risk curve and effect scale across the 19 beneficial trials (exploratory, post-unsealing)\n",
          "*A. Held-out AIPW treatment effect across bins of risk relative to each trial's constant-OR peak (people helped per 10,000 reached)*\n",
          "| Risk minus peak ($r - r^*$) | Trials | Test units | Held-out AIPW (95% CI) | Constant OR | Constant RR (bad) | Constant RD |",
          "|---|---|---|---|---|---|---|"]
    for lbl in ("< -0.35", "-0.35 to -0.15", "-0.15 to +0.05 (peak)", "+0.05 to +0.25", "+0.25 to +0.38", "> +0.38"):
        v = im["binned_relative_to_peak"][lbl]
        o.append(f"| {lbl} | {v['k']} | {n(v['total_test_n'])} | {pooled10(v)} | {v10(v['or_mean'])} | {v10(v['rr_bad_mean'])} | {v10(v['rd_mean'])} |")
    o += ["\n*B. Direct within-trial downturn above peak ($r - r^* > 0.25$ minus $|r - r^*| \\le 0.15$, people helped per 10,000 reached)*\n",
          "| Group | Trials | Held-out downturn (95% CI) | P | Trials with negative point estimate | Constant-OR predicted | Constant-RR predicted |",
          "|---|---|---|---|---|---|---|"]
    for grp, g_lbl in (("all", "All spanning trials"), ("voting", "Voting trials"), ("non_voting", "Non-voting trials")):
        v = im["downturn_high_minus_peak_pooled"][grp]
        o.append(f"| {g_lbl} | {v['k']} | {pooled10(v)} | {v['p_two_sided']:.3f} | {v['n_negative_point_est']} of {v['k']} | "
                 f"{v10(v['mean_pred_or_diff'])} | {v10(v['mean_pred_rr_bad_diff'])} |")
    ist_d = im["ist_downturn_power"]
    o += ["\n*C. Unbiased individual-level paired MSE comparisons ($10^{-4}$ probability$^2$; negative favors first model)*\n",
          "| Comparison | All 19 trials (95% CI) | P | First model lower MSE (individually significant vs second) | Voting 7 trials (95% CI) | Non-voting 12 trials (95% CI) |",
          "|---|---|---|---|---|---|"]
    for cmp_nm, c_lbl in (("or_minus_rr_bad", "Constant OR minus constant RR (bad)"),
                          ("or_minus_rd", "Constant OR minus constant RD"),
                          ("or_minus_rr_good", "Constant OR minus constant RR (good)")):
        ga = im["paired_mse_pooled"][cmp_nm]["all19"]
        gv = im["paired_mse_pooled"][cmp_nm]["voting"]
        gnv = im["paired_mse_pooled"][cmp_nm]["non_voting"]
        o.append(f"| {c_lbl} | {ci(ga['mean'], ga['lo95'], ga['hi95'], 2)} | {ga['p_two_sided']:.3f} | "
                 f"{ga['n_first_model_better']} of 19 ({ga['n_ci_favors_first']} vs {ga['n_ci_favors_second']}) | "
                 f"{ci(gv['mean'], gv['lo95'], gv['hi95'], 2)} | {ci(gnv['mean'], gnv['lo95'], gnv['hi95'], 2)} |")
    o.append(f"\nEach trial's risk $r$ is the recalibrated baseline risk of the undesirable outcome from the hashed context-split model, and "
             f"$r^* = \\sigma(c/2)$ is its constant-OR peak. In the 10 trials with at least 50 test units in both the peak zone and far above the peak, "
             f"the International Stroke Trial showed an observed downturn of {v10(ist_d['obs_high_minus_peak'])} (SE {v10(ist_d['se_diff'])}) vs constant-OR "
             f"prediction {v10(ist_d['pred_or_high_minus_peak'])}; detecting the constant-OR downturn at 80% power would require "
             f"{ist_d['n_multiplier_for_80pct_power']:.1f} times IST's test sample (MDD$_{{80}}$ = {v10(ist_d['mdd80_downturn'])}). "
             "Paired MSE differences $(g - \\hat\\tau_A)^2 - (g - \\hat\\tau_B)^2$ cancel irreducible outcome noise identically under known randomization propensities.\n")

    # Table S19 -> S20: Random selection anchor, mid-propensity voting, tight budgets, PATH interaction, stronger tuned DR
    va = b2["gain_over_random_ate"]
    mp = b2["midprop_voting7"]
    tb = b1["tight_budgets_pooled"]
    ps = b2["path_and_stronger_dr_pooled"]
    o += ["\n## Table S19. Practitioner comparators, tighter budgets, and exploratory CATE sensitivity analyses (people helped per 10,000 reached, 95% CI; exploratory, post-unsealing)\n",
          "*A. Gain over random selection (top 20% policy value minus trial average treatment effect)*\n",
          "| Ranking | All 22 trials | Voting (7 trials) | Medicine (4 trials) | Non-voting (15 trials) |",
          "|---|---|---|---|---|"]
    for nm, lbl in (("risk", "Risk ranking"), ("scale", "Scale ranking"), ("response", "Response ranking"),
                    ("dr", "Doubly robust learner (registered)"), ("path_int", "PATH 4-parameter interaction"),
                    ("dr_strong", "Tuned risk-augmented DR (70% train)")):
        o.append(f"| {lbl} | {pooled10(va[nm]['all22'])} | {pooled10(va[nm]['voting7'])} | {pooled10(va[nm]['medicine4'])} | {pooled10(va[nm]['non_voting15'])} |")
    o += ["\n*B. Mid-propensity targeting ($-|p_0 - 0.5|$) in 7 voting trials, tighter budgets (1% and 5%), PATH interaction, and tuned 70%-train DR*\n",
          "| Comparison | Trials | Pooled gain (95% CI) |", "|---|---|---|"]
    o.append(f"| Mid-propensity over risk ranking (voting, 20% budget) | {mp['midprop_minus_risk']['k']} | {pooled10(mp['midprop_minus_risk'])} |")
    o.append(f"| Scale over mid-propensity ranking (voting, 20% budget) | {mp['scale_minus_midprop']['k']} | {pooled10(mp['scale_minus_midprop'])} |")
    o.append(f"| Response over mid-propensity ranking (voting, 20% budget) | {mp['response_minus_midprop']['k']} | {pooled10(mp['response_minus_midprop'])} |")
    o.append(f"| Doubly robust over mid-propensity ranking (voting, 20% budget) | {mp['dr_minus_midprop']['k']} | {pooled10(mp['dr_minus_midprop'])} |")
    o.append(f"| Scale over risk ranking at 5% budget | {tb['0.05']['scale_minus_risk']['k']} | {pooled10(tb['0.05']['scale_minus_risk'])} |")
    o.append(f"| Doubly robust over scale ranking at 5% budget | {tb['0.05']['benefit_minus_scale']['k']} | {pooled10(tb['0.05']['benefit_minus_scale'])} |")
    o.append(f"| Scale over risk ranking at 1% budget | {tb['0.01']['scale_minus_risk']['k']} | {pooled10(tb['0.01']['scale_minus_risk'])} |")
    o.append(f"| Doubly robust over scale ranking at 1% budget | {tb['0.01']['benefit_minus_scale']['k']} | {pooled10(tb['0.01']['benefit_minus_scale'])} |")
    o.append(f"| PATH 4-parameter interaction over risk ranking (20% budget) | {ps['path_int_minus_risk']['all']['k']} | {pooled10(ps['path_int_minus_risk']['all'])} |")
    o.append(f"| PATH 4-parameter interaction over 3-parameter scale ranking (20% budget) | {ps['path_int_minus_scale']['all']['k']} | {pooled10(ps['path_int_minus_scale']['all'])} |")
    o.append(f"| Tuned risk-augmented DR (70% train) over risk ranking (20% budget) | {ps['dr_strong_minus_risk']['all']['k']} | {pooled10(ps['dr_strong_minus_risk']['all'])} |")
    o.append(f"| Tuned risk-augmented DR (70% train) over 70%-train scale ranking (20% budget) | {b2['dr_strong_vs_scale_70pct']['k']} | {pooled10(b2['dr_strong_vs_scale_70pct'])} |")
    o.append(f"| Tuned risk-augmented DR (70% train) over 50%-train scale ranking (20% budget) | {ps['dr_strong_minus_scale']['all']['k']} | {pooled10(ps['dr_strong_minus_scale']['all'])} |")
    o.append(f"| 70%-train scale ranking over 50%-train scale ranking (20% budget) | {b2['scale_70pct_vs_scale_50pct']['k']} | {pooled10(b2['scale_70pct_vs_scale_50pct'])} |")
    o.append(f"| Tuned risk-augmented DR (70% train) over response ranking (20% budget) | {ps['dr_strong_minus_response']['all']['k']} | {pooled10(ps['dr_strong_minus_response']['all'])} |")
    o.append(f"| Tuned risk-augmented DR (70% train) over registered DR learner (20% budget) | {ps['dr_strong_minus_dr']['all']['k']} | {pooled10(ps['dr_strong_minus_dr']['all'])} |")
    o.append("\nThe tuned risk-augmented DR learner added cross-fitted logit predicted risk as an explicit feature, selected gradient-boosting "
             "regularization on the development split by minimizing out-of-sample DR pseudo-outcome MSE, and refit on context plus development "
             "splits combined (70% of each trial). Against the 50%-trained scale ranking, its domain pools were "
             f"{pooled10(ps['dr_strong_minus_scale']['non_voting15'])} outside voting, {pooled10(ps['dr_strong_minus_scale']['voting7'])} in voting, "
             f"and {pooled10(ps['dr_strong_minus_scale']['medicine4'])} in the 4 clinical trials.\n")

    # Table S20 -> S21: Statistical power (MDD_80) and trial ATEs
    pw = b1["power_mdd80"]
    ates = b2["ate_summary"]["per_trial"]
    o += ["\n## Table S20. Statistical power (80% minimum detectable differences) and test-split average treatment effects by trial (people helped per 10,000 reached; exploratory, post-unsealing)\n",
          "| Trial | Domain | Test units | Test ATE | Scale vs risk SE (MDD$_{80}$) | Doubly robust vs risk SE (MDD$_{80}$) | Doubly robust vs scale SE (MDD$_{80}$) |",
          "|---|---|---|---|---|---|---|"]
    for t in W1 + W2:
        r = pw["per_trial"][t]
        o.append(f"| {META[t][0]} | {META[t][1]} | {n(r['n_test'])} | {v10(ates[t])} | "
                 f"{v10(r['se_scale_vs_risk'])} ({v10(r['mdd80_scale_vs_risk'])}) | "
                 f"{v10(r['se_dr_vs_risk'])} ({v10(r['mdd80_dr_vs_risk'])}) | "
                 f"{v10(r['se_dr_vs_scale'])} ({v10(r['mdd80_dr_vs_scale'])}) |")
    o.append(f"\nMDD$_{{80}} = 2.80 \\times \\text{{SE}}$ is the two-sided 80%-power minimum detectable difference at $\\alpha = 0.05$ for the 20% budget contrast. "
             f"Across all 22 trials, per-trial DR-vs-risk SE ranged from {v10(pw['per_trial_dr_se_min'])} to {v10(pw['per_trial_dr_se_max'])} "
             f"(median {v10(pw['per_trial_dr_se_median'])}; median MDD$_{{80}}$ {v10(pw['per_trial_dr_mdd80_median'])}). Across the 4 clinical trials, "
             f"per-trial DR-vs-risk MDD$_{{80}}$ ranged from {v10(pw['clinical_4_mdd80_range'][0])} to {v10(pw['clinical_4_mdd80_range'][1])} "
             f"(pooled clinical MDD$_{{80}}$ {v10(pw['pooled_clinical_dr_over_risk_4']['mdd80'])}). Pooled across all 22 trials, MDD$_{{80}}$ was "
             f"{v10(pw['pooled_dr_over_risk_22']['mdd80'])} for doubly robust over risk and {v10(pw['pooled_dr_over_scale_22']['mdd80'])} for doubly robust over scale.\n")

    # Table S21 -> S22: Circular plug-in vs held-out sealed AIPW optimism across all 5 methods
    opt = b2["optimism_all_models"]
    o += ["\n## Table S21. Circular model-based (plug-in) versus held-out sealed AIPW gains over the risk ranking (people helped per 10,000 reached; exploratory, post-unsealing)\n",
          "| Ranking | Trials | Unweighted mean plug-in gain | Unweighted mean held-out AIPW gain | Random-effects overstatement (95% CI) | Trials overstated |",
          "|---|---|---|---|---|---|"]
    for nm, lbl in (("dr", "Doubly robust learner"), ("cpfn", "CausalPFN"), ("cf", "Causal forest"),
                    ("rose", "Calibrated ensemble"), ("scale", "Scale ranking (3-parameter)")):
        v = opt[nm]
        o.append(f"| {lbl} | {v['n_trials']} | {v10(v['mean_plugin'])} | {v10(v['mean_heldout'])} | {pooled10(v)} | {v['n_overstated']} of {v['n_trials']} |")
    o.append("\nPlug-in gain evaluates each ranking's top 20% minus the risk ranking's top 20% using that model's own predicted treatment effects "
             "on the test units. Held-out AIPW gain evaluates the exact same selections using the sealed test outcomes and known design propensities. "
             "Random-effects overstatement pools the paired per-trial difference (plug-in minus held-out) with cluster-bootstrap variances.\n")

    # Table S22 -> S23: Learning curves in the 4 largest trials
    lc = b2["learning_curves_big4"]
    o += ["\n## Table S22. Training-size learning curves in the four largest trials (total N > 200,000; people helped per 10,000 reached; exploratory, post-unsealing)\n",
          "| Trial (test ATE) | Context training N | Scale over full risk | Doubly robust over full risk | Doubly robust minus scale (SD across 3 subsamples) |",
          "|---|---|---|---|---|"]
    for t in ("voter", "x5", "lenta", "w2_robocall2008"):
        pts_lc = sorted(lc[t].items(), key=lambda kv: kv[1]["n_ctx"])
        for k_sz, v in pts_lc:
            lbl_sz = f"{n(v['n_ctx'])} (full)" if k_sz == "full" else n(v["n_ctx"])
            sd_s = "–" if k_sz == "full" else v10(v["dr_minus_scale_sd_across_seeds"])
            o.append(f"| {META[t][0]} ({v10(ates[t])}) | {lbl_sz} | {v10(v['scale_minus_fullrisk'])} | "
                     f"{v10(v['dr_minus_fullrisk'])} | {v10(v['dr_minus_scale'])} ({sd_s}) |")
    o.append("\nSubsamples drew randomization clusters without replacement from the context split across 3 independent random seeds per sample size, "
             "fit both the 3-parameter scale model and the doubly robust learner on each subsample, and evaluated both on the sealed test split.\n")

    return "\n".join(o)


RB_ORDER = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4", "routerbench__llama70b_to_claude2",
            "routerbench__mixtral_to_gpt4"]


def shift_si(t: str) -> str:
    """Renumber SI table references by one: the trial table is Table S1, so the generated tables start at S2."""
    import re
    return re.sub(r"\bS(\d{1,2})\b", lambda m_: f"S{int(m_.group(1)) + 1}", t)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    import re
    fix = lambda txt: re.sub(r"(?<![A-Za-z0-9])-(?=\d)", "\u2212", txt)  # noqa: E731  true minus sign
    (OUT / "main_tables.md").write_text(fix(shift_si(main_tables())) + "\n")
    (OUT / "si_tables.md").write_text(fix(si_tables()) + "\n")
    ms = OUT.parent / "risk_vs_benefit_targeting_22_trials_pnas_manuscript.md"
    t = ms.read_text()
    a, b = t.index("## Tables\n"), t.index("## Figure legends")
    ms.write_text(t[:a] + "## Tables\n\n" + fix(shift_si(main_tables())).replace("## Table", "### Table") + "\n" + t[b:])
    notes = (OUT.parent / "si_appendix_notes.md").read_text()
    early, late = fix(shift_si(si_tables())).split("{{SPLIT}}")
    early = fix("\n".join(trial_table())) + "\n" + early
    assert "{{TABLES_LATE}}" in notes
    (OUT.parent / "risk_vs_benefit_targeting_22_trials_si_appendix.md").write_text(
        notes.replace("{{TABLES}}", early.rstrip() + "\n").replace("{{TABLES_LATE}}", late.strip() + "\n"))
    print("wrote", OUT / "main_tables.md", OUT / "si_tables.md")


if __name__ == "__main__":
    main()
