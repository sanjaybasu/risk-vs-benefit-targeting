"""Figures 1-4 for the paper, read strictly from canonical JSON results files.

Sources:
  - results/final_results.json
  - results/theory_checks.json
  - results/ben_exploratory_analyses.json
  - results/ben_additional_exploratory.json
  - results/wave3_results.json
  - results/continuous_count_results.json
  - results/cluster_cf_and_28trial_summary.json

Figure 1 (question 1): A, effect-scale curves (constant OR, RR, RD); B, practical check vs posted
scale-predicted gap across all 23 beneficial trials (Waves 1, 2, and 3).
Figure 2 (question 2 & mechanism): A, individual-level held-out AIPW curve (19 Wave 1-2 beneficial
trials) vs constant OR, RR, RD; B, within-trial downturn above peak (10 Wave 1-2 spanning trials +
Wave 3 IST-3 + domain pools); C, trial-level forecast vs observed gain (19 Wave 1-2 + 4 Wave 3);
D, model routing (4 pairs).
Figure 3 (question 3): A, doubly robust over risk by trial (Waves 1-3, 28 trials); B, pooled gains
over risk ranking across Waves 1, 2, 3, and all 28 trials (including cluster-respecting causal forest);
C, learned rankings over simple alternatives.
Figure 4 (validity, health-care confirmation, continuous outcomes, and scaling): A, circular plug-in
vs held-out AIPW across all 28 trials and Wave 3; B, Wave 3 health-uptake and clinical trials (6 trials);
C, continuous/count outcomes (7 trials, baseline-count minus binarized constant-OR scale); D, training-size
learning curves in the 4 largest trials.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "notebooks" / "system-three" / "paper" / "figures"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
W1C, W2C, W3C = "#2a78d6", "#eb6834", "#0f766e"

LABEL = {
    "gusto": "GUSTO-I", "ist": "IST-1", "synbiotic": "Synbiotic", "elaia2": "ELAIA-2",
    "voter": "Michigan mailer", "sf_email": "SF email", "karlan_list": "Karlan–List",
    "stepup": "StepUp", "hillstrom": "Hillstrom", "sixcity": "Six-city", "newhaven": "New Haven",
    "lenta": "Lenta", "x5": "X5",
    "w2_robocall2008": "Michigan robocall", "w2_sanmateo_email": "San Mateo email",
    "w2_wisconsin_recall": "Wisconsin letter", "w2_ffh_charity": "Freedom from Hunger",
    "w2_sa_loan": "SA loan offer", "w2_mooc_megastudy": "edX plan-making",
    "w2_jsa_florida": "JSA Florida", "w2_jtpa": "JTPA", "w2_turkey_overdraft": "Turkey overdraft",
    "w3_campos_vaccine": "Sweden vaccine", "w3_mtics_crc_fit": "Catalonia CRC FIT",
    "w3_mtics_crc_remind": "Catalonia CRC SMS", "w3_alsan_covid": "US COVID video",
    "w3_ist3": "IST-3", "w3_kat_aki": "KAT-AKI",
}
RBL = {
    "routerbench__mistral7b_to_gpt4": "Mistral-7B → GPT-4",
    "routerbench__gpt35_to_gpt4": "GPT-3.5 → GPT-4",
    "routerbench__llama70b_to_claude2": "Llama-70B → Claude 2",
    "routerbench__mixtral_to_gpt4": "Mixtral → GPT-4",
}
CONT_LBL = {
    "ffh_amount": "Freedom from Hunger ($)",
    "illinois_spend": "Illinois medical spend ($)",
    "stepup_visits": "StepUp gym visits",
    "jtpa_earnings": "JTPA earnings ($)",
    "hillstrom_spend": "Hillstrom spend ($)",
    "karlan_amount": "Karlan–List giving ($)",
    "oregon_ed_visits": "Oregon ED visits (adverse)",
}


def style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=6.8)
    if grid:
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def figure1(fr, th, w3):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.05), facecolor=SURFACE)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    r = np.linspace(0.001, 0.999, 400)
    u = np.log(r / (1 - r))
    for orr, ls in ((0.8, "-"), (0.5, (0, (5, 2))), (0.3, (0, (1, 1.5)))):
        c = -np.log(orr)
        tau = 10000 * (sig(u) - sig(u - c))
        a.plot(r, tau, color=INK, linestyle=ls, linewidth=1.35, label=f"Constant OR = {orr:g}")
        rs = sig(c / 2)
        a.scatter([rs], [10000 * (sig(c / 2) - sig(-c / 2))], s=18, color=INK, zorder=3)
    a.plot(r, 10000 * 0.20 * r, color=W2C, linestyle=(0, (3, 1.5)), linewidth=1.2, label="Constant RR = 0.8")
    a.plot(r, np.full_like(r, 1000.0), color=W3C, linestyle=":", linewidth=1.1, label="Constant RD = 1,000")
    a.set_ylim(0, 4150)
    a.set_xlabel("Baseline risk of undesirable outcome", fontsize=7.3, color=INK)
    a.set_ylabel("Benefit of the action\n(people helped per 10,000 reached)", fontsize=7.3, color=INK)
    a.set_title("A. Effect-scale curves across baseline risk", fontsize=7.4, loc="left", color=INK)
    a.legend(frameon=False, fontsize=5.9, loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.5)
    style(a)

    # Panel B: all 23 beneficial trials (19 Wave 1-2 + 4 Wave 3)
    elig12 = list(fr["Q2"]["points"])
    for t in elig12:
        cv = th["trials"][t]["umax_plus_uq"] - th["trials"][t]["c"]
        g = 10 * fr["Q2"]["points"][t]["x"]
        w2 = t.startswith("w2_")
        b.scatter([cv], [g], s=24, color=W2C if w2 else W1C, marker="s" if w2 else "o", zorder=3)
        if cv < 0 or g > 300 or t == "ist":
            off = {"gusto": (-10, 26), "w2_jtpa": (-16, 18), "synbiotic": (2, 10),
                   "ist": (-24, 14), "sixcity": (-24, -12), "hillstrom": (-42, -2)}.get(t, (4, 3))
            b.annotate(LABEL[t], (cv, g), xytext=off,
                       arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5} if t in ("gusto", "w2_jtpa", "ist") else None,
                       textcoords="offset points", fontsize=6.0, color=INK)
    for t in w3["beneficial_trials"]:
        ci = w3["per_trial"][t]["context"]
        cv = ci["check_val"]
        g = 10 * (ci["posted_gap"] or 0.0)
        b.scatter([cv], [g], s=32, color=W3C, marker="^", zorder=4)
        off = {"w3_ist3": (6, 4), "w3_campos_vaccine": (5, 8), "w3_mtics_crc_fit": (-48, 36), "w3_alsan_covid": (-42, 54)}.get(t, (4, 4))
        b.annotate(LABEL[t], (cv, g), xytext=off,
                   arrowprops={"arrowstyle": "-", "color": W3C, "linewidth": 0.5} if cv < 0 else None,
                   textcoords="offset points", fontsize=5.9, color=W3C)
    b.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    b.set_xlabel("Check value: $u_{\\max} + u_q - c$ (logit units)", fontsize=7.3, color=INK)
    b.set_ylabel("Posted scale-predicted gap\n(people helped per 10,000 reached)", fontsize=7.3, color=INK)
    b.set_title("B. Practical check vs. posted gap (23 beneficial trials)", fontsize=7.4, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.2, color=W1C, label="Wave 1 (n=13)"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4.2, color=W2C, label="Wave 2 (n=6)"),
         plt.Line2D([], [], marker="^", linestyle="", markersize=4.5, color=W3C, label="Wave 3 health (n=4)")]
    b.legend(handles=h, frameon=False, fontsize=6.0, loc="upper left")
    style(b)
    fig.tight_layout(pad=0.8)
    save(fig, "figure1")


def figure2(fr, b1, w3):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.15), facecolor=SURFACE)
    a, b, c, d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: Individual-level held-out curve relative to peak r* (19 Wave 1-2 trials)
    bins = b1["individual_mechanism"]["binned_relative_to_peak"]
    order = ["< -0.35", "-0.35 to -0.15", "-0.15 to +0.05 (peak)", "+0.05 to +0.25", "+0.25 to +0.38", "> +0.38"]
    xs = np.array([bins[k]["mean_delta"] for k in order])
    obs = np.array([10 * bins[k]["mean"] for k in order])
    lo = np.array([10 * bins[k]["lo95"] for k in order])
    hi = np.array([10 * bins[k]["hi95"] for k in order])
    a.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    a.axhline(0.0, color=MUTED, linewidth=0.6)
    a.plot(xs, [10 * bins[k]["or_mean"] for k in order], color=W1C, linewidth=1.45, marker="o", markersize=3.3, label="Constant OR (scale)")
    a.plot(xs, [10 * bins[k]["rr_bad_mean"] for k in order], color=W2C, linewidth=1.3, linestyle="--", marker="^", markersize=3.3, label="Constant RR (bad)")
    a.plot(xs, [10 * bins[k]["rd_mean"] for k in order], color=W3C, linewidth=1.1, linestyle=":", marker="s", markersize=2.8, label="Constant RD")
    a.errorbar(xs, obs, yerr=[obs - lo, hi - obs], fmt="o", color=INK, ecolor=INK, elinewidth=1.2,
               capsize=2.3, markersize=4.3, zorder=5, label="Held-out AIPW (95% CI)")
    a.set_xlabel("Predicted risk minus constant-OR peak ($r - r^*$)", fontsize=7.0, color=INK)
    a.set_ylabel("Held-out treatment effect\n(people helped per 10,000 reached)", fontsize=7.0, color=INK)
    a.set_title("A. Individual-level curve (19 Wave 1–2 trials)", fontsize=7.2, loc="left", color=INK)
    a.set_ylim(-60, 760)
    a.legend(frameon=False, fontsize=5.6, loc="upper right")
    style(a)

    # Panel B: Within-trial downturn above peak (10 Wave 1-2 spanning + Wave 3 IST-3 + pools)
    dt = b1["individual_mechanism"]["downturn_per_trial"]
    dp = b1["individual_mechanism"]["downturn_high_minus_peak_pooled"]
    w3_dt = w3["pooled"]["w3_h1_downturn_per_trial"]
    vot_t = sorted([t for t, v in dt.items() if v["domain"] == "voting"], key=lambda t: dt[t]["obs_high_minus_peak"])
    nv_t = sorted([t for t, v in dt.items() if v["domain"] != "voting"], key=lambda t: dt[t]["obs_high_minus_peak"])
    y_pos, yticks, ylabels = 0.0, [], []
    for grp_ts, col in ((vot_t, W1C), (nv_t, W2C)):
        for t in grp_ts:
            v = dt[t]
            est, se = 10 * v["obs_high_minus_peak"], 10 * v["se_diff"]
            b.plot([est - 1.96 * se, est + 1.96 * se], [y_pos, y_pos], color=col, linewidth=1.0)
            b.scatter([est], [y_pos], s=16, color=col, zorder=3)
            b.scatter([10 * v["pred_or_high_minus_peak"]], [y_pos], s=13, facecolors="none", edgecolors=MUTED, marker="D", linewidth=0.8, zorder=4)
            yticks.append(y_pos); ylabels.append(LABEL[t]); y_pos -= 1.0
        y_pos -= 0.35
    # Add Wave 3 IST-3 spanning clinical trial!
    if "w3_ist3" in w3_dt:
        v3 = w3_dt["w3_ist3"]
        est, se = 10 * v3["obs_high_minus_peak"], 10 * v3["se_diff"]
        b.plot([est - 1.96 * se, est + 1.96 * se], [y_pos, y_pos], color=W3C, linewidth=1.15)
        b.scatter([est], [y_pos], s=22, color=W3C, marker="^", zorder=3)
        b.scatter([10 * v3["pred_or_diff"]], [y_pos], s=13, facecolors="none", edgecolors=MUTED, marker="D", linewidth=0.8, zorder=4)
        yticks.append(y_pos); ylabels.append("IST-3 (Wave 3)"); y_pos -= 1.35
    for key, lbl_p, col in (("voting", "Pooled voting (k=6)", W1C),
                            ("non_voting", "Pooled non-voting (k=4)", W2C),
                            ("all", "Pooled W1–2 (k=10)", INK)):
        vp = dp[key]
        b.plot([10 * vp["lo95"], 10 * vp["hi95"]], [y_pos, y_pos], color=col, linewidth=1.6)
        b.scatter([10 * vp["mean"]], [y_pos], s=24, color=col, marker="D", zorder=4)
        yticks.append(y_pos); ylabels.append(lbl_p); y_pos -= 1.0
    b.axvline(0.0, color=MUTED, linewidth=0.75)
    b.set_yticks(yticks); b.set_yticklabels(ylabels, fontsize=5.8, color=INK)
    b.tick_params(axis="y", length=0)
    b.set_xlim(-1250, 1900)
    b.set_xlabel("High-risk minus peak-zone effect\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    b.set_title("B. Within-trial downturn above peak", fontsize=7.2, loc="left", color=INK)
    h_b = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.8, color=W1C, label="Voting"),
           plt.Line2D([], [], marker="o", linestyle="", markersize=3.8, color=W2C, label="Non-voting"),
           plt.Line2D([], [], marker="^", linestyle="", markersize=4.2, color=W3C, label="Wave 3 clinical"),
           plt.Line2D([], [], marker="D", linestyle="", markersize=3.6, markerfacecolor="none", markeredgecolor=MUTED, color=MUTED, label="Const.-OR pred.")]
    b.legend(handles=h_b, frameon=True, facecolor=SURFACE, framealpha=0.92, edgecolor="none", fontsize=5.4, loc="upper right")
    style(b, grid=False); b.spines["left"].set_visible(False)

    # Panel C: Registered trial-level forecast (19 Wave 1-2 + Wave 3 beneficial trials)
    q2 = fr["Q2"]
    pts = q2["points"]
    x = 10 * np.array([p["x"] for p in pts.values()])
    beta = np.array(q2["primary_fit"]["beta"], dtype=float).copy()
    vcov = np.array(q2["primary_fit"]["vcov"], dtype=float).copy()
    beta[0] *= 10.0; vcov[0, 0] *= 100.0; vcov[0, 1] *= 10.0; vcov[1, 0] *= 10.0
    df = q2["primary_fit"]["df"]
    grid = np.linspace(-40, x.max() + 45, 100)
    G = np.column_stack([np.ones_like(grid), grid])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, vcov, G))
    crit = stats.t.ppf(0.975, df)
    c.fill_between(grid, G @ beta - crit * se, G @ beta + crit * se, color=MUTED, alpha=0.14, linewidth=0)
    w1 = q2["wave1_only"]
    c.plot(grid, G @ beta, color=INK, linewidth=1.4,
           label="W1–2 (19): {:.2f} ({:.2f} to {:.2f})\nW1 (13): {:.2f} ({:.2f} to {:.2f})".format(
               q2["primary"]["slope"], q2["primary"]["lo95"], q2["primary"]["hi95"], w1["slope"], w1["lo95"], w1["hi95"]))
    o = q2["informative_through_origin"]
    c.plot(grid, o["slope"] * grid, color=INK, linewidth=1.05, linestyle=(0, (1, 1.5)),
           label=f"16 trials, origin: {o['slope']:.2f} ({o['lo95']:.2f} to {o['hi95']:.2f})")
    c.plot(grid, grid, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), label="Observed equals forecast")
    for t, p in pts.items():
        w2 = t.startswith("w2_")
        col = W2C if w2 else W1C
        px, py, plo, phi = 10 * p["x"], 10 * p["y"], 10 * p["lo95"], 10 * p["hi95"]
        c.plot([px, px], [plo, phi], color=col, linewidth=0.85, alpha=0.8)
        c.scatter([px], [py], s=20, color=col, marker="s" if w2 else "o", zorder=3)
        if p["x"] > 25:
            c.annotate(LABEL[t], (px, py), xytext={"voter": (-24, -20), "sixcity": (-34, -12), "hillstrom": (-8, 10), "newhaven": (-54, 4)}.get(t, (4, 3)),
                       textcoords="offset points", fontsize=5.8, color=INK)
    # Overlay Wave 3 beneficial trials
    for t in w3["beneficial_trials"]:
        ci = w3["per_trial"][t]["context"]
        sr = w3["per_trial"][t]["scored"]["pairs"]["scale_minus_risk"]
        px, py, plo, phi = 10 * (ci["posted_gap"] or 0.0), 10 * sr["estimate"], 10 * sr["lo95"], 10 * sr["hi95"]
        c.plot([px, px], [plo, phi], color=W3C, linewidth=1.0, alpha=0.9)
        c.scatter([px], [py], s=28, color=W3C, marker="^", zorder=4)
        if t == "w3_ist3":
            c.annotate("IST-3", (px, py), xytext=(-28, -2), textcoords="offset points", fontsize=5.8, color=W3C)
    c.set_title("C. Registered trial-level forecast (Waves 1–3)", fontsize=7.2, loc="left", color=INK)
    c.set_ylim(-2550, 1500)
    c.set_xlabel("Scale-predicted gap\n(people helped per 10,000 reached)", fontsize=7.0, color=INK)
    c.set_ylabel("Observed gain, scale over risk\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    h_c = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.6, color=W1C, label="Wave 1"),
           plt.Line2D([], [], marker="s", linestyle="", markersize=3.6, color=W2C, label="Wave 2"),
           plt.Line2D([], [], marker="^", linestyle="", markersize=4.0, color=W3C, label="Wave 3")]
    c.legend(handles=h_c + c.get_legend_handles_labels()[0], frameon=True, facecolor=SURFACE, framealpha=0.93, edgecolor="none", fontsize=4.9, loc="lower right", handlelength=1.3, labelspacing=0.28)
    style(c)

    # Panel D: Model routing
    rb = q2["routing"]
    for t, r in rb.items():
        if t not in RBL:
            continue
        xs_r, ys_r = 10 * r["forecast"], r["scale_minus_risk"]
        col = INK if r["flag"] else MUTED
        d.plot([xs_r, xs_r], [10 * ys_r["lo95"], 10 * ys_r["hi95"]], color=col, linewidth=1.0)
        d.scatter([xs_r], [10 * ys_r["estimate"]], s=34 if not r["flag"] else 26, color=col if r["flag"] else SURFACE, edgecolor=col,
                  linewidth=1.4, zorder=4 if not r["flag"] else 3)
        off = {"routerbench__mixtral_to_gpt4": (14, 20), "routerbench__gpt35_to_gpt4": (20, 32),
               "routerbench__mistral7b_to_gpt4": (14, -22), "routerbench__llama70b_to_claude2": (-10, 16)}[t]
        d.annotate(RBL[t], (xs_r, 10 * ys_r["estimate"]), xytext=off, textcoords="offset points", fontsize=5.8, color=INK,
                   ha="right" if off[0] < 0 else "left",
                   arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 4})
    d.plot([-50, 1150], [-50, 1150], color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    d.axhline(0, color=MUTED, linewidth=0.65)
    d.set_xlim(-50, 1520)
    d.set_xlabel("Scale-predicted gap\n(prompts improved per 10,000 routed)", fontsize=7.0, color=INK)
    d.set_ylabel("Observed gain, scale over risk\n(prompts improved per 10,000 routed, 95% CI)", fontsize=7.0, color=INK)
    d.set_title("D. Model routing (known unit effects)", fontsize=7.2, loc="left", color=INK)
    h_d = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.0, color=INK, label="Flagged by diagnostic"),
           plt.Line2D([], [], marker="o", linestyle="", markersize=4.0, markerfacecolor=SURFACE, markeredgecolor=MUTED, color=MUTED, label="Not flagged")]
    d.legend(handles=h_d, frameon=False, fontsize=5.8, loc="lower left")
    style(d)

    fig.tight_layout(pad=0.8, h_pad=0.95, w_pad=0.85)
    save(fig, "figure2")


def figure3(fr, w3, c28):
    q3 = fr["Q3"]
    fig = plt.figure(figsize=(7.0, 5.25), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[2.45, 1.1], hspace=0.44, wspace=1.08)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])

    # Panel A: DR over risk across all 28 trials (Waves 1, 2, 3)
    pt = dict(q3["dr"]["per_trial"])
    for t in w3["trials"]:
        pt[t] = w3["per_trial"][t]["scored"]["pairs"]["benefit_minus_risk"]
    w1_ts = sorted([t for t in fr["trials"]["wave1"]], key=lambda t: -pt[t]["estimate"])
    w2_ts = sorted([t for t in fr["trials"]["wave2"]], key=lambda t: -pt[t]["estimate"])
    w3_ts = sorted([t for t in w3["trials"]], key=lambda t: -pt[t]["estimate"])
    y, ticks, labels = 0.0, [], []
    for group, col, mk in ((w1_ts, W1C, "o"), (w2_ts, W2C, "s"), (w3_ts, W3C, "^")):
        for t in group:
            v = pt[t]
            a.plot([10 * v["lo95"], 10 * v["hi95"]], [y, y], color=col, linewidth=1.0)
            a.scatter([10 * v["estimate"]], [y], s=16, color=col, marker=mk, zorder=3)
            ticks.append(y); labels.append(LABEL[t]); y -= 1.0
        y -= 0.7
    a.axvline(0, color=MUTED, linewidth=0.7)
    a.set_yticks(ticks); a.set_yticklabels(labels, fontsize=5.8, color=INK); a.tick_params(axis="y", length=0)
    a.set_xlim(-1750, 2350)
    a.set_xlabel("Gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    a.set_title("A. Doubly robust over risk (28 trials)", fontsize=7.3, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.8, color=W1C, label="Wave 1 (13)"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=3.8, color=W2C, label="Wave 2 (9)"),
         plt.Line2D([], [], marker="^", linestyle="", markersize=4.2, color=W3C, label="Wave 3 health (6)")]
    a.legend(handles=h, frameon=True, facecolor=SURFACE, framealpha=0.95, edgecolor="none", fontsize=5.5, loc="upper right", bbox_to_anchor=(1.04, 1.0))
    style(a, grid=False); a.spines["left"].set_visible(False)

    # Panel B: Pooled gain over risk (W1, W2, W3, All 28) for Scale, Response, DR, Cluster-CF
    p28 = c28["pooled_28"]["all28"]
    cf = c28["cluster_cf_summary"]
    w3p = w3["pooled"]["all6"]
    rows_b = [
        ("Scale ranking", q3["scale_over_risk"]["wave1"], q3["scale_over_risk"]["wave2"], w3p["scale_minus_risk"], p28["scale_over_risk"]),
        ("Response ranking", q3["response_over_risk_paired"]["wave1"], q3["response_over_risk_paired"]["wave2"], w3p["response_minus_risk"], p28["response_over_risk"]),
        ("Doubly robust", q3["dr"]["wave1"], q3["dr"]["wave2"], w3p["benefit_minus_risk"], p28["dr_over_risk"]),
        ("Cluster causal forest", None, None, cf["wave3_6"]["cf_over_risk"], cf["all28"]["cf_over_risk"]),
        ("Calibrated ensemble", q3["rose"]["wave1"], q3["rose"]["wave2"], w3p["rose_minus_risk"], None),
    ]
    y, ticks, labels = 0.0, [], []
    for name, r1, r2, r3, rall in rows_b:
        entries = []
        if r1: entries.append(("W1", W1C, "o", r1))
        if r2: entries.append(("W2", W2C, "s", r2))
        if r3: entries.append(("W3", W3C, "^", r3))
        if rall: entries.append(("all", INK, "D", rall))
        if name == "Cluster causal forest":
            entries.insert(0, ("W1–2", W1C, "o", cf["wave12_22"]["cf_over_risk"]))
        if name == "Calibrated ensemble":
            entries.append(("W1–2", INK, "D", q3["rose"]["all"]))
        for wl, col, mk, r in entries:
            lo_, hi_, mn_ = 10 * r["lo95"], 10 * r["hi95"], 10 * r["mean"]
            if wl == "all" and "pi95" in r:
                b.fill_between([10 * v for v in r["pi95"]], y - 0.28, y + 0.28, color=MUTED, alpha=0.17, linewidth=0)
            b.plot([max(lo_, -950), min(hi_, 950)], [y, y], color=col, linewidth=1.4 if wl == "all" else 1.0)
            if -950 <= mn_ <= 950:
                b.scatter([mn_], [y], s=20, color=col, marker=mk, zorder=3)
            ticks.append(y); labels.append(f"{name}, {wl} (k={r['k']})"); y -= 1.0
        y -= 0.65
    b.axvline(0, color=MUTED, linewidth=0.7)
    b.set_xlim(-980, 980)
    b.set_yticks(ticks); b.set_yticklabels(labels, fontsize=5.6, color=INK); b.tick_params(axis="y", length=0)
    b.set_xlabel("Pooled gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    b.set_title("B. Pooled gain over risk (Waves 1–3)", fontsize=7.3, loc="left", color=INK)
    b.text(0.99, 0.01, "Gray band: 95% PI (clipped at ±950)", transform=b.transAxes, ha="right", va="bottom", fontsize=5.4, color=MUTED)
    style(b, grid=False); b.spines["left"].set_visible(False)

    # Panel C: Learned over simple alternatives (All 28 where available, else 22)
    rows_c = [
        ("Doubly robust over scale (28)", p28["dr_over_scale"]),
        ("Tuned 70% DR over scale (28)", p28["dr_strong_over_scale"]),
        ("Cluster CF over scale (28)", cf["all28"]["cf_over_scale"]),
        ("Cluster CF over response (28)", cf["all28"]["cf_over_response"]),
        ("Doubly robust over response (22)", q3["learners_over_response"]["dr"]),
        ("CausalPFN over scale (22)", q3["learners_over_scale"]["causalpfn"]),
        ("Calibrated ens. over scale (22)", q3["learners_over_scale"]["rose"]),
    ]
    y, ticks, labels = 0.0, [], []
    for lbl, r in rows_c:
        if "pi95" in r:
            c.fill_between([10 * v for v in r["pi95"]], y - 0.28, y + 0.28, color=MUTED, alpha=0.17, linewidth=0)
        c.plot([10 * r["lo95"], 10 * r["hi95"]], [y, y], color=INK, linewidth=1.45)
        c.scatter([10 * r["mean"]], [y], s=20, color=INK, marker="D", zorder=3)
        ticks.append(y); labels.append(lbl); y -= 1.0
    c.axvline(0, color=MUTED, linewidth=0.7)
    c.set_yticks(ticks); c.set_yticklabels(labels, fontsize=5.7, color=INK); c.tick_params(axis="y", length=0)
    c.set_xlim(-350, 480)
    c.set_xlabel("Pooled gain over simple alternative\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    c.set_title("C. Learned over simple alternatives", fontsize=7.3, loc="left", color=INK)
    style(c, grid=False); c.spines["left"].set_visible(False)

    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.17, right=0.97)
    save(fig, "figure3")


def figure4(b2, w3, cc, c28):
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.95), facecolor=SURFACE)
    a, b, c, d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: Circular plug-in vs held-out sealed AIPW across all 28 trials + Wave 3 confirmatory
    p28 = c28["pooled_28"]["all28"]
    cf28 = c28["cluster_cf_summary"]["all28"]["optimism"]
    w3opt = w3["pooled"]["optimism_all6"]
    # Compute unweighted means across 28 trials from b2 (22) + w3 (6)
    opt_w12 = b2["optimism_all_models"]
    def comb_means(k22, kw3, n1=22, n2=6):
        pl = (n1 * opt_w12[k22]["mean_plugin"] + n2 * w3opt[kw3]["mean_plugin"]) / (n1 + n2)
        ho = (n1 * opt_w12[k22]["mean_heldout"] + n2 * w3opt[kw3]["mean_heldout"]) / (n1 + n2)
        return 10 * pl, 10 * ho

    dr28_pl, dr28_ho = comb_means("dr", "benefit")
    rs28_pl, rs28_ho = comb_means("rose", "rose")
    sc20_pl, sc20_ho = comb_means("scale", "scale", n1=17, n2=3)

    rows_a = [
        ("DR learner, Wave 3\n(6/6 over; confirm.)", 10 * w3opt["benefit"]["mean_plugin"], 10 * w3opt["benefit"]["mean_heldout"], int(round(10 * w3opt["benefit"]["mean"]))),
        ("DR learner, all 28\n(28/28 over)", dr28_pl, dr28_ho, int(round(10 * p28["dr_opt"]["mean"]))),
        ("Cluster causal forest\n(21/28 over)", 10 * cf28["mean_plugin"], 10 * cf28["mean_heldout"], int(round(10 * cf28["mean"]))),
        ("Calibrated ensemble\n(21/28 over)", rs28_pl, rs28_ho, int(round(10 * p28["rose_opt"]["mean"]))),
        ("Scale ranking, 3-param\n(10/20 over)", sc20_pl, sc20_ho, int(round(10 * p28["scale_opt"]["mean"]))),
    ]
    ys = np.arange(len(rows_a))[::-1]
    for yi, (lbl, pl, ho, ov) in zip(ys, rows_a):
        a.plot([ho, pl], [yi, yi], color=MUTED, linewidth=1.4, zorder=2)
        a.scatter([pl], [yi], s=28, color=W2C, marker="s", zorder=4, label="Model plug-in" if yi == ys[0] else None)
        a.scatter([ho], [yi], s=28, color=W1C, marker="o", zorder=4, label="Held-out AIPW" if yi == ys[0] else None)
        a.text(max(pl, ho) + 45, yi, f"+{ov:,}" if ov > 0 else f"{ov:,}", va="center", ha="left", fontsize=5.8, color=INK)
    a.axvline(0, color=MUTED, linewidth=0.7)
    a.set_yticks(ys); a.set_yticklabels([r[0] for r in rows_a], fontsize=5.7, color=INK); a.tick_params(axis="y", length=0)
    a.set_xlim(-350, 2600)
    a.set_xlabel("Mean gain over risk ranking\n(people helped per 10,000 reached)", fontsize=7.0, color=INK)
    a.set_title("A. Circular plug-in vs. held-out AIPW (28 trials)", fontsize=7.2, loc="left", color=INK)
    a.legend(frameon=False, fontsize=5.7, loc="lower right")
    style(a, grid=False); a.spines["left"].set_visible(False)

    # Panel B: Wave 3 confirmatory health-uptake & clinical trials (DR vs Scale vs Risk)
    w3_order = ["w3_campos_vaccine", "w3_mtics_crc_fit", "w3_alsan_covid", "w3_mtics_crc_remind", "w3_ist3", "w3_kat_aki"]
    ys_b = np.arange(len(w3_order))[::-1]
    for yi, t in zip(ys_b, w3_order):
        ps = w3["per_trial"][t]["scored"]["pairs"]
        for dy, key, col, mk, lbl_m in [
            (+0.22, "scale_minus_risk", W1C, "o", "Scale over risk"),
            (0.0, "response_minus_risk", W3C, "s", "Response over risk"),
            (-0.22, "benefit_minus_risk", W2C, "^", "DR over risk"),
        ]:
            r = ps[key]
            b.plot([max(10 * r["lo95"], -2950), min(10 * r["hi95"], 2500)], [yi + dy, yi + dy], color=col, linewidth=1.0)
            b.scatter([10 * r["estimate"]], [yi + dy], s=18, color=col, marker=mk, zorder=3, label=lbl_m if yi == ys_b[0] else None)
    b.axvline(0, color=MUTED, linewidth=0.75)
    b.set_yticks(ys_b); b.set_yticklabels([LABEL[t] for t in w3_order], fontsize=5.9, color=INK); b.tick_params(axis="y", length=0)
    b.set_xlim(-3000, 2500)
    b.set_xlabel("Gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.0, color=INK)
    b.set_title("B. Wave 3 health-uptake & clinical trials", fontsize=7.2, loc="left", color=INK)
    b.legend(frameon=True, facecolor=SURFACE, framealpha=0.93, edgecolor="none", fontsize=5.4, loc="upper left")
    style(b, grid=False); b.spines["left"].set_visible(False)

    # Panel C: Continuous & count outcomes (7 trials, W3-H5): Baseline-count minus Binarized constant-OR scale (mSD)
    c_order = ["ffh_amount", "illinois_spend", "stepup_visits", "jtpa_earnings", "hillstrom_spend", "karlan_amount", "oregon_ed_visits"]
    ys_c = np.arange(len(c_order))[::-1]
    for yi, t in zip(ys_c, c_order):
        r = cc["per_trial"][t]["milli_sd"]["pairs"]["base_mean_minus_bin_scale"]
        z0 = cc["per_trial"][t]["meta"]["zero_share_ctrl"]
        col = W3C if (z0 < 0.90 and t != "oregon_ed_visits") else MUTED
        c.plot([r["lo95"], r["hi95"]], [yi, yi], color=col, linewidth=1.15)
        c.scatter([r["estimate"]], [yi], s=22, color=col, zorder=3)
    c.axvline(0, color=MUTED, linewidth=0.75)
    c.set_yticks(ys_c)
    c.set_yticklabels([f"{CONT_LBL[t]}\n({100*cc['per_trial'][t]['meta']['zero_share_ctrl']:.0f}% zeros)" for t in c_order], fontsize=5.4, color=INK)
    c.tick_params(axis="y", length=0)
    c.set_xlabel("Baseline-count minus binarized scale\n(milli-SD of control outcome, 95% CI)", fontsize=7.0, color=INK)
    c.set_title("C. Continuous & count outcomes (7 trials, W3-H5)", fontsize=7.2, loc="left", color=INK)
    style(c, grid=False); c.spines["left"].set_visible(False)

    # Panel D: Learning curves in 4 largest trials
    lc = b2["learning_curves_big4"]
    styles_lc = {
        "voter": (W1C, "o", "-", "Michigan mailer (ATE 769)"),
        "x5": (W3C, "s", "-", "X5 text message (ATE 222)"),
        "lenta": (MUTED, "^", "--", "Lenta text message (ATE 48)"),
        "w2_robocall2008": (W2C, "D", "-", "Michigan robocall (ATE 73)"),
    }
    for t, (col, mk, ls, lbl) in styles_lc.items():
        pts_lc = sorted(lc[t].values(), key=lambda z: z["n_ctx"])
        ns = np.array([p["n_ctx"] / 1000.0 for p in pts_lc])
        ds = np.array([10 * p["dr_minus_scale"] for p in pts_lc])
        d.plot(ns, ds, color=col, marker=mk, linestyle=ls, linewidth=1.3, markersize=3.8, label=lbl)
    d.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
    d.set_xscale("log")
    d.set_xticks([5, 15, 50, 100, 350])
    d.set_xticklabels(["5k", "15k", "50k", "100k", "350k"])
    d.set_xlabel("Context training sample size (log scale)", fontsize=7.0, color=INK)
    d.set_ylabel("Doubly robust minus scale\n(people helped per 10,000 reached)", fontsize=7.0, color=INK)
    d.set_title("D. Learning curves in the 4 largest trials", fontsize=7.2, loc="left", color=INK)
    d.legend(frameon=False, fontsize=5.3, loc="lower left")
    style(d)

    fig.tight_layout(pad=0.8, h_pad=0.95, w_pad=0.85)
    save(fig, "figure4")


def main() -> None:
    fr = json.loads((ROOT / "results" / "final_results.json").read_text())
    th = json.loads((ROOT / "results" / "theory_checks.json").read_text())
    b1 = json.loads((ROOT / "results" / "ben_exploratory_analyses.json").read_text())
    b2 = json.loads((ROOT / "results" / "ben_additional_exploratory.json").read_text())
    w3 = json.loads((ROOT / "results" / "wave3_results.json").read_text())
    cc = json.loads((ROOT / "results" / "continuous_count_results.json").read_text())
    c28 = json.loads((ROOT / "results" / "cluster_cf_and_28trial_summary.json").read_text())
    figure1(fr, th, w3)
    figure2(fr, b1, w3)
    figure3(fr, w3, c28)
    figure4(b2, w3, cc, c28)


if __name__ == "__main__":
    main()
