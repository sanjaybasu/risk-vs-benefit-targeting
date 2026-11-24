"""Figures 1-4 for the paper, read from canonical results JSON files only.

Sources:
  - results/final_results.json
  - results/theory_checks.json
  - results/ben_exploratory_analyses.json
  - results/ben_additional_exploratory.json

Figure 1 (question 1): A, benefit against baseline risk under constant odds ratio vs constant risk ratio
and constant risk difference; B, practical check value against posted scale-predicted gap (19 trials).
Figure 2 (question 2 & individual-level mechanism): A, individual-level held-out AIPW effect across bins
of risk relative to each trial's peak r* (19 beneficial trials) vs constant OR, RR, and RD curves;
B, within-trial downturn above peak (10 spanning trials + domain pools); C, registered trial-level
forecast vs observed gain (19 trials); D, model routing with known effects (4 pairs).
Figure 3 (question 3): A, doubly robust over risk by trial; B, pooled gain over risk by wave and all
trials; C, learned rankings over simple alternatives.
Figure 4 (evaluation validity & sample-size scaling): A, circular plug-in vs held-out sealed AIPW gain
over risk across all 5 methods; B, training-size learning curves (DR minus scale) in the 4 largest trials.

All gains, forecasts, and curves are scaled to per 10,000 reached (per 10,000 routed for model routing).
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
W1C, W2C = "#2a78d6", "#eb6834"
TEAL, RED, PURP = "#0f766e", "#b91c1c", "#6d28d9"

LABEL = {"gusto": "GUSTO-I", "ist": "IST", "synbiotic": "Synbiotic", "elaia2": "ELAIA-2", "voter": "Michigan mailer",
         "sf_email": "SF email", "karlan_list": "Karlan–List", "stepup": "StepUp", "hillstrom": "Hillstrom",
         "sixcity": "Six-city", "newhaven": "New Haven", "lenta": "Lenta", "x5": "X5",
         "w2_robocall2008": "Michigan robocall", "w2_sanmateo_email": "San Mateo email",
         "w2_wisconsin_recall": "Wisconsin letter", "w2_ffh_charity": "Freedom from Hunger", "w2_sa_loan": "SA loan offer",
         "w2_mooc_megastudy": "edX plan-making", "w2_jsa_florida": "JSA Florida", "w2_jtpa": "JTPA",
         "w2_turkey_overdraft": "Turkey overdraft"}
RBL = {"routerbench__mistral7b_to_gpt4": "Mistral-7B → GPT-4", "routerbench__gpt35_to_gpt4": "GPT-3.5 → GPT-4",
       "routerbench__llama70b_to_claude2": "Llama-70B → Claude 2", "routerbench__mixtral_to_gpt4": "Mixtral → GPT-4"}
LEARNER = {"scale_over_risk": "Scale ranking", "dr": "Doubly robust learner", "causal_forest": "Causal forest", "causalpfn": "CausalPFN",
           "rose": "Calibrated ensemble"}


def style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, labelsize=7)
    if grid:
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def figure1(fr, th):
    # PNAS full two-column width = 17.8 cm = 7.0 in; all text >= 6.5 pt at final print size
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.15), facecolor=SURFACE)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    r = np.linspace(0.001, 0.999, 400)
    u = np.log(r / (1 - r))
    for orr, ls in ((0.8, "-"), (0.5, (0, (5, 2))), (0.3, (0, (1, 1.5)))):
        c = -np.log(orr)
        tau = 10000 * (sig(u) - sig(u - c))
        a.plot(r, tau, color=INK, linestyle=ls, linewidth=1.35, label=f"Constant OR = {orr:g}")
        rs = sig(c / 2)
        a.scatter([rs], [10000 * (sig(c / 2) - sig(-c / 2))], s=18, color=INK, zorder=3)
    # Reference constant risk ratio (20% relative reduction, matching OR=0.8 at low risk) & constant RD
    a.plot(r, 10000 * 0.20 * r, color=W2C, linestyle=(0, (3, 1.5)), linewidth=1.2, label="Constant RR = 0.8")
    a.plot(r, np.full_like(r, 1000.0), color=TEAL, linestyle=":", linewidth=1.1, label="Constant RD = 1,000")
    a.set_ylim(0, 4150)
    a.set_xlabel("Baseline risk of undesirable outcome", fontsize=7.5, color=INK)
    a.set_ylabel("Benefit of the action\n(people helped per 10,000 reached)", fontsize=7.5, color=INK)
    a.set_title("A. Effect-scale curves across baseline risk", fontsize=7.5, loc="left", color=INK)
    a.legend(frameon=False, fontsize=6.0, loc="upper left", ncol=2, columnspacing=0.9, handlelength=1.6)
    style(a)

    elig = list(fr["Q2"]["points"])
    for t in elig:
        cv = th["trials"][t]["umax_plus_uq"] - th["trials"][t]["c"]
        g = 10 * fr["Q2"]["points"][t]["x"]
        w2 = t.startswith("w2_")
        b.scatter([cv], [g], s=26, color=W2C if w2 else W1C, marker="s" if w2 else "o", zorder=3)
        if cv < 0 or g > 300 or t == "ist":
            b.annotate(LABEL[t], (cv, g), xytext={"gusto": (-6, 18), "w2_jtpa": (-10, 18), "ist": (-22, 12), "sixcity": (-24, -12), "hillstrom": (-42, -2)}.get(t, (4, 3)),
                       arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5} if t in ("gusto", "w2_jtpa", "ist") else None,
                       textcoords="offset points", fontsize=6.5, color=INK)
    b.axvline(0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    b.set_xlabel("Check value: $u_{\\max} + u_q - c$ (logit units)", fontsize=7.5, color=INK)
    b.set_ylabel("Posted scale-predicted gap\n(people helped per 10,000 reached)", fontsize=7.5, color=INK)
    b.set_title(f"B. Practical check vs. posted gap ({len(elig)} trials)", fontsize=7.5, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4.5, color=W2C, label="Wave 2")]
    b.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left")
    style(b)
    fig.tight_layout(pad=0.8)
    save(fig, "figure1")


def figure2(fr, b1):
    # 2x2 full-width PNAS figure: A = individual-level risk-binned curve; B = within-trial downturn;
    # C = registered trial-level forecast vs observed; D = model routing
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.25), facecolor=SURFACE)
    a, b, c, d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel A: Individual-level held-out curve relative to peak r*
    bins = b1["individual_mechanism"]["binned_relative_to_peak"]
    order = ["< -0.35", "-0.35 to -0.15", "-0.15 to +0.05 (peak)", "+0.05 to +0.25", "+0.25 to +0.38", "> +0.38"]
    xs = np.array([bins[k]["mean_delta"] for k in order])
    obs = np.array([10 * bins[k]["mean"] for k in order])
    lo = np.array([10 * bins[k]["lo95"] for k in order])
    hi = np.array([10 * bins[k]["hi95"] for k in order])
    y_or = np.array([10 * bins[k]["or_mean"] for k in order])
    y_rr = np.array([10 * bins[k]["rr_bad_mean"] for k in order])
    y_rd = np.array([10 * bins[k]["rd_mean"] for k in order])

    a.axvline(0.0, color=MUTED, linewidth=0.8, linestyle=(0, (4, 3)))
    a.axhline(0.0, color=MUTED, linewidth=0.6)
    a.plot(xs, y_or, color=W1C, linewidth=1.5, linestyle="-", marker="o", markersize=3.5, label="Constant OR (scale)")
    a.plot(xs, y_rr, color=W2C, linewidth=1.35, linestyle="--", marker="^", markersize=3.5, label="Constant RR (bad outcome)")
    a.plot(xs, y_rd, color=TEAL, linewidth=1.15, linestyle=":", marker="s", markersize=3.0, label="Constant RD")
    a.errorbar(xs, obs, yerr=[obs - lo, hi - obs], fmt="o", color=INK, ecolor=INK, elinewidth=1.2,
               capsize=2.5, markersize=4.5, zorder=5, label="Held-out AIPW (95% CI)")
    a.set_xlabel("Predicted risk minus constant-OR peak ($r - r^*$)", fontsize=7.2, color=INK)
    a.set_ylabel("Held-out treatment effect\n(people helped per 10,000 reached)", fontsize=7.2, color=INK)
    a.set_title("A. Individual-level curve across 19 trials (exploratory)", fontsize=7.4, loc="left", color=INK)
    a.set_ylim(-60, 760)
    a.legend(frameon=False, fontsize=5.8, loc="upper right")
    style(a)

    # Panel B: Within-trial downturn above peak (r - r* > 0.25 minus |r - r*| <= 0.15)
    dt = b1["individual_mechanism"]["downturn_per_trial"]
    dp = b1["individual_mechanism"]["downturn_high_minus_peak_pooled"]
    # Order: voting trials sorted by point est, then non-voting trials, then pools
    vot_t = sorted([t for t, v in dt.items() if v["domain"] == "voting"], key=lambda t: dt[t]["obs_high_minus_peak"])
    nv_t = sorted([t for t, v in dt.items() if v["domain"] != "voting"], key=lambda t: dt[t]["obs_high_minus_peak"])
    y_pos, yticks, ylabels = 0.0, [], []
    for grp_ts, col in ((vot_t, W1C), (nv_t, W2C)):
        for t in grp_ts:
            v = dt[t]
            est = 10 * v["obs_high_minus_peak"]
            se = 10 * v["se_diff"]
            b.plot([est - 1.96 * se, est + 1.96 * se], [y_pos, y_pos], color=col, linewidth=1.0)
            b.scatter([est], [y_pos], s=18, color=col, zorder=3)
            # Mark constant-OR prediction as tiny open diamond
            b.scatter([10 * v["pred_or_high_minus_peak"]], [y_pos], s=14, facecolors="none", edgecolors=MUTED, marker="D", linewidth=0.8, zorder=4)
            yticks.append(y_pos); ylabels.append(LABEL[t]); y_pos -= 1.0
        y_pos -= 0.45
    for key, lbl_p, col in (("voting", "Pooled voting (k=6)", W1C),
                            ("non_voting", "Pooled non-voting (k=4)", W2C),
                            ("all", "Pooled all spanning (k=10)", INK)):
        vp = dp[key]
        b.plot([10 * vp["lo95"], 10 * vp["hi95"]], [y_pos, y_pos], color=col, linewidth=1.7)
        b.scatter([10 * vp["mean"]], [y_pos], s=26, color=col, marker="D", zorder=4)
        yticks.append(y_pos); ylabels.append(lbl_p); y_pos -= 1.0
    b.axvline(0.0, color=MUTED, linewidth=0.75)
    b.set_yticks(yticks); b.set_yticklabels(ylabels, fontsize=6.0, color=INK)
    b.tick_params(axis="y", length=0)
    b.set_xlim(-1250, 1300)
    b.set_xlabel("High-risk minus peak-zone effect\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    b.set_title("B. Within-trial downturn above peak", fontsize=7.4, loc="left", color=INK)
    h_b = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=W1C, label="Voting"),
           plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=W2C, label="Non-voting"),
           plt.Line2D([], [], marker="D", linestyle="", markersize=3.8, markerfacecolor="none", markeredgecolor=MUTED, color=MUTED, label="Const.-OR pred.")]
    b.legend(handles=h_b, frameon=True, facecolor=SURFACE, framealpha=0.92, edgecolor="none", fontsize=5.8, loc="upper right")
    style(b, grid=False)
    b.spines["left"].set_visible(False)

    # Panel C: Registered trial-level forecast vs observed gain (19 trials)
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
    c.plot(grid, G @ beta, color=INK, linewidth=1.45,
           label="19 trials: {:.2f} ({:.2f} to {:.2f})\nwave 1: {:.2f} ({:.2f} to {:.2f})".format(
               q2["primary"]["slope"], q2["primary"]["lo95"], q2["primary"]["hi95"], w1["slope"], w1["lo95"], w1["hi95"]))
    o = q2["informative_through_origin"]
    c.plot(grid, o["slope"] * grid, color=INK, linewidth=1.1, linestyle=(0, (1, 1.5)),
           label=f"16 trials, origin: {o['slope']:.2f} ({o['lo95']:.2f} to {o['hi95']:.2f})")
    c.plot(grid, grid, color=MUTED, linewidth=0.95, linestyle=(0, (4, 3)), label="Observed equals forecast")
    c.annotate("3 zero-gap trials\n(90.6% weight; ESS=3.6)", (0, 0), xytext=(55, 1140), textcoords="data", ha="left",
               fontsize=6.0, color=MUTED, arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.7, "relpos": (0.0, 0.0)})
    for t, p in pts.items():
        w2 = t.startswith("w2_")
        col = W2C if w2 else W1C
        px, py, plo, phi = 10 * p["x"], 10 * p["y"], 10 * p["lo95"], 10 * p["hi95"]
        c.plot([px, px], [plo, phi], color=col, linewidth=0.9, alpha=0.8)
        c.scatter([px], [py], s=22, color=col, marker="s" if w2 else "o", zorder=3)
        if p["x"] > 25:
            c.annotate(LABEL[t], (px, py), xytext={"voter": (-24, -22), "sixcity": (-36, -13), "hillstrom": (-34, 7), "newhaven": (4, 3)}.get(t, (4, 3)),
                       textcoords="offset points", fontsize=6.0, color=INK)
    c.set_title(f"C. Registered trial-level forecast ({q2['k']} trials)", fontsize=7.4, loc="left", color=INK)
    c.set_ylim(-1700, 1500)
    c.set_xlabel("Scale-predicted gap\n(people helped per 10,000 reached)", fontsize=7.2, color=INK)
    c.set_ylabel("Observed gain, scale over risk\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    h_c = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.8, color=W1C, label="Wave 1"),
           plt.Line2D([], [], marker="s", linestyle="", markersize=3.8, color=W2C, label="Wave 2")]
    c.legend(handles=h_c + c.get_legend_handles_labels()[0], frameon=False, fontsize=5.6, loc="lower right", handlelength=1.6)
    style(c)

    # Panel D: Model routing (where constant OR fails)
    rb = q2["routing"]
    for t, r in rb.items():
        if t not in RBL:
            continue
        xs_r, ys_r = 10 * r["forecast"], r["scale_minus_risk"]
        col = INK if r["flag"] else MUTED
        d.plot([xs_r, xs_r], [10 * ys_r["lo95"], 10 * ys_r["hi95"]], color=col, linewidth=1.0)
        d.scatter([xs_r], [10 * ys_r["estimate"]], s=36 if not r["flag"] else 28, color=col if r["flag"] else SURFACE, edgecolor=col,
                  linewidth=1.4, zorder=4 if not r["flag"] else 3)
        off = {"routerbench__mixtral_to_gpt4": (14, 20), "routerbench__gpt35_to_gpt4": (20, 34),
               "routerbench__mistral7b_to_gpt4": (14, -22), "routerbench__llama70b_to_claude2": (-10, 16)}[t]
        d.annotate(RBL[t], (xs_r, 10 * ys_r["estimate"]), xytext=off, textcoords="offset points", fontsize=6.0, color=INK,
                   ha="right" if off[0] < 0 else "left",
                   arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 4})
    lim = [-50, 1150]
    d.plot(lim, lim, color=MUTED, linewidth=0.95, linestyle=(0, (4, 3)))
    d.axhline(0, color=MUTED, linewidth=0.65)
    d.set_xlim(-50, 1520)
    d.set_xlabel("Scale-predicted gap\n(prompts improved per 10,000 routed)", fontsize=7.2, color=INK)
    d.set_ylabel("Observed gain, scale over risk\n(prompts improved per 10,000 routed, 95% CI)", fontsize=7.2, color=INK)
    d.set_title("D. Model routing (known unit effects)", fontsize=7.4, loc="left", color=INK)
    h_d = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.2, color=INK, label="Flagged by diagnostic"),
           plt.Line2D([], [], marker="o", linestyle="", markersize=4.2, markerfacecolor=SURFACE, markeredgecolor=MUTED, color=MUTED,
                      label="Not flagged")]
    d.legend(handles=h_d, frameon=False, fontsize=6.0, loc="lower left")
    style(d)

    fig.tight_layout(pad=0.85, h_pad=1.0, w_pad=0.9)
    save(fig, "figure2")


def figure3(fr):
    q3 = fr["Q3"]
    fig = plt.figure(figsize=(7.0, 5.25), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.05], height_ratios=[2.55, 1.05], hspace=0.42, wspace=1.05)
    a = fig.add_subplot(gs[:, 0]); b = fig.add_subplot(gs[0, 1]); c = fig.add_subplot(gs[1, 1])
    pt = q3["dr"]["per_trial"]
    w1 = sorted([t for t in pt if not t.startswith("w2_")], key=lambda t: -pt[t]["estimate"])
    w2 = sorted([t for t in pt if t.startswith("w2_")], key=lambda t: -pt[t]["estimate"])
    y, ticks, labels = 0, [], []
    for group, col, mk in ((w1, W1C, "o"), (w2, W2C, "s")):
        for t in group:
            v = pt[t]
            a.plot([10 * v["lo95"], 10 * v["hi95"]], [y, y], color=col, linewidth=1.1)
            a.scatter([10 * v["estimate"]], [y], s=18, color=col, marker=mk, zorder=3)
            ticks.append(y); labels.append(LABEL[t]); y -= 1
        y -= 0.8
    a.axvline(0, color=MUTED, linewidth=0.7)
    a.set_yticks(ticks); a.set_yticklabels(labels, fontsize=6.5, color=INK); a.tick_params(axis="y", length=0)
    a.set_xlim(-1600, 2000)
    a.set_xlabel("Gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    a.set_title("A. Doubly robust over risk, by trial", fontsize=7.5, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4, color=W2C, label="Wave 2")]
    a.legend(handles=h, frameon=False, fontsize=6.5, loc="lower right")
    style(a, grid=False); a.spines["left"].set_visible(False)

    y, ticks, labels = 0, [], []
    rows_b = list(LEARNER.items())
    if "response_over_risk_paired" in q3:
        rows_b.insert(1, ("response_over_risk_paired", "Response ranking"))
    for k, name in rows_b:
        r = q3[k]
        for w, col, mk, wl in (("wave1", W1C, "o", "W1"), ("wave2", W2C, "s", "W2"), ("all", INK, "D", "all")):
            if w == "all":
                b.fill_between([10 * v for v in r["all"]["pi95"]], y - 0.3, y + 0.3, color=MUTED, alpha=0.18, linewidth=0)
            b.plot([10 * r[w]["lo95"], 10 * r[w]["hi95"]], [y, y], color=col, linewidth=1.5 if w == "all" else 1.1)
            b.scatter([10 * r[w]["mean"]], [y], s=22, color=col, marker=mk, zorder=3)
            ticks.append(y); labels.append(f"{name}, {wl} (k={r[w]['k']})"); y -= 1
        y -= 0.7
    b.axvline(0, color=MUTED, linewidth=0.7)
    b.set_yticks(ticks); b.set_yticklabels(labels, fontsize=6.2, color=INK); b.tick_params(axis="y", length=0)
    b.set_xlabel("Pooled gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    b.set_title("B. Pooled gain over the risk ranking", fontsize=7.5, loc="left", color=INK)
    b.text(0.99, 0.01, "Gray band: 95% PI, new trial", transform=b.transAxes, ha="right", va="bottom", fontsize=6.0, color=MUTED)
    style(b, grid=False); b.spines["left"].set_visible(False)

    y, ticks, labels = 0, [], []
    short_l = {"dr": "Doubly robust", "causal_forest": "Causal forest", "causalpfn": "CausalPFN", "rose": "Calibrated ens."}
    for k, name in short_l.items():
        for src, lab in (("learners_over_scale", "over scale"), ("learners_over_response", "over response")):
            r = q3[src][k]
            c.fill_between([10 * v for v in r["pi95"]], y - 0.3, y + 0.3, color=MUTED, alpha=0.18, linewidth=0)
            c.plot([10 * r["lo95"], 10 * r["hi95"]], [y, y], color=INK, linewidth=1.5)
            c.scatter([10 * r["mean"]], [y], s=22, color=INK, marker="D", zorder=3)
            ticks.append(y); labels.append(f"{name} {lab}"); y -= 1
        y -= 0.45
    c.axvline(0, color=MUTED, linewidth=0.7)
    c.set_yticks(ticks); c.set_yticklabels(labels, fontsize=6.2, color=INK); c.tick_params(axis="y", length=0)
    c.set_xlim(-300, 550)
    c.set_xlabel("Pooled gain over alternative\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    c.set_title("C. Learned over simple alternatives", fontsize=7.5, loc="left", color=INK)
    style(c, grid=False); c.spines["left"].set_visible(False)
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.16, right=0.97)
    save(fig, "figure3")


def figure4(b2):
    # PNAS full width = 7.0 in; 2 panels: A = circular plug-in vs held-out optimism; B = learning curves in 4 largest trials
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.95), facecolor=SURFACE)

    # Panel A: Plug-in vs held-out gain over risk across all 5 methods
    opt = b2["optimism_all_models"]
    order = [("dr", "Doubly robust\n(22/22 over)"),
             ("cpfn", "CausalPFN\n(21/22 over)"),
             ("cf", "Causal forest\n(19/22 over)"),
             ("rose", "Calibrated ens.\n(15/22 over)"),
             ("scale", "Scale ranking\n(7/17 over)")]
    ys = np.arange(len(order))[::-1]
    for y_i, (k, _) in zip(ys, order):
        v = opt[k]
        pl = 10 * v["mean_plugin"]
        ho = 10 * v["mean_heldout"]
        a.plot([ho, pl], [y_i, y_i], color=MUTED, linewidth=1.4, zorder=2)
        a.scatter([pl], [y_i], s=32, color=W2C, marker="s", zorder=4, label="Model plug-in" if k == "dr" else None)
        a.scatter([ho], [y_i], s=32, color=W1C, marker="o", zorder=4, label="Held-out AIPW" if k == "dr" else None)
        over_re = int(round(10 * v["mean"]))
        a.text(max(pl, ho) + 30, y_i, f"+{over_re}" if over_re > 0 else f"{over_re}",
               va="center", ha="left", fontsize=6.2, color=INK)
    a.axvline(0, color=MUTED, linewidth=0.7)
    a.set_yticks(ys); a.set_yticklabels([lbl for _, lbl in order], fontsize=6.2, color=INK)
    a.tick_params(axis="y", length=0)
    a.set_xlim(-60, 1270)
    a.set_xlabel("Mean gain over risk ranking\n(people helped per 10,000 reached)", fontsize=7.2, color=INK)
    a.set_title("A. Circular plug-in vs. held-out AIPW (exploratory)", fontsize=7.3, loc="left", color=INK)
    a.legend(frameon=False, fontsize=6.0, loc="lower right")
    style(a, grid=False); a.spines["left"].set_visible(False)

    # Panel B: Learning curves (DR minus Scale) in the 4 largest trials
    lc = b2["learning_curves_big4"]
    styles_lc = {
        "voter": (W1C, "o", "-", "Michigan mailer (ATE 769)"),
        "x5": (TEAL, "s", "-", "X5 text message (ATE 222)"),
        "lenta": (MUTED, "^", "--", "Lenta text message (ATE 48)"),
        "w2_robocall2008": (W2C, "D", "-", "Michigan robocall (ATE 73)"),
    }
    for t, (col, mk, ls, lbl) in styles_lc.items():
        pts_lc = sorted(lc[t].values(), key=lambda z: z["n_ctx"])
        ns = np.array([p["n_ctx"] / 1000.0 for p in pts_lc])
        ds = np.array([10 * p["dr_minus_scale"] for p in pts_lc])
        b.plot(ns, ds, color=col, marker=mk, linestyle=ls, linewidth=1.35, markersize=4.0, label=lbl)
    b.axhline(0, color=MUTED, linewidth=0.75, linestyle=(0, (4, 3)))
    b.set_xscale("log")
    b.set_xticks([5, 15, 50, 100, 350])
    b.set_xticklabels(["5k", "15k", "50k", "100k", "350k"])
    b.set_xlabel("Context training sample size (log scale)", fontsize=7.2, color=INK)
    b.set_ylabel("Doubly robust minus scale ranking\n(people helped per 10,000 reached)", fontsize=7.2, color=INK)
    b.set_title("B. Learning curves in the 4 largest trials (exploratory)", fontsize=7.3, loc="left", color=INK)
    b.legend(frameon=False, fontsize=5.7, loc="lower left")
    style(b)

    fig.tight_layout(pad=0.8)
    save(fig, "figure4")


def main() -> None:
    fr = json.loads((ROOT / "results" / "final_results.json").read_text())
    th = json.loads((ROOT / "results" / "theory_checks.json").read_text())
    b1 = json.loads((ROOT / "results" / "ben_exploratory_analyses.json").read_text())
    b2 = json.loads((ROOT / "results" / "ben_additional_exploratory.json").read_text())
    figure1(fr, th)
    figure2(fr, b1)
    figure3(fr)
    figure4(b2)


if __name__ == "__main__":
    main()
