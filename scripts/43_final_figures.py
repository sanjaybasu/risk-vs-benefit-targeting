"""Figures 1-3 for the paper, read from results/final_results.json and results/theory_checks.json only.

Figure 1 (question 1): A, benefit against baseline risk under a constant odds ratio; B, the practical
check value against the posted scale-predicted gap in each eligible trial.
Figure 2 (question 2): A, observed gain of the scale ranking over the risk ranking against the posted
forecast, eligible trials, with the random-effects fit; B, model routing with known effects.
Figure 3 (question 3): A, doubly robust benefit ranking over risk ranking, per trial; B, pooled gain of
each ranking by wave and over all trials, with the new-trial 95% prediction interval; C, learned rankings
over the two simple alternatives.
All gains and gaps are scaled to per 10,000 reached (per 10,000 routed for model routing).
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.05), facecolor=SURFACE)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    r = np.linspace(0.001, 0.999, 400)
    u = np.log(r / (1 - r))
    for orr, ls in ((0.8, "-"), (0.5, (0, (5, 2))), (0.3, (0, (1, 1.5)))):
        c = -np.log(orr)
        tau = 10000 * (sig(u) - sig(u - c))
        a.plot(r, tau, color=INK, linestyle=ls, linewidth=1.35, label=f"Odds ratio {orr:g}")
        rs = sig(c / 2)
        a.scatter([rs], [10000 * (sig(c / 2) - sig(-c / 2))], s=18, color=INK, zorder=3)
    a.set_xlabel("Risk of undesirable outcome without action", fontsize=7.5, color=INK)
    a.set_ylabel("Benefit of the action\n(people helped per 10,000 reached)", fontsize=7.5, color=INK)
    a.set_title("A. Benefit peaks above 50% risk (constant odds ratio)", fontsize=7.5, loc="left", color=INK)
    a.legend(frameon=False, fontsize=6.5, loc="upper left")
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
    b.set_title(f"B. Practical check vs. posted gap ({len(elig)} trials)", fontsize=7.5, loc="left",
                color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4.5, color=W2C, label="Wave 2")]
    b.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left")
    style(b)
    fig.tight_layout(pad=0.8)
    save(fig, "figure1")


def figure2(fr):
    q2 = fr["Q2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.25), facecolor=SURFACE)
    pts = q2["points"]
    x = 10 * np.array([p["x"] for p in pts.values()])
    beta = np.array(q2["primary_fit"]["beta"], dtype=float).copy()
    vcov = np.array(q2["primary_fit"]["vcov"], dtype=float).copy()
    beta[0] *= 10.0
    vcov[0, 0] *= 100.0
    vcov[0, 1] *= 10.0
    vcov[1, 0] *= 10.0
    df = q2["primary_fit"]["df"]
    grid = np.linspace(-50, x.max() + 50, 100)
    G = np.column_stack([np.ones_like(grid), grid])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, vcov, G))
    crit = stats.t.ppf(0.975, df)
    a.fill_between(grid, G @ beta - crit * se, G @ beta + crit * se, color=MUTED, alpha=0.14, linewidth=0)
    w1 = q2["wave1_only"]
    a.plot(grid, G @ beta, color=INK, linewidth=1.6,
           label="19 trials: {:.2f} ({:.2f} to {:.2f})\nwave 1: {:.2f} ({:.2f} to {:.2f})".format(
               q2["primary"]["slope"], q2["primary"]["lo95"], q2["primary"]["hi95"], w1["slope"], w1["lo95"], w1["hi95"]))
    o = q2["informative_through_origin"]
    a.plot(grid, o["slope"] * grid, color=INK, linewidth=1.2, linestyle=(0, (1, 1.5)),
           label=f"16 trials, origin: {o['slope']:.2f} ({o['lo95']:.2f} to {o['hi95']:.2f})")
    a.plot(grid, grid, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)), label="Observed equals forecast")
    a.annotate("3 trials: forecast 0,\nobserved gain 0", (0, 0), xytext=(-55, 1140), textcoords="data", ha="left",
               fontsize=6.5, color=MUTED, arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.7, "relpos": (0.0, 0.0)})
    for t, p in pts.items():
        w2 = t.startswith("w2_")
        col = W2C if w2 else W1C
        px, py, plo, phi = 10 * p["x"], 10 * p["y"], 10 * p["lo95"], 10 * p["hi95"]
        a.plot([px, px], [plo, phi], color=col, linewidth=0.95, alpha=0.8)
        a.scatter([px], [py], s=24, color=col, marker="s" if w2 else "o", zorder=3)
        if p["x"] > 25:
            a.annotate(LABEL[t], (px, py), xytext={"voter": (-24, -24), "sixcity": (-38, -14), "hillstrom": (-34, 8), "newhaven": (4, 3)}.get(t, (4, 3)),
                       textcoords="offset points", fontsize=6.5, color=INK)
    a.set_title(f"A. Forecast and observed gain, {q2['k']} trials", fontsize=7.5, loc="left", color=INK)
    a.set_ylim(-1700, 1500)
    a.set_xlabel("Scale-predicted gap\n(people helped per 10,000 reached)", fontsize=7.5, color=INK)
    a.set_ylabel("Observed gain, scale over risk\n(people helped per 10,000 reached, 95% CI)", fontsize=7.5, color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4, color=W2C, label="Wave 2")]
    a.legend(handles=h + a.get_legend_handles_labels()[0], frameon=False, fontsize=6.2, loc="lower right", handlelength=1.8)
    style(a)
    rb = q2["routing"]
    for t, r in rb.items():
        if t not in RBL:
            continue
        xs, ys = 10 * r["forecast"], r["scale_minus_risk"]
        col = INK if r["flag"] else MUTED
        b.plot([xs, xs], [10 * ys["lo95"], 10 * ys["hi95"]], color=col, linewidth=1.0)
        b.scatter([xs], [10 * ys["estimate"]], s=40 if not r["flag"] else 30, color=col if r["flag"] else SURFACE, edgecolor=col,
                  linewidth=1.5, zorder=4 if not r["flag"] else 3)
        off = {"routerbench__mixtral_to_gpt4": (14, 22), "routerbench__gpt35_to_gpt4": (22, 38),
               "routerbench__mistral7b_to_gpt4": (14, -24), "routerbench__llama70b_to_claude2": (-10, 18)}[t]
        b.annotate(RBL[t], (xs, 10 * ys["estimate"]), xytext=off, textcoords="offset points", fontsize=6.5, color=INK,
                   ha="right" if off[0] < 0 else "left",
                   arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 4})
    lim = [-50, 1150]
    b.plot(lim, lim, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    b.axhline(0, color=MUTED, linewidth=0.7)
    b.set_xlim(-50, 1500)
    b.set_xlabel("Scale-predicted gap\n(prompts improved per 10,000 routed)", fontsize=7.5, color=INK)
    b.set_ylabel("Observed gain, scale over risk\n(prompts improved per 10,000 routed, 95% CI)", fontsize=7.5, color=INK)
    b.set_title("B. Model routing, where each unit's effect is known", fontsize=7.5, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=INK, label="Flagged by diagnostic"),
         plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, markerfacecolor=SURFACE, markeredgecolor=MUTED, color=MUTED,
                    label="Not flagged")]
    b.legend(handles=h, frameon=False, fontsize=6.5, loc="lower left")
    style(b)
    fig.tight_layout(pad=0.8)
    save(fig, "figure2")


def figure3(fr):
    q3 = fr["Q3"]
    # PNAS full two-column width = 17.8 cm = 7.0 in
    fig = plt.figure(figsize=(7.0, 5.8), facecolor=SURFACE)
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
            ticks.append(y)
            labels.append(LABEL[t])
            y -= 1
        y -= 0.8
    a.axvline(0, color=MUTED, linewidth=0.7)
    a.set_yticks(ticks)
    a.set_yticklabels(labels, fontsize=6.5, color=INK)
    a.tick_params(axis="y", length=0)
    a.set_xlim(-1600, 2000)
    a.set_xlabel("Gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    a.set_title("A. Doubly robust over risk, by trial", fontsize=7.5, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", markersize=4, color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", markersize=4, color=W2C, label="Wave 2")]
    a.legend(handles=h, frameon=False, fontsize=6.5, loc="lower right")
    style(a, grid=False)
    a.spines["left"].set_visible(False)
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
            ticks.append(y)
            labels.append(f"{name}, {wl} (k={r[w]['k']})")
            y -= 1
        y -= 0.7
    b.axvline(0, color=MUTED, linewidth=0.7)
    b.set_yticks(ticks)
    b.set_yticklabels(labels, fontsize=6.2, color=INK)
    b.tick_params(axis="y", length=0)
    b.set_xlabel("Pooled gain over risk ranking\n(people helped per 10,000 reached, 95% CI)", fontsize=7.2, color=INK)
    b.set_title("B. Pooled gain over the risk ranking", fontsize=7.5, loc="left", color=INK)
    b.text(0.99, 0.01, "Gray band: 95% PI, new trial", transform=b.transAxes, ha="right",
           va="bottom", fontsize=6.0, color=MUTED)
    style(b, grid=False)
    b.spines["left"].set_visible(False)
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
    style(c, grid=False)
    c.spines["left"].set_visible(False)
    fig.subplots_adjust(top=0.95, bottom=0.10, left=0.16, right=0.97)
    save(fig, "figure3")


def main() -> None:
    fr = json.loads((ROOT / "results" / "final_results.json").read_text())
    th = json.loads((ROOT / "results" / "theory_checks.json").read_text())
    figure1(fr, th)
    figure2(fr)
    figure3(fr)


if __name__ == "__main__":
    main()
