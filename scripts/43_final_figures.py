"""Figures 1-3 for the paper, read from results/final_results.json and results/theory_checks.json only.

Figure 1 (question 1): A, benefit against baseline risk under a constant odds ratio; B, the theorem's
condition value against the posted scale-predicted gap in each eligible trial.
Figure 2 (question 2): A, observed gain of the scale ranking over the risk ranking against the posted
forecast, eligible trials, with the random-effects fit; B, model routing with known effects.
Figure 3 (question 3): A, doubly robust benefit ranking over risk ranking, per trial; B, pooled gain of
each learned ranking by wave and over all trials, with the new-trial 95% prediction interval.
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
LEARNER = {"scale_over_risk": "Scale ranking", "dr": "Doubly robust", "causal_forest": "Causal forest", "causalpfn": "CausalPFN",
           "rose": "Calibrated ensemble"}


def style(ax, grid=True):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid:
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def figure1(fr, th):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.2), facecolor=SURFACE)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    r = np.linspace(0.001, 0.999, 400)
    u = np.log(r / (1 - r))
    for orr, ls in ((0.8, "-"), (0.5, (0, (5, 2))), (0.3, (0, (1, 1.5)))):
        c = -np.log(orr)
        tau = 1000 * (sig(u) - sig(u - c))
        a.plot(r, tau, color=INK, linestyle=ls, linewidth=1.6, label=f"Odds ratio {orr:g}")
        rs = sig(c / 2)
        a.scatter([rs], [1000 * (sig(c / 2) - sig(-c / 2))], s=22, color=INK, zorder=3)
    a.set_xlabel("Risk of the undesirable outcome without the action", fontsize=9, color=INK)
    a.set_ylabel("Benefit of the action\n(outcomes improved per 1,000 actions)", fontsize=9, color=INK)
    a.set_title("A. Benefit peaks above 50% risk under a constant odds ratio", fontsize=9, loc="left", color=INK)
    a.legend(frameon=False, fontsize=7.5, loc="upper left")
    style(a)
    elig = list(fr["Q2"]["points"])
    for t in elig:
        cv = th["trials"][t]["umax_plus_uq"] - th["trials"][t]["c"]
        g = fr["Q2"]["points"][t]["x"]
        w2 = t.startswith("w2_")
        b.scatter([cv], [g], s=36, color=W2C if w2 else W1C, marker="s" if w2 else "o", zorder=3)
        if cv < 0 or g > 30:
            b.annotate(LABEL[t], (cv, g), xytext={"gusto": (-6, 22), "w2_jtpa": (-10, 22)}.get(t, (5, 4)),
                       arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.6} if t in ("gusto", "w2_jtpa") else None, textcoords="offset points", fontsize=7.5, color=INK)
    b.axvline(0, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    b.set_xlabel("Condition value: $u_{max} + u_q - c$ (logit units)", fontsize=9, color=INK)
    b.set_ylabel("Posted scale-predicted gap\n(per 1,000 actions)", fontsize=9, color=INK)
    b.set_title(f"B. The condition separates zero and positive gaps ({len(elig)} trials)", fontsize=9, loc="left",
                color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", color=W2C, label="Wave 2")]
    b.legend(handles=h, frameon=False, fontsize=7.5, loc="upper left")
    style(b)
    fig.tight_layout()
    save(fig, "figure1")


def figure2(fr):
    q2 = fr["Q2"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 4.4), facecolor=SURFACE)
    pts = q2["points"]
    x = np.array([p["x"] for p in pts.values()])
    beta, vcov, df = np.array(q2["primary_fit"]["beta"]), np.array(q2["primary_fit"]["vcov"]), q2["primary_fit"]["df"]
    grid = np.linspace(-5, x.max() + 5, 100)
    G = np.column_stack([np.ones_like(grid), grid])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, vcov, G))
    crit = stats.t.ppf(0.975, df)
    a.fill_between(grid, G @ beta - crit * se, G @ beta + crit * se, color=MUTED, alpha=0.14, linewidth=0)
    a.plot(grid, G @ beta, color=INK, linewidth=2, label="19 trials: {:.2f} ({:.2f} to {:.2f})".format(q2["primary"]["slope"], q2["primary"]["lo95"], q2["primary"]["hi95"]))
    o = q2["informative_through_origin"]
    a.plot(grid, o["slope"] * grid, color=INK, linewidth=1.4, linestyle=(0, (1, 1.5)),
           label=f"16 trials, through origin: {o['slope']:.2f} ({o['lo95']:.2f} to {o['hi95']:.2f})")
    a.plot(grid, grid, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="Observed equals forecast")
    w1 = q2["wave1_only"]
    a.plot([], [], color="none", label=f"Wave 1 (registered primary): {w1['slope']:.2f} ({w1['lo95']:.2f} to {w1['hi95']:.2f})")
    a.annotate("3 trials at 0, 0\nby construction", (0, 0), xytext=(12, -105), textcoords="data", ha="left",
               fontsize=7.5, color=MUTED, arrowprops={"arrowstyle": "->", "color": MUTED, "linewidth": 0.8})
    for t, p in pts.items():
        w2 = t.startswith("w2_")
        col = W2C if w2 else W1C
        a.plot([p["x"], p["x"]], [p["lo95"], p["hi95"]], color=col, linewidth=1.1, alpha=0.8)
        a.scatter([p["x"]], [p["y"]], s=34, color=col, marker="s" if w2 else "o", zorder=3)
        if p["x"] > 25:
            a.annotate(LABEL[t], (p["x"], p["y"]), xytext={"voter": (6, -12), "sixcity": (-46, -12)}.get(t, (5, 3)), textcoords="offset points", fontsize=7.5, color=INK)
    s = q2["primary"]
    a.set_title(f"A. Forecast and observed gain, {q2['k']} trials", fontsize=9, loc="left", color=INK)
    a.set_ylim(-170, 150)
    a.set_xlabel("Scale-predicted gap, posted before scoring (per 1,000 actions)", fontsize=9, color=INK)
    a.set_ylabel("Observed gain of scale ranking over risk ranking\n(per 1,000 actions, 95% CI)", fontsize=9, color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", color=W2C, label="Wave 2")]
    a.legend(handles=h + a.get_legend_handles_labels()[0], frameon=False, fontsize=7.5, loc="lower right")
    style(a)
    rb = q2["routing"]
    for t, r in rb.items():
        if t not in RBL:
            continue
        xs, ys = r["forecast"], r["scale_minus_risk"]
        col = INK if r["flag"] else MUTED
        b.plot([xs, xs], [ys["lo95"], ys["hi95"]], color=col, linewidth=1.2)
        b.scatter([xs], [ys["estimate"]], s=40, color=col if r["flag"] else SURFACE, edgecolor=col, linewidth=1.6,
                  zorder=3)
        b.annotate(RBL[t], (xs, ys["estimate"]), xytext={"routerbench__mixtral_to_gpt4": (8, 8), "routerbench__gpt35_to_gpt4": (8, -16), "routerbench__mistral7b_to_gpt4": (8, -24)}.get(t, (6, 4)), textcoords="offset points",
                   fontsize=7.5, color=INK)
    lim = [-5, 115]
    b.plot(lim, lim, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    b.axhline(0, color=MUTED, linewidth=0.8)
    b.set_xlim(-5, 150)
    b.set_xlabel("Scale-predicted gap (per 1,000 actions)", fontsize=9, color=INK)
    b.set_ylabel("Observed gain of scale over risk ranking\n(per 1,000 actions, 95% CI)", fontsize=9, color=INK)
    b.set_title("B. Model routing, where each unit's effect is known", fontsize=9, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=INK, label="Flagged by context-only diagnostic"),
         plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=MUTED, color=MUTED,
                    label="Not flagged")]
    b.legend(handles=h, frameon=False, fontsize=7.5, loc="lower left")
    style(b)
    fig.tight_layout()
    save(fig, "figure2")


def figure3(fr):
    q3 = fr["Q3"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.4, 7.2), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.05, 1]})
    pt = q3["dr"]["per_trial"]
    w1 = sorted([t for t in pt if not t.startswith("w2_")], key=lambda t: -pt[t]["estimate"])
    w2 = sorted([t for t in pt if t.startswith("w2_")], key=lambda t: -pt[t]["estimate"])
    y, ticks, labels = 0, [], []
    for group, col, mk in ((w1, W1C, "o"), (w2, W2C, "s")):
        for t in group:
            v = pt[t]
            a.plot([v["lo95"], v["hi95"]], [y, y], color=col, linewidth=1.3)
            a.scatter([v["estimate"]], [y], s=24, color=col, marker=mk, zorder=3)
            ticks.append(y)
            labels.append(LABEL[t])
            y -= 1
        y -= 0.8
    a.axvline(0, color=MUTED, linewidth=0.8)
    a.set_yticks(ticks)
    a.set_yticklabels(labels, fontsize=7.5, color=INK)
    a.tick_params(axis="y", length=0)
    a.set_xlim(-160, 200)
    a.set_xlabel("Gain over risk ranking (per 1,000 actions, 95% CI)", fontsize=9, color=INK)
    a.set_title("A. Doubly robust ranking over the risk ranking, by trial", fontsize=9, loc="left", color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", color=W2C, label="Wave 2")]
    a.legend(handles=h, frameon=False, fontsize=7.5, loc="lower right")
    style(a, grid=False)
    a.spines["left"].set_visible(False)
    y, ticks, labels = 0, [], []
    for k, name in LEARNER.items():
        r = q3[k]
        for w, col, mk, wl in (("wave1", W1C, "o", "wave 1"), ("wave2", W2C, "s", "wave 2"), ("all", INK, "D", "all trials")):
            if w == "all":
                b.fill_between(r["all"]["pi95"], y - 0.3, y + 0.3, color=MUTED, alpha=0.18, linewidth=0)
            b.plot([r[w]["lo95"], r[w]["hi95"]], [y, y], color=col, linewidth=1.8 if w == "all" else 1.3)
            b.scatter([r[w]["mean"]], [y], s=30, color=col, marker=mk, zorder=3)
            ticks.append(y)
            labels.append(f"{name}, {wl} (k = {r[w]['k']})")
            y -= 1
        y -= 0.8
    b.axvline(0, color=MUTED, linewidth=0.8)
    b.set_yticks(ticks)
    b.set_yticklabels(labels, fontsize=7.5, color=INK)
    b.tick_params(axis="y", length=0)
    b.set_xlabel("Pooled gain over risk ranking (per 1,000 actions, 95% CI)", fontsize=9, color=INK)
    b.set_title("B. Pooled gain over the risk ranking, by ranking and wave", fontsize=9, loc="left", color=INK)
    b.text(0.99, 0.01, "Gray band: 95% prediction interval for a new trial", transform=b.transAxes, ha="right",
           va="bottom", fontsize=7.5, color=MUTED)
    style(b, grid=False)
    b.spines["left"].set_visible(False)
    fig.tight_layout()
    save(fig, "figure3")


def main() -> None:
    fr = json.loads((ROOT / "results" / "final_results.json").read_text())
    th = json.loads((ROOT / "results" / "theory_checks.json").read_text())
    figure1(fr, th)
    figure2(fr)
    figure3(fr)


if __name__ == "__main__":
    main()
