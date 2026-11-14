"""Figure 2: gain of the doubly robust benefit ranking over risk ranking, per trial and wave, with the
pooled mean (95% CI) and the 95% prediction interval for a new trial. Reads results only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "notebooks" / "system-three" / "paper" / "figures"
INK, MUTED, SURFACE, W1C, W2C = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6", "#eb6834"
LABEL = {"gusto": "GUSTO-I", "ist": "IST", "synbiotic": "Synbiotic", "elaia2": "ELAIA-2", "voter": "Michigan mailer",
         "sf_email": "SF email", "karlan_list": "Karlan–List", "stepup": "StepUp", "hillstrom": "Hillstrom",
         "sixcity": "Six-city", "newhaven": "New Haven", "lenta": "Lenta", "x5": "X5",
         "w2_robocall2008": "Michigan robocall", "w2_sanmateo_email": "San Mateo email",
         "w2_wisconsin_recall": "Wisconsin letter", "w2_ffh_charity": "Freedom from Hunger", "w2_sa_loan": "SA loan offer",
         "w2_mooc_megastudy": "edX plan-making", "w2_wa_bonus": "WA bonus", "w2_jsa_florida": "JSA Florida",
         "w2_jtpa": "JTPA", "w2_turkey_overdraft": "Turkey overdraft"}


def main() -> None:
    r = json.loads((ROOT / "results" / "combined_learned_rankings.json").read_text())["dr"]
    rows = sorted(r["per_trial"].items(), key=lambda kv: (kv[1]["wave"], -kv[1]["estimate"]))
    fig, ax = plt.subplots(figsize=(6.2, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(rows)
    for i, (t, v) in enumerate(rows):
        y = n - i + (0 if v["wave"] == 1 else -1)
        col = W1C if v["wave"] == 1 else W2C
        ax.plot([v["lo95"], v["hi95"]], [y, y], color=col, linewidth=1.4)
        ax.scatter([v["estimate"]], [y], s=26, color=col, zorder=3, marker="o" if v["wave"] == 1 else "s")
        ax.text(-200, y, LABEL[t], va="center", ha="left", fontsize=7.5, color=INK)
    yp = -1.5
    ax.fill_between(r["pi95_new_trial"], yp - 0.35, yp + 0.35, color=MUTED, alpha=0.18, linewidth=0)
    ax.plot([r["lo95"], r["hi95"]], [yp, yp], color=INK, linewidth=2.4)
    ax.scatter([r["mean"]], [yp], s=44, color=INK, marker="D", zorder=3)
    ax.text(-200, yp, "Pooled, 23 trials\n(band: new-trial 95% PI)", va="center", ha="left", fontsize=7.5, color=INK)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlim(-200, 200)
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel("Gain of benefit ranking over risk ranking at 20% budget\n(outcomes improved per 1,000 actions, 95% CI)",
                  fontsize=8.5, color=INK)
    h = [plt.Line2D([], [], marker="o", linestyle="", color=W1C, label="Wave 1"),
         plt.Line2D([], [], marker="s", linestyle="", color=W2C, label="Wave 2 (pre-registered replication)")]
    ax.legend(handles=h, frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure2.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", OUT / "figure2.png")


if __name__ == "__main__":
    main()
