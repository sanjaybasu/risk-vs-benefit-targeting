"""Figure 1 for Paper 1: observed scale-minus-risk gap against the posted forecast.

Panel A: the confirmatory trials. y = observed V(scale) - V(risk) at 20% from the custodian scorer
output (95% CI), x = posted scale-predicted gap. Identity line (perfect calibration), fitted
random-effects meta-regression line with its 95% band (07_meta_regression.fit_re, primary
weights), biological trials in orange and behavioral in blue, holdouts as hollow markers, direct
labels on the trials that carry the slope. Panel B: the exact-truth layer (Tier A), same axes logic.

Reads only results files and the scorer output; no hand-entered values.
Usage: uv run python scripts/17_figure1.py --scored S.json [--tierA T.json] --out DIR [--synthetic]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BEHAVIORAL, BIOLOGICAL = "#2a78d6", "#eb6834"  # validated categorical slots 1-2 (dataviz palette)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
LABEL = {"gusto": "GUSTO-I", "ist": "IST", "synbiotic": "Synbiotic", "elaia2": "ELAIA-2",
         "voter": "Michigan", "sf_email": "SF email", "karlan_list": "Karlan–List", "stepup": "StepUp",
         "hillstrom": "Hillstrom", "sixcity": "Six-city", "newhaven": "New Haven", "lenta": "Lenta",
         "x5": "X5"}

# Fixed label offsets (points) to avoid collisions between neighboring trials.
OFFSET = {"voter": (-28, 10), "hillstrom": (6, -14), "ist": (8, 2), "gusto": (-4, -30), "stepup": (-44, 4)}
OFFSET_B = {"routerbench__gpt35_to_gpt4": (6, -12), "routerbench__mixtral_to_gpt4": (6, 6)}

spec = importlib.util.spec_from_file_location("meta", ROOT / "scripts" / "07_meta_regression.py")
meta = importlib.util.module_from_spec(spec)
sys.modules["meta"] = meta
spec.loader.exec_module(meta)


def gap_rows(scored: dict, key: str = "scale-minus-risk", q: float = 0.2) -> dict:
    out = {}
    for t, r in scored["trials"].items():
        row = next(x for x in r["table"] if x["ranking"] == key and x["budget"] == q)
        out[t] = (row["estimate"], row["lo95"], row["hi95"])
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def panel_a(ax, scored: dict):
    post = meta.posted()
    trials = meta.confirmatory(post)
    holdouts = set(json.loads((ROOT / "results" / "holdouts.json").read_text())["holdouts"])
    obs = gap_rows(scored)
    x = np.array([post[t]["x"] for t in trials])
    y = np.array([obs[t][0] for t in trials])
    v = np.array([post[t]["v_primary"] for t in trials])
    X = np.column_stack([np.ones_like(x), x])
    f = meta.fit_re(y, X, v)
    grid = np.linspace(min(0, x.min()) - 5, x.max() + 5, 100)
    G = np.column_stack([np.ones_like(grid), grid])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, f.vcov, G))
    from scipy import stats

    crit = stats.t.ppf(0.975, f.df)
    fit = G @ f.beta
    ax.fill_between(grid, fit - crit * se, fit + crit * se, color=MUTED, alpha=0.12, linewidth=0)
    ax.plot(grid, fit, color=INK, linewidth=2, label="Fitted (random effects)")
    ax.plot(grid, grid, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), label="Perfect calibration")
    for t, xi, yi in zip(trials, x, y):
        col = BIOLOGICAL if t in meta.BIOLOGICAL else BEHAVIORAL
        lo, hi = obs[t][1], obs[t][2]
        ax.plot([xi, xi], [lo, hi], color=col, linewidth=1.2, alpha=0.8)
        hollow = t in holdouts
        ax.scatter([xi], [yi], s=44, color=SURFACE if hollow else col, edgecolor=col, linewidth=1.8,
                   zorder=3)
        if abs(xi) > 20 or hollow or t == "gusto":
            ax.annotate("GUSTO-I and Synbiotic" if t == "gusto" else LABEL[t], (xi, yi), xytext=OFFSET.get(t, (6, 4)), textcoords="offset points",
                        fontsize=8, color=INK)
    ax.set_xlabel("Posted scale-predicted gap (per 1,000 actions)", fontsize=9, color=INK)
    ax.set_ylabel("Observed gain of scale over risk ranking\n(per 1,000 actions, 95% CI)", fontsize=9,
                  color=INK)
    s = meta.slope_test(f)
    ax.set_title(f"A. Randomized trials (k = {len(trials)}); slope {s['slope']:.2f} "
                 f"(95% CI {s['lo95']:.2f} to {s['hi95']:.2f})", fontsize=9, loc="left", color=INK)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=BEHAVIORAL, label="Behavioral"),
               plt.Line2D([], [], marker="o", linestyle="", color=BIOLOGICAL, label="Biological"),
               plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE,
                          markeredgecolor=MUTED, color=MUTED, label="Forecast holdout")]
    ax.legend(handles=handles + ax.get_legend_handles_labels()[0], frameon=False, fontsize=7.5,
              loc="upper left")
    style(ax)


def panel_b(ax, tier_scored: dict):
    man = json.loads((ROOT / "results" / "tierA_manifest.json").read_text())
    obs = gap_rows(tier_scored)
    names = [n for n in man if n in obs and man[n]["auc_p0_oof"] >= 0.60]
    x = np.array([man[n]["scale_forecast_gap_q20"] for n in names])
    y = np.array([obs[n][0] for n in names])
    for n, xi, yi in zip(names, x, y):
        ax.plot([xi, xi], [obs[n][1], obs[n][2]], color=BEHAVIORAL, linewidth=1.2)
        ax.scatter([xi], [yi], s=44, color=BEHAVIORAL, zorder=3)
        ax.annotate(n.split("__")[1].replace("_to_", "→"), (xi, yi), xytext=OFFSET_B.get(n, (6, 4)),
                    textcoords="offset points", fontsize=7.5, color=INK)
    lim = [min(0, x.min(), y.min()) - 5, max(x.max(), y.max()) + 10]
    ax.plot(lim, lim, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.set_xlabel("Context-only forecast (per 1,000)", fontsize=9, color=INK)
    ax.set_ylabel("Observed gain, exact truth (per 1,000)", fontsize=9, color=INK)
    ax.set_title("B. Exact-truth layer (LLM routing)", fontsize=9, loc="left", color=INK)
    style(ax)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, type=Path)
    ap.add_argument("--tierA", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    ncol = 2 if a.tierA else 1
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 4.4), facecolor=SURFACE)
    axes = np.atleast_1d(axes)
    panel_a(axes[0], json.loads(a.scored.read_text()))
    if a.tierA:
        panel_b(axes[1], json.loads(a.tierA.read_text()))
    if a.synthetic:
        fig.text(0.5, 0.5, "SYNTHETIC — NOT RESULTS", ha="center", va="center", fontsize=28,
                 color="#e34948", alpha=0.25, rotation=20)
    fig.tight_layout()
    a.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(a.out / f"figure1.{ext}", dpi=200, facecolor=SURFACE)
    print("wrote", a.out / "figure1.png")


if __name__ == "__main__":
    main()
