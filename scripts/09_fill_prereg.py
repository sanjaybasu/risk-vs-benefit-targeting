"""Fill pre-registration v1 placeholders from results files, draw the holdouts, and hash artifacts.

Reads results/{dataset_table, scale_forecast, h1_power_simulation, dev_baselines}.json and
predictions/manifest.json. Draws the forecast holdouts from the confirmatory set, one per tertile
of the posted moderator, with seed 20261113 (red-team F02), and writes results/holdouts.json.
Writes notebooks/system-three/preregistration/PREREGISTRATION_v1_filled.md and fails if any
{{TOKEN}} is left unfilled.

Usage: uv run python scripts/09_fill_prereg.py
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from system_three.analysis import prep

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
NB = ROOT.parents[1] / "notebooks" / "system-three" / "preregistration"
HOLDOUT_SEED = 20261113
Q = "0.2"

LABEL = {"gusto": "GUSTO-I", "ist": "International Stroke Trial", "synbiotic": "Synbiotic sepsis",
         "elaia1": "ELAIA-1", "elaia2": "ELAIA-2", "voter": "Michigan social pressure",
         "sf_email": "San Francisco email", "karlan_list": "Karlan–List", "stepup": "StepUp",
         "hillstrom": "Hillstrom", "oregon": "Oregon", "sixcity": "Six-city canvass",
         "newhaven": "New Haven", "lenta": "Lenta", "x5": "X5 RetailHero"}
CLASS = {  # (class, basis for the rule in Section 3)
    "gusto": ("Biological", "drug (thrombolytic), NEJM 1993 methods"),
    "ist": ("Biological", "drug (aspirin), Lancet 1997 methods"),
    "synbiotic": ("Biological", "oral synbiotic, Nature 2017 methods"),
    "elaia1": ("Behavioral", "EHR alert to clinicians, BMJ 2021 methods"),
    "elaia2": ("Behavioral", "EHR alert to clinicians, Nat Commun 2023 methods"),
    "voter": ("Behavioral", "mailing, APSR 2008"), "sf_email": ("Behavioral", "email, QJPS 2012"),
    "karlan_list": ("Behavioral", "solicitation letter, AER 2007"),
    "stepup": ("Behavioral", "digital programs, Nature 2021"),
    "hillstrom": ("Behavioral", "marketing email, challenge description"),
    "oregon": ("Behavioral", "insurance lottery, Science 2014"),
    "sixcity": ("Behavioral", "door-to-door canvass, J Polit 2003"),
    "newhaven": ("Behavioral", "door-to-door canvass, APSR 2000"),
    "lenta": ("Behavioral", "SMS campaign, dataset description"),
    "x5": ("Behavioral", "SMS campaign, competition description")}
UNIT = {"voter": "household", "sf_email": "email address", "oregon": "household",
        "sixcity": "household", "newhaven": "household", "synbiotic": "infant (Study_ID)"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    dt_ = json.loads((RES / "dataset_table.json").read_text())
    fc = json.loads((RES / "scale_forecast.json").read_text())
    sim = json.loads((RES / "h1_power_simulation.json").read_text())
    conf = sim["confirmatory_trials"]
    excluded = sim["excluded_by_sign_rule"]
    # Holdouts: one per tertile of the posted moderator (F02).
    xs = sorted(conf, key=lambda t: fc[t]["gap"][Q])
    tert = np.array_split(np.array(xs), 3)
    rng = np.random.default_rng(HOLDOUT_SEED)
    holdouts = [str(rng.choice(g)) for g in tert]
    (RES / "holdouts.json").write_text(json.dumps({"seed": HOLDOUT_SEED, "tertiles": [list(map(str, g)) for g in tert],
                                                    "holdouts": holdouts}, indent=1))
    # Table D
    rows = ["| Trial | Class (basis) | n (contrast) | Unit / clusters, all arms | Design SE | Max IPW weight | "
            "Reproduction |", "|---|---|---|---|---|---|---|"]
    for t in list(conf) + list(excluded):
        r, f = dt_[t], fc[t]
        rep = r["reproduction"]
        if rep["published"] is None:
            repro = "no verified target"
        elif rep["metric"] == "n_units":
            repro = f"{int(rep['reproduced']):,} vs {int(rep['published']):,} participants ({rep['status']})"
        else:
            repro = f"{rep['reproduced']:.4g} vs {rep['published']:g} ({rep['status']})"
        cls, basis = CLASS[t]
        se = 1000 * r["mde_per_action_at_q20"] / 2.8
        # Exact two-arm count: context and dev from the frozen prep, test from the seal step.
        n_contrast = len(prep(t, "context")) + len(prep(t, "dev")) + r["test"]
        rows.append(f"| {LABEL[t]}{' (sign rule: out)' if t in excluded else ''} | {cls} ({basis}) | "
                    f"{n_contrast:,} | {UNIT.get(t, 'individual')} / {r['clusters']:,} | {se:.1f} | "
                    f"{f['max_ipw_weight']:.1f} | {repro} |")
    # Table M
    m = ["| Trial | Base rate | β (95% CI) | AUC of p̂0 | Posted gap x at 20% (SE) | Half-sample mean | "
         "Beneficial sign |", "|---|---|---|---|---|---|---|"]
    for t in list(conf) + list(excluded):
        f = fc[t]
        m.append(f"| {LABEL[t]} | {f['control_base_rate']:.3f} | {f['beta']:+.3f} "
                 f"({f['beta_ci95'][0]:+.3f}, {f['beta_ci95'][1]:+.3f}) | {f['auc_p0_oof']:.3f}"
                 f"{' (no risk signal)' if f['auc_p0_oof'] < 0.60 else ''} | {f['gap'][Q]:+.2f} "
                 f"({f['gap_se'][Q]:.2f}) | {f['gap_halfsample_mean'][Q]:+.2f} | "
                 f"{'yes' if f['beneficial_sign'] else 'no'} |")
    # Power table
    g = sim["grid"]
    p = ["| Sampling noise | True slope | τ = 0 | τ = 10 | τ = 20 | τ = 30 |", "|---|---|---|---|---|---|"]
    for noise in ("paired", "conservative"):
        for slope in (0.0, 0.5, 1.0):
            cells = [g.get(f"noise={noise},slope={slope},tau={tau},trunc=False", {}).get("p_weak_form_positive")
                     for tau in (0, 10, 20, 30)]
            if None in cells:
                continue
            p.append(f"| {noise} | {slope} | " + " | ".join(f"{c:.2f}" for c in cells) + " |")
    type1 = [g[k]["p_weak_form_positive"] for k in g if ",slope=0.0," in k and "trunc=False" in k
             and not k.startswith("weights=")]
    size_p = [g[k]["p_weak_form_positive"] for k in g if k.startswith("noise=paired,slope=0.0,") and "trunc=False" in k]
    size_d = [g[k]["p_weak_form_positive"] for k in g if k.startswith("weights=design")]
    max_x = max(fc[t]["gap"][Q] for t in conf)
    low = [LABEL[t] for t in conf if 1000 * dt_[t]["mde_per_action_at_q20"] / 2.8 > max_x]
    inf = ["| Trial | Posted gap | Hat value | Share of slope information |", "|---|---|---|---|"]
    for t, r in sorted(sim["influence"].items(), key=lambda kv: -kv[1]["slope_information_share"]):
        inf.append(f"| {LABEL[t]} | {r['x']:+.2f} | {r['hat']:.2f} | {r['slope_information_share']:.2f} |")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    src_hash = hashlib.sha256(b"".join(sha(p).encode() for p in sorted((ROOT / "src").rglob("*.py")))).hexdigest()
    fill = {
        "DATE": dt.date.today().isoformat(), "DATE_M": dt.date.today().isoformat(),
        "TABLE_D": "\n".join(rows), "TABLE_M": "\n".join(m), "TABLE_POWER": "\n".join(p),
        "TABLE_INFLUENCE": "\n".join(inf),
        "TYPE1": f"{min(type1):.3f} to {max(type1):.3f} one-sided across τ and noise settings",
        "UNVERIFIED_REPRO": ", ".join(LABEL[t] for t in conf if dt_[t]["reproduction"]["published"] is None),
        "EXCLUDED_SIGN": ", ".join(LABEL[t] for t in excluded),
        "CONFIRMATORY": ", ".join(LABEL[t] for t in conf), "K": str(len(conf)),
        "N_HOLDOUT": str(len(holdouts)), "SEED_HOLDOUT": str(HOLDOUT_SEED),
        "HOLDOUTS": ", ".join(LABEL[t] for t in holdouts),
        "COMMIT": commit,
        "SHA_SCORER": sha(ROOT / "scripts" / "score_sealed.py")[:16],
        "SHA_META": sha(ROOT / "scripts" / "07_meta_regression.py")[:16],
        "SHA_FORECAST": f"script {sha(ROOT / 'scripts' / '06_scale_forecast.py')[:16]}, src {src_hash[:16]}",
        "SHA_DEV": sha(RES / "dev_baselines.json")[:16] if (RES / "dev_baselines.json").exists() else "MISSING",
        "SHA_UVLOCK": sha(ROOT / "uv.lock")[:16],
        "SIZE_PAIRED": f"{min(size_p):.3f} to {max(size_p):.3f}",
        "SIZE_DESIGN": f"{min(size_d):.3f} to {max(size_d):.3f}",
        "LOW_POWER": ", ".join(low), "MAX_X": f"{max_x:.1f}",
    }
    text = (NB / "PREREGISTRATION_v1.md").read_text()
    for k, v in fill.items():
        text = text.replace("{{" + k + "}}", v)
    left = re.findall(r"\{\{[A-Z_0-9]+\}\}", text)
    if left or "MISSING" in text:
        raise SystemExit(f"unfilled: {left} {'MISSING' if 'MISSING' in text else ''}")
    (NB / "PREREGISTRATION_v1_filled.md").write_text(text)
    print("holdouts:", holdouts, "| commit:", commit)


if __name__ == "__main__":
    main()
