"""Tables 1 and 2 for the combined paper, written from results files only (no hand-entered values).

Table 1: the 13 confirmatory trials (class, outcome, test units scored, posted scale-predicted gap).
Table 2: per-trial gains over risk ranking at the 20% budget (per 1,000 actions, 95% CI) for the
scale ranking and each learned ranking, and the pooled random-effects rows.
Output: notebooks/system-three/paper/tables/table1.md, table2.md
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parents[1] / "notebooks" / "system-three" / "paper" / "tables"
CONF = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup", "hillstrom",
        "sixcity", "newhaven", "lenta", "x5"]
LABEL = {"gusto": "GUSTO-I", "ist": "IST", "synbiotic": "Synbiotic (neonatal sepsis)", "elaia2": "ELAIA-2",
         "voter": "Michigan social pressure", "sf_email": "SF email", "karlan_list": "Karlan–List matching",
         "stepup": "StepUp megastudy", "hillstrom": "Hillstrom email", "sixcity": "Six-city canvass",
         "newhaven": "New Haven canvass", "lenta": "Lenta", "x5": "X5"}
BIO = {"gusto", "ist", "synbiotic"}
HOLD = set(json.loads((ROOT / "results" / "holdouts.json").read_text())["holdouts"])
SETS = [("predictions", "scale-minus-risk", "Scale"), ("predictions", "benefit-minus-risk", "DR-learner"),
        ("predictions_rose", "benefit-minus-risk", "Rose"), ("predictions_causalpfn", "benefit-minus-risk", "CausalPFN"),
        ("predictions_dopfn", "benefit-minus-risk", "Do-PFN"), ("predictions_llm", "benefit-minus-risk", "Language model")]


def ci(r: dict) -> str:
    return f"{r['estimate']:.1f} ({r['lo95']:.1f} to {r['hi95']:.1f})"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dt = json.loads((ROOT / "results" / "dataset_table.json").read_text())
    fc = json.loads((ROOT / "results" / "scale_forecast.json").read_text())
    sm = json.loads((ROOT / "results" / "scoring_summary.json").read_text())
    dv = json.loads((ROOT / "results" / "derived_for_text.json").read_text())
    pt = sm["per_trial"]
    t1 = ["| Trial | Class | Outcome (better) | Test units scored | Scale-predicted gap, per 1,000 |",
          "|---|---|---|---|---|"]
    for t in CONF:
        name = LABEL[t] + (" (holdout)" if t in HOLD else "")
        t1.append(f"| {name} | {'Biological' if t in BIO else 'Behavioral'} | {dt[t]['outcome']} "
                  f"({dt[t]['better']}) | {pt[t]['predictions']['n']:,} | {fc[t]['gap']['0.2']:.1f} |")
    (OUT / "table1.md").write_text("Table 1. Confirmatory trials.\n\n" + "\n".join(t1) + "\n")
    head = "| Trial | " + " | ".join(lbl for _, _, lbl in SETS) + " |"
    t2 = [head, "|" + "---|" * (len(SETS) + 1)]
    for t in CONF:
        cells = [ci(pt[t][s][k]) if s in pt[t] else "–" for s, k, _ in SETS]
        t2.append(f"| {LABEL[t]} | " + " | ".join(cells) + " |")
    pooled = []
    for s, k, _ in SETS:
        r = sm["pooled_main_scale-minus-risk_q20"] if k == "scale-minus-risk" else sm["pooled_benefit_minus_risk_q20"][s]
        pooled.append(f"{r['mean']:.1f} ({r['lo95']:.1f} to {r['hi95']:.1f})")
    t2.append("| Pooled, random effects | " + " | ".join(pooled) + " |")
    note = ("\nValues are gains over risk ranking at the 20% budget, in outcomes improved per 1,000 actions, "
            "with 95% cluster-bootstrap CIs (pooled rows: random-effects, Knapp–Hartung, variance floor "
            "(2 per 1,000)²). The language model was scored on a hashed sample of up to 2,000 test units per "
            "trial, against risk ranking on the same units. In the scale column, the pooled value reflects the large weights of GUSTO-I and the synbiotic trial, whose gaps are 0 by construction; the unweighted mean of the 13 estimates is " + f"{dv['unweighted_mean']['scale-minus-risk']:.1f}" + ". The registered analysis tests how the scale gain varies with the scale-predicted gap (Figure 1A), not its mean. The pooling across trials was not pre-specified.\n")
    (OUT / "table2.md").write_text("Table 2. Gain over risk ranking by ranking method.\n\n" + "\n".join(t2) + "\n" + note)
    print((OUT / "table1.md").read_text())
    print((OUT / "table2.md").read_text())


if __name__ == "__main__":
    main()
