"""Paper 2 contender: a frontier LLM asked who benefits (panel C29-C31), one pre-registered run.

Model: claude-opus-5-5 (flagship; the cheaper of the available flagships - no OpenAI key on this
machine), Message Batches (50% price), effort low, structured JSON output. The model sees a neutral
description of the trial (population, action, reference, outcome and direction; no published
results) and one person's pre-assignment covariates, and returns its estimate of the change in the
outcome's probability under the action. It sees no outcome data, so it is a no-data prior that
measures transfer of background knowledge, not identification (C31). It may know published
results of famous trials; that is part of what it measures.

Sample: per confirmatory trial, at most N_PER_TRIAL test states chosen by a salted hash of row_id
(no outcome is read). The frozen scorer scores the subset (--allow-subset) against the risk and
scale rankings on the same rows. Predictions are hashed and posted before unsealing.

Cost: about 26,000 requests; estimated $80-150 at batch prices. Pilot first.
Requests bypass the local compression proxy (base_url is api.anthropic.com).

Usage:
  uv run python scripts/13_llm_comparator.py build             # write requests (no API call)
  uv run python scripts/13_llm_comparator.py submit --pilot 50 # pilot batch
  uv run python scripts/13_llm_comparator.py submit            # full batch(es)
  uv run python scripts/13_llm_comparator.py collect           # fetch results, write predictions
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from system_three.splits import load

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "llm"
STATE = OUT / "batch_state.json"
MODEL = "claude-opus-5-5"
N_PER_TRIAL = 2000
SALT = "system-three-llm-sample-2026-11-12"
CONFIRMATORY = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup",
                "hillstrom", "sixcity", "newhaven", "lenta", "x5"]

TRIAL = {
    "gusto": ("adults with acute myocardial infarction within 6 hours of symptom onset, in a 1990s "
              "international trial", "accelerated tissue plasminogen activator (alteplase) with "
              "intravenous heparin", "streptokinase", "death within 30 days", "lower"),
    "ist": ("adults with suspected acute ischaemic stroke within 48 hours of onset, in a 1990s "
            "international trial", "aspirin 300 mg daily for up to 14 days", "avoid aspirin",
            "dead or dependent at 6 months", "lower"),
    "synbiotic": ("healthy newborns in rural India enrolled in the first days of life",
                  "an oral synbiotic (probiotic plus prebiotic) for 7 days", "placebo",
                  "sepsis or death within 60 days", "lower"),
    "elaia2": ("hospitalized adults with acute kidney injury who were receiving NSAIDs, PPIs, or "
               "RAAS inhibitors", "an automated electronic alert to clinicians recommending "
               "medication review", "usual care (no alert)",
               "AKI progression, dialysis, or death within 14 days", "lower"),
    "voter": ("registered voters in Michigan households before a 2006 primary election",
              "a mailing showing the household's and neighbors' past turnout and promising an "
              "update after the election", "no mailing", "voted in the primary", "higher"),
    "sf_email": ("registered voters in San Francisco with an email address on file, before a June "
                 "2010 election", "an email reminder from the official registrar of voters",
                 "no email", "voted in the election", "higher"),
    "karlan_list": ("prior donors to a US nonprofit receiving a fundraising letter",
                    "a letter announcing that a matching donor will match their gift",
                    "a standard letter with no match", "made a donation", "higher"),
    "stepup": ("members of a US gym chain who signed up for a 4-week exercise program study",
               "a digital behavior-change program (reminders, plans, incentives, or similar)",
               "a placebo program with minimal content", "visited the gym at least once during the "
               "4 weeks", "higher"),
    "hillstrom": ("customers of a US online retailer who purchased in the past 12 months",
                  "a marketing email featuring men's merchandise", "no email",
                  "visited the website within 2 weeks", "higher"),
    "sixcity": ("registered voters in six US cities before 2001 local elections",
                "a door-to-door canvassing visit encouraging them to vote", "no visit",
                "voted in the election", "higher"),
    "newhaven": ("registered voters in New Haven, Connecticut, before the 1998 election",
                 "a door-to-door canvassing visit encouraging them to vote", "no visit",
                 "voted in the election", "higher"),
    "lenta": ("customers of a Russian grocery chain with a loyalty card",
              "an SMS marketing message", "no message", "visited a store after the campaign",
              "higher"),
    "x5": ("customers of a Russian grocery chain with a loyalty card",
           "an SMS marketing message", "no message", "made a purchase after the campaign", "higher"),
}

SYSTEM = (
    "You estimate individual treatment effects for a research benchmark. For each person you are "
    "given a randomized study's population, the action, the comparison, the outcome, and the "
    "person's characteristics measured before assignment. Estimate how much the action changes "
    "the probability of the outcome for this specific person, compared with the comparison "
    "condition. Use your background knowledge of the intervention and of how such effects vary "
    "between people. Return the estimated change in probability as a number between -1 and 1 "
    "(for example 0.03 means the action raises the probability by 3 percentage points)."
)
SCHEMA = {"type": "object", "properties": {"effect": {"type": "number"}},
          "required": ["effect"], "additionalProperties": False}


def h(key: str) -> float:
    return int.from_bytes(hashlib.sha256(f"{SALT}|{key}".encode()).digest()[:8], "big") / 2**64


def sample(trial: str) -> pd.DataFrame:
    t = load(trial, "test")
    t = t.assign(_h=t["row_id"].astype(str).map(lambda r: h(f"{trial}|{r}")))
    return t.nsmallest(min(N_PER_TRIAL, len(t)), "_h").drop(columns="_h").reset_index(drop=True)


def person_text(row: pd.Series) -> str:
    feats = {c[2:]: row[c] for c in row.index if c.startswith("x_")}
    clean = {k: (None if (isinstance(v, float) and math.isnan(v)) else
                 (round(v, 4) if isinstance(v, float) else v)) for k, v in feats.items()}
    if isinstance(row.get("state"), str) and row["state"]:
        clean["details"] = row["state"]
    return json.dumps({k: v for k, v in clean.items() if v is not None})


def prompt(trial: str, row: pd.Series) -> str:
    pop, act, ref, outcome, better = TRIAL[trial]
    return (f"Study population: {pop}.\nAction: {act}.\nComparison: {ref}.\nOutcome: {outcome} "
            f"(higher probability is {'better' if better == 'higher' else 'worse'}).\n"
            f"Person's characteristics before assignment (variable names as in the study data): "
            f"{person_text(row)}\n\nEstimate the change in the probability of the outcome for this "
            f"person under the action compared with the comparison.")


def rebuild(trials: list[str]) -> None:
    """Rebuild requests for the named trials after a loader fix (deviation 2026-11-12): their old
    answers move to results_superseded.jsonl and their batch entries are retired, so submit only
    sends the new prompts and collect cannot mix old and new answers."""
    req = pd.read_parquet(OUT / "requests.parquet")
    old = req[req["trial"].isin(trials)]
    res_path = OUT / "results.jsonl"
    keep, sup = [], []
    for line in res_path.read_text().splitlines():
        (sup if json.loads(line)["custom_id"] in set(old["custom_id"]) else keep).append(line)
    (OUT / "results_superseded.jsonl").open("a").write("\n".join(sup) + "\n")
    res_path.write_text("\n".join(keep) + "\n")
    st = json.loads(STATE.read_text())
    for b in st["batches"]:
        b["custom_ids"] = [c for c in b["custom_ids"] if c not in set(old["custom_id"])]
        b.setdefault("retired_ids_superseded", 0)
    STATE.write_text(json.dumps(st))
    rows = []
    for t in trials:
        smp = sample(t)
        for _, r in smp.iterrows():
            rows.append({"custom_id": f"{t}__v2__{hashlib.sha1(str(r['row_id']).encode()).hexdigest()[:16]}",
                         "trial": t, "row_id": str(r["row_id"]), "prompt": prompt(t, r)})
    new = pd.concat([req[~req["trial"].isin(trials)], pd.DataFrame(rows)], ignore_index=True)
    new.to_parquet(OUT / "requests.parquet", index=False)
    print(f"rebuilt {len(rows):,} requests for {trials}; superseded {len(sup):,} answers")


def build() -> Path:
    OUT.mkdir(exist_ok=True)
    rows = []
    for t in CONFIRMATORY:
        s = sample(t)
        for _, r in s.iterrows():
            rows.append({"custom_id": f"{t}__{hashlib.sha1(str(r['row_id']).encode()).hexdigest()[:20]}",
                         "trial": t, "row_id": str(r["row_id"]), "prompt": prompt(t, r)})
    df = pd.DataFrame(rows)
    p = OUT / "requests.parquet"
    df.to_parquet(p, index=False)
    print(f"{len(df):,} requests; mean prompt chars {df['prompt'].str.len().mean():.0f}")
    return p


def client():
    import anthropic

    return anthropic.Anthropic(base_url="https://api.anthropic.com")  # bypass local proxy


def params(text: str) -> dict:
    return {"model": MODEL, "max_tokens": 4000, "system": SYSTEM,
            "messages": [{"role": "user", "content": text}],
            "output_config": {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}}}


def submit(pilot: int | None) -> None:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    req = pd.read_parquet(OUT / "requests.parquet")
    st = json.loads(STATE.read_text()) if STATE.exists() else {"batches": []}
    done = {cid for b in st["batches"] for cid in b["custom_ids"]}
    todo = req[~req["custom_id"].isin(done)]
    if pilot:
        todo = todo.groupby("trial", group_keys=False).head(max(1, pilot // len(CONFIRMATORY) + 1)).head(pilot)
    c = client()
    for i in range(0, len(todo), 10000):
        chunk = todo.iloc[i:i + 10000]
        b = c.messages.batches.create(requests=[
            Request(custom_id=r.custom_id, params=MessageCreateParamsNonStreaming(**params(r.prompt)))
            for r in chunk.itertuples()])
        st["batches"].append({"id": b.id, "custom_ids": chunk["custom_id"].tolist(),
                              "pilot": bool(pilot)})
        STATE.write_text(json.dumps(st))  # persist before waiting (batch gotchas)
        print("submitted", b.id, len(chunk))


def collect() -> None:
    c = client()
    st = json.loads(STATE.read_text())
    res_path = OUT / "results.jsonl"
    have = set()
    if res_path.exists():
        have = {json.loads(line)["custom_id"] for line in res_path.read_text().splitlines()}
    usage = {"input": 0, "output": 0}
    with res_path.open("a") as fh:
        for b in st["batches"]:
            info = c.messages.batches.retrieve(b["id"])
            if info.processing_status != "ended":
                print(b["id"], info.processing_status, info.request_counts)
                continue
            for r in c.messages.batches.results(b["id"]):
                if r.custom_id in have or r.custom_id not in set(b["custom_ids"]):
                    continue
                rec = {"custom_id": r.custom_id, "type": r.result.type}
                if r.result.type == "succeeded":
                    m = r.result.message
                    usage["input"] += m.usage.input_tokens
                    usage["output"] += m.usage.output_tokens
                    rec["stop_reason"] = m.stop_reason
                    text = next((blk.text for blk in m.content if blk.type == "text"), "")
                    try:
                        rec["effect"] = float(json.loads(text)["effect"])
                    except Exception:
                        rec["effect"] = None
                fh.write(json.dumps(rec) + "\n")
    cost = usage["input"] / 1e6 * 2.0 + usage["output"] / 1e6 * 10.0
    print(f"new tokens in {usage['input']:,} out {usage['output']:,}; batch cost ${cost:.2f}")


def write_predictions() -> None:
    """Assemble scorer-format files: benefit = sign-adjusted LLM effect; other columns from main."""
    req = pd.read_parquet(OUT / "requests.parquet")
    res = pd.DataFrame([json.loads(x) for x in (OUT / "results.jsonl").read_text().splitlines()])
    d = req.merge(res[["custom_id", "effect"]], on="custom_id", how="left")
    out = ROOT / "predictions_llm"
    out.mkdir(exist_ok=True)
    man = {}
    for t in CONFIRMATORY:
        sub = d[d["trial"] == t]
        main_p = pd.read_parquet(ROOT / "predictions" / f"{t}.parquet")
        main_p["row_id"] = main_p["row_id"].astype(str)
        p = main_p.merge(sub[["row_id", "effect"]], on="row_id", how="inner")
        better = TRIAL[t][4]
        s = 1.0 if better == "higher" else -1.0
        miss = int(p["effect"].isna().sum())
        p["benefit"] = s * p["effect"].fillna(0.0)  # missing answers rank as zero effect
        p = p.drop(columns=["effect"])
        f = out / f"{t}.parquet"
        p.to_parquet(f, index=False)
        man[t] = {"n": int(len(p)), "missing_answers": miss,
                  "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (ROOT / "results" / "llm_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    print(json.dumps({t: (v["n"], v["missing_answers"]) for t, v in man.items()}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["build", "rebuild", "submit", "collect", "predictions"])
    ap.add_argument("--trials", default="")
    ap.add_argument("--pilot", type=int, default=None)
    a = ap.parse_args()
    {"build": build, "rebuild": lambda: rebuild([t for t in a.trials.split(",") if t]),
     "submit": lambda: submit(a.pilot), "collect": collect,
     "predictions": write_predictions}[a.mode]()
