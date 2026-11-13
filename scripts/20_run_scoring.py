"""Run the frozen scorers on the sealed test split, in parallel on Modal (registration update 3).

The scoring logic is the frozen code: scripts/score_sealed.py (score_trial) and
scripts/19_score_contract.py (score), imported unchanged. Before each call the prediction file's
SHA-256 is checked against the posted manifest. Every call is logged with a UTC time stamp to
scoring/SCORING_LOG.jsonl, and every output is written to scoring/<stage>/<set>__<trial>.json and
hashed in scoring/<stage>/outputs_manifest.json. Failed calls are logged too.

Stage 1: non-holdout trials, all prediction sets, the exact-truth layer.
Stage 2: the holdouts (IST, StepUp, Hillstrom), run only after their forecasts are posted.
Usage: modal run scripts/20_run_scoring.py --stage stage1
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT.parent / "system-three-sealed"
HOLD = ["ist", "stepup", "hillstrom"]
NONHOLD = ["gusto", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "sixcity", "newhaven",
           "lenta", "x5"]
FLIPS = ["elaia1", "oregon"]  # excluded by the sign rule; scored for the registered sensitivity fit
TIER_A = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
          "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4",
          "swebench__p25_to_p75", "swebench__p50_to_p90", "swebench__p10_to_p50"]
RB = TIER_A[:4]


def jobs(stage: str) -> list[tuple[str, str, bool, str]]:
    """(prediction set, trial, subset mode, scorer) for a stage."""
    tr = NONHOLD if stage == "stage1" else HOLD
    out = [("predictions", t, False, "policy") for t in tr + (FLIPS if stage == "stage1" else [])]
    for s in ("predictions_causalpfn", "predictions_dopfn", "predictions_rose", "predictions_splitctx"):
        out += [(s, t, False, "policy") for t in tr]
    out += [("predictions_llm", t, True, "policy") for t in tr]
    out += [("predictions_loeo", t, False, "policy") for t in tr if t in ("elaia2", "synbiotic", "stepup", "voter")]
    out += [("predictions_rose", t, False, "contract") for t in tr]
    if stage == "stage1":
        out += [("predictions_tierA", t, True, "policy") for t in TIER_A]
        out += [("predictions_rose", t, True, "policy") for t in RB]
        out += [("predictions_rose", t, True, "contract") for t in RB]
    return out


app = modal.App("system-three-scoring")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(ROOT / "requirements.lock.txt"))
         .add_local_dir(ROOT / "src", "/root/src", copy=True)
         .add_local_file(ROOT / "scripts" / "score_sealed.py", "/root/scripts/score_sealed.py", copy=True)
         .add_local_file(ROOT / "scripts" / "19_score_contract.py", "/root/scripts/score_contract.py",
                         copy=True))


@app.function(image=image, cpu=2, memory=16384, timeout=4 * 3600)
def score(set_name: str, trial: str, subset: bool, kind: str, sealed_bytes: bytes, meta: dict,
          pred_bytes: bytes) -> dict:
    import importlib.util
    import io
    import sys

    import pandas as pd

    sys.path.insert(0, "/root/src")
    path = "/root/scripts/score_sealed.py" if kind == "policy" else "/root/scripts/score_contract.py"
    spec = importlib.util.spec_from_file_location("scorer", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sealed = pd.read_parquet(io.BytesIO(sealed_bytes))
    pred = pd.read_parquet(io.BytesIO(pred_bytes))
    if kind == "policy":
        res = mod.score_trial(sealed, pred, meta, subset=subset)
    else:
        res = mod.score(sealed, pred, meta, subset)
    return {"set": set_name, "trial": trial, "subset": subset, "kind": kind, "result": res,
            "scorer_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def posted_hash(set_name: str, trial: str) -> str:
    man = {"predictions": ("predictions/manifest.json", "predictions"),
           "predictions_causalpfn": ("results/causalpfn_manifest.json", "sha256"),
           "predictions_dopfn": ("results/dopfn_manifest.json", "sha256"),
           "predictions_llm": ("results/llm_manifest.json", "sha256"),
           "predictions_tierA": ("results/tierA_manifest.json", "sha256"),
           "predictions_rose": ("results/rose_manifest.json", "sha256"),
           "predictions_loeo": ("results/loeo_manifest.json", "sha256"),
           "predictions_splitctx": ("results/scale_forecast_splitctx.json", "sha256")}[set_name]
    return json.loads((ROOT / man[0]).read_text())[trial][man[1]]


@app.local_entrypoint()
def main(stage: str = "stage1"):
    out_dir = ROOT / "scoring" / stage
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ROOT / "scoring" / "SCORING_LOG.jsonl"
    now = lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")  # noqa: E731
    calls = []
    for set_name, trial, subset, kind in jobs(stage):
        pf = ROOT / set_name / f"{trial}.parquet"
        got = hashlib.sha256(pf.read_bytes()).hexdigest()
        if got != posted_hash(set_name, trial):
            raise SystemExit(f"{set_name}/{trial}: prediction hash does not match the posted manifest")
        meta = json.loads((SEALED / f"{trial}_meta.json").read_text())
        calls.append((set_name, trial, subset, kind, (SEALED / f"{trial}_test.parquet").read_bytes(),
                      meta, pf.read_bytes()))
    with log.open("a") as fh:
        fh.write(json.dumps({"utc": now(), "event": "start", "stage": stage, "n_calls": len(calls)}) + "\n")
    manifest = {}
    for r in score.starmap(calls, return_exceptions=True, order_outputs=False):
        if isinstance(r, Exception):
            with log.open("a") as fh:
                fh.write(json.dumps({"utc": now(), "event": "error", "stage": stage, "error": repr(r)[:500]}) + "\n")
            continue
        name = f"{r['set']}__{r['trial']}{'__contract' if r['kind'] == 'contract' else ''}.json"
        p = out_dir / name
        p.write_text(json.dumps(r, indent=1, default=float))
        manifest[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        with log.open("a") as fh:
            fh.write(json.dumps({"utc": now(), "event": "scored", "stage": stage, "file": name,
                                 "sha256": manifest[name], "scorer_sha256": r["scorer_sha256"]}) + "\n")
        print("scored", name, flush=True)
    (out_dir / "outputs_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    with log.open("a") as fh:
        fh.write(json.dumps({"utc": now(), "event": "end", "stage": stage, "n_outputs": len(manifest)}) + "\n")
