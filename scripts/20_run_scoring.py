"""Run the frozen scorers on this machine, one process per (prediction set, trial) (registration update 3).

The scoring code runs unchanged through its own command line: scripts/score_sealed.py and
scripts/19_score_contract.py. Each of those checks every prediction file against a flat manifest
of posted hashes and stops on a mismatch. This runner writes those flat manifests from the posted
ones, runs the jobs in parallel, and records:
  scoring/SCORING_LOG.jsonl              UTC start/end/exit code for every job, failures included
  scoring/<stage>/logs/<job>.log         stdout and stderr of every job
  scoring/<stage>/<job>.json             scorer output
  scoring/<stage>/outputs_manifest.json  SHA-256 of every output and log
The sealed folder is a command-line argument; this file names no sealed path.

Usage: uv run python scripts/20_run_scoring.py --sealed DIR --stage stage1 [--workers 12]
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOLD = ["ist", "stepup", "hillstrom"]
NONHOLD = ["gusto", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "sixcity", "newhaven",
           "lenta", "x5"]
FLIPS = ["elaia1", "oregon"]  # excluded by the sign rule; scored for the registered sensitivity fit
TIER_A = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
          "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4",
          "swebench__p25_to_p75", "swebench__p50_to_p90", "swebench__p10_to_p50"]
RB = TIER_A[:4]
POSTED = {"predictions": ("predictions/manifest.json", "predictions"),
          "predictions_causalpfn": ("results/causalpfn_manifest.json", "sha256"),
          "predictions_dopfn": ("results/dopfn_manifest.json", "sha256"),
          "predictions_llm": ("results/llm_manifest.json", "sha256"),
          "predictions_tierA": ("results/tierA_manifest.json", "sha256"),
          "predictions_rose": ("results/rose_manifest.json", "sha256"),
          "predictions_loeo": ("results/loeo_manifest.json", "sha256"),
          "predictions_splitctx": ("results/scale_forecast_splitctx.json", "sha256"),
          "predictions_llm_masked": ("results/llm_masked_manifest.json", "sha256"),
          "predictions_rose_v2": ("results/rose_v2_manifest.json", "sha256"),
          "predictions_cforest": ("results/cforest_manifest.json", "sha256")}


def jobs(stage: str) -> list[tuple[str, str, bool, str]]:
    """(prediction set, trial, subset mode, scorer) for a stage, in the registered order."""
    if stage == "cforest":  # registration update 7
        w2 = ["w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
              "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
        return [("predictions_cforest", t, False, "policy") for t in NONHOLD + HOLD + w2]
    if stage == "rose_v2_policy":  # exploratory: one Rose version (v2) across both waves; hashes in update 6
        return [("predictions_rose_v2", t, False, "policy") for t in NONHOLD + HOLD]
    if stage == "wave2":  # wave-2 registration; forecasts and hashes posted first (results/w2_forecasts.json)
        w2 = ["w2_robocall2008", "w2_sanmateo_email", "w2_wisconsin_recall", "w2_ffh_charity", "w2_sa_loan",
              "w2_mooc_megastudy", "w2_wa_bonus", "w2_jsa_florida", "w2_jtpa", "w2_turkey_overdraft"]
        return ([(s_, t, False, "policy") for s_ in ("predictions", "predictions_causalpfn", "predictions_rose_v2")
                 for t in w2] + [("predictions_rose_v2", t, False, "contract") for t in w2])
    if stage == "rose_v2":  # registration update 6
        return ([("predictions_rose_v2", t, False, "contract") for t in NONHOLD + HOLD]
                + [("predictions_rose_v2", t, True, "contract") for t in RB])
    if stage == "masked":  # registration updates 4-5
        return [("predictions_llm_masked", t, True, "policy") for t in NONHOLD + HOLD]
    tr = NONHOLD if stage == "stage1" else HOLD
    out = [("predictions", t, False, "policy") for t in tr + (FLIPS if stage == "stage1" else [])]
    for s in ("predictions_causalpfn", "predictions_dopfn", "predictions_rose", "predictions_splitctx"):
        out += [(s, t, False, "policy") for t in tr]
    out += [("predictions_llm", t, True, "policy") for t in tr]
    out += [("predictions_loeo", t, False, "policy") for t in tr
            if t in json.loads((ROOT / POSTED["predictions_loeo"][0]).read_text())]
    out += [("predictions_rose", t, False, "contract") for t in tr]
    if stage == "stage1":
        out += [("predictions_tierA", t, True, "policy") for t in TIER_A]
        out += [("predictions_rose", t, True, "policy") for t in RB]
        out += [("predictions_rose", t, True, "contract") for t in RB]
    return out


def flat_manifest(set_name: str) -> Path:
    path, key = POSTED[set_name]
    posted = json.loads((ROOT / path).read_text())
    flat = {t: (v[key] if isinstance(v, dict) else v) for t, v in posted.items()
            if isinstance(v, (dict, str)) and (not isinstance(v, dict) or key in v)}
    out = ROOT / "scoring" / "manifests" / f"{set_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(flat, indent=1, sort_keys=True))
    return out


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sealed", required=True, type=Path)
    ap.add_argument("--stage", required=True, choices=["stage1", "stage2", "masked", "rose_v2", "wave2", "rose_v2_policy", "cforest"])
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    out_dir = ROOT / "scoring" / a.stage
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)
    log = ROOT / "scoring" / "SCORING_LOG.jsonl"
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    todo = jobs(a.stage)
    mans = {s: flat_manifest(s) for s in {j[0] for j in todo}}

    def write(rec: dict) -> None:
        with log.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    write({"utc": now(), "event": "start", "stage": a.stage, "n_jobs": len(todo), "git": git})

    def run(job):
        set_name, trial, subset, kind = job
        name = f"{set_name}__{trial}{'__contract' if kind == 'contract' else ''}"
        script = "score_sealed.py" if kind == "policy" else "19_score_contract.py"
        cmd = [sys.executable, f"scripts/{script}", "--sealed", str(a.sealed), "--predictions", set_name,
               "--manifest", str(mans[set_name]), "--trials", trial, "--out", str(out_dir / f"{name}.json")]
        if subset:
            cmd.append("--allow-subset")
        t0 = now()
        with (out_dir / "logs" / f"{name}.log").open("w") as fh:
            rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
        write({"utc_start": t0, "utc_end": now(), "event": "job", "stage": a.stage, "job": name,
               "exit_code": rc})
        print(rc, name, flush=True)
        return name, rc

    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(run, todo))
    man = {p.relative_to(out_dir).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(out_dir.rglob("*")) if p.is_file() and p.name != "outputs_manifest.json"}
    (out_dir / "outputs_manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    write({"utc": now(), "event": "end", "stage": a.stage, "failed": [n for n, rc in res if rc]})


if __name__ == "__main__":
    main()
