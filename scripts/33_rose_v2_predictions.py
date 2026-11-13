"""Rose v2 predictions (src/system_three/rose_v2.py; registration update 5), on Modal CPU.

Fit on context, calibrate on the development split, answer for test states (test outcomes are never
read). Trials calibrate on doubly robust pseudo-outcomes; RouterBench pairs on exact development
effects. Output columns follow the frozen scorers' convention: benefit = s * mean, and lo90 / hi90
are on the same sign-adjusted scale (the scorer multiplies by s to return to the outcome scale).
Output: predictions_rose_v2/<dataset>.parquet; results/rose_v2_manifest.json (hashes, c, dev harm).
Usage: modal run scripts/33_rose_v2_predictions.py [--only a,b]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
TRIALS = ["gusto", "ist", "synbiotic", "elaia2", "voter", "sf_email", "karlan_list", "stepup",
          "hillstrom", "sixcity", "newhaven", "lenta", "x5"]
TIER_A = ["routerbench__mistral7b_to_gpt4", "routerbench__gpt35_to_gpt4",
          "routerbench__llama70b_to_claude2", "routerbench__mixtral_to_gpt4"]

app = modal.App("system-three-rose-v2")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(ROOT / "requirements.lock.txt"))
         .add_local_dir(ROOT / "src", "/root/src", copy=True))


@app.function(image=image, cpu=8, memory=32768, timeout=8 * 3600)
def run(name: str, ctx: tuple, dev: dict, Xt) -> dict:
    import sys

    sys.path.insert(0, "/root/src")
    from system_three import rose_v2

    f = rose_v2.fit(*ctx)
    f = rose_v2.calibrate(f, **dev)
    a = rose_v2.answer(f, Xt)
    return {"name": name, "c": f.c, "calibrated_on": f.calibrated_on, **{k: v.tolist() for k, v in a.items()}}


def prep_trial(t):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from system_three import rankers
    from system_three.analysis import prep
    from system_three.splits import load

    ctx = prep(t, "context")
    Xc, cols = rankers.design_matrix(ctx)
    dv = prep(t, "dev")
    Xd, _ = rankers.design_matrix(dv, cols)
    test = load(t, "test")
    Xt, _ = rankers.design_matrix(test, cols)
    c = (Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(), ctx["better"].iat[0],
         ctx["cluster_id"].to_numpy())
    d = {"Xd": Xd, "wd": dv["W"].to_numpy(), "yd": dv["outcome"].to_numpy(), "ed": dv["e"].to_numpy(),
         "gd": dv["cluster_id"].to_numpy()}
    return (t, c, d, Xt), test["row_id"].to_numpy(), ctx["better"].iat[0]


def prep_tierA(name):
    import importlib.util
    import sys

    import numpy as np

    sys.path.insert(0, str(ROOT / "src"))
    from system_three.splits import load

    spec = importlib.util.spec_from_file_location("ta", ROOT / "scripts" / "16_tierA_predictions.py")
    ta = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ta)
    ctx, dev, test = load(name, "context"), load(name, "dev"), load(name, "test")
    sc, sd, st = ta.parse(ctx["state"]), ta.parse(dev["state"]), ta.parse(test["state"])
    ok = set(ctx.groupby(sc["eval"].to_numpy())["outcome"].apply(lambda s: s.isin([0, 1]).all()).loc[lambda b: b].index)
    kc, kd, kt = sc["eval"].isin(ok).to_numpy(), sd["eval"].isin(ok).to_numpy(), st["eval"].isin(ok).to_numpy()
    ctx, sc = ctx[kc].reset_index(drop=True), sc[kc].reset_index(drop=True)
    dev, sd = dev[kd].reset_index(drop=True), sd[kd].reset_index(drop=True)
    test, st = test[kt].reset_index(drop=True), st[kt].reset_index(drop=True)
    Xc, Xt = ta.features(sc, st, "routerbench")
    _, Xd = ta.features(sc, sd, "routerbench")
    w = (ctx["action"] != ctx["reference"]).to_numpy(int)
    c = (Xc, w, ctx["outcome"].to_numpy(), np.full(len(ctx), 0.5), ctx["better"].iat[0], ctx["cluster_id"].to_numpy())
    d = {"Xd": Xd, "tau_d": (dev["y_act"] - dev["y_ref"]).to_numpy(), "yact_d": dev["y_act"].to_numpy(),
         "yref_d": dev["y_ref"].to_numpy(), "gd": dev["cluster_id"].to_numpy()}
    return (name, c, d, Xt), test["row_id"].to_numpy(), ctx["better"].iat[0]


@app.local_entrypoint()
def main(only: str = ""):
    import numpy as np
    import pandas as pd

    names = [n for n in only.split(",") if n] or TRIALS + TIER_A
    jobs, rows = [], {}
    for n in names:
        args, rid, better = prep_tierA(n) if n.startswith("routerbench") else prep_trial(n)
        jobs.append(args)
        rows[n] = (rid, better)
    out = ROOT / "predictions_rose_v2"
    out.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "rose_v2_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in run.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300])
            continue
        n = r["name"]
        base = pd.read_parquet(ROOT / ("predictions_tierA" if n.startswith("routerbench") else "predictions")
                               / f"{n}.parquet").set_index("row_id")
        rid, better = rows[n]
        p = base.loc[rid].reset_index()
        s = 1.0 if better == "higher" else -1.0
        p["benefit"] = s * np.asarray(r["mean"])
        lo, hi = s * np.asarray(r["lo90"]), s * np.asarray(r["hi90"])
        p["lo90"], p["hi90"] = np.minimum(lo, hi), np.maximum(lo, hi)
        for k in ("p_harm", "p_harm_ind", "p_harm_ind_calibrated", "fh_lo", "fh_hi", "support", "abstain"):
            p[k] = np.asarray(r[k])
        f = out / f"{n}.parquet"
        p.to_parquet(f, index=False)
        man[n] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "c": r["c"], "calibrated_on": r["calibrated_on"],
                  "abstain_rate": float(np.mean(r["abstain"]))}
        print(n, "ok c=%.3f on=%s" % (r["c"], r["calibrated_on"]), flush=True)
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
