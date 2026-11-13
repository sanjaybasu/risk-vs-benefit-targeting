"""Rose v0 predictions (Paper 2 reference model; src/system_three/rose.py), on Modal CPU.

For each confirmatory trial and each Tier A dataset: fit Rose on context, calibrate the interval
on the development split where exact truth exists (Tier A: tau = y_act - y_ref), and answer the
contract for every test state. Test outcomes are never read (test_state has none).

Output: predictions_rose/<dataset>.parquet in the scorer's format (benefit = sign-adjusted mean;
risk, scale, m0, m1 copied from the main or Tier A predictions) plus contract columns lo90, hi90,
p_harm, support, abstain; results/rose_manifest.json with hashes and the conformal factor.
Usage: modal run scripts/18_rose_predictions.py
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

app = modal.App("system-three-rose")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install_from_requirements(str(ROOT / "requirements.lock.txt"))
         .add_local_dir(ROOT / "src", "/root/src", copy=True))


@app.function(image=image, cpu=8, memory=32768, timeout=6 * 3600)
def run(name: str, Xc, w, y, e, groups, better, Xt, Xd=None, tau_d=None) -> dict:
    import sys

    sys.path.insert(0, "/root/src")
    from system_three import rose

    f = rose.fit(Xc, w, y, e, better, groups)
    if Xd is not None:
        f = rose.calibrate(f, Xd, tau_d)
    a = rose.answer(f, Xt)
    return {"name": name, "c": f.c, "calibrated": f.calibrated, **{k: v.tolist() for k, v in a.items()}}


def prep_trial(t):
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from system_three import rankers
    from system_three.analysis import prep
    from system_three.splits import load

    ctx = prep(t, "context")
    Xc, cols = rankers.design_matrix(ctx)
    test = load(t, "test")
    Xt, _ = rankers.design_matrix(test, cols)
    return (t, Xc, ctx["W"].to_numpy(), ctx["outcome"].to_numpy(), ctx["e"].to_numpy(),
            ctx["cluster_id"].to_numpy(), ctx["better"].iat[0], Xt), test["row_id"].to_numpy()


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
    ok = set(ctx.groupby(sc["eval"].to_numpy())["outcome"].apply(lambda s: s.isin([0, 1]).all())
             .loc[lambda b: b].index)
    kc, kd, kt = sc["eval"].isin(ok).to_numpy(), sd["eval"].isin(ok).to_numpy(), st["eval"].isin(ok).to_numpy()
    ctx, sc = ctx[kc].reset_index(drop=True), sc[kc].reset_index(drop=True)
    dev, sd = dev[kd].reset_index(drop=True), sd[kd].reset_index(drop=True)
    test, st = test[kt].reset_index(drop=True), st[kt].reset_index(drop=True)
    Xc, Xt = ta.features(sc, st, "routerbench")
    _, Xd = ta.features(sc, sd, "routerbench")
    w = (ctx["action"] != ctx["reference"]).to_numpy(int)
    tau_d = (dev["y_act"] - dev["y_ref"]).to_numpy()
    return ((name, Xc, w, ctx["outcome"].to_numpy(), np.full(len(ctx), 0.5),
             ctx["cluster_id"].to_numpy(), ctx["better"].iat[0], Xt, Xd, tau_d),
            test["row_id"].to_numpy())


@app.local_entrypoint()
def main(only: str = ""):
    import numpy as np
    import pandas as pd

    names = [n for n in only.split(",") if n] or TRIALS + TIER_A
    jobs, rows = [], {}
    for n in names:
        args, rid = (prep_tierA(n) if n.startswith("routerbench") else prep_trial(n))
        jobs.append(args)
        rows[n] = (rid, args[6])  # row ids and outcome direction
    out = ROOT / "predictions_rose"
    out.mkdir(exist_ok=True)
    man_path = ROOT / "results" / "rose_manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    for r in run.starmap(jobs, return_exceptions=True):
        if isinstance(r, Exception):
            print("FAILED", repr(r)[:300])
            continue
        n = r["name"]
        base_dir = "predictions_tierA" if n.startswith("routerbench") else "predictions"
        base = pd.read_parquet(ROOT / base_dir / f"{n}.parquet").set_index("row_id")
        rid, better = rows[n]
        p = base.loc[rid].reset_index()
        s = 1.0 if better == "higher" else -1.0
        p["benefit"] = s * np.asarray(r["mean"])
        for k in ("lo90", "hi90", "p_harm", "support", "abstain"):
            p[k] = np.asarray(r[k])
        f = out / f"{n}.parquet"
        p.to_parquet(f, index=False)
        man[n] = {"sha256": hashlib.sha256(f.read_bytes()).hexdigest(), "conformal_c": r["c"],
                  "calibrated": r["calibrated"], "abstain_rate": float(np.mean(r["abstain"]))}
        print(n, "ok c=%.2f abstain=%.3f" % (r["c"], man[n]["abstain_rate"]))
    man_path.write_text(json.dumps(man, indent=1, sort_keys=True))
