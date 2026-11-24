"""Score cluster-respecting causal forest across all 28 trials (Waves 1-2 + Wave 3) and build unified 28-trial summary.

Reads:
  - predictions_cforest_cluster/<trial>.parquet (SHA-256 verified against results/cforest_cluster_manifest.json)
  - results/final_results.json, results/ben_exploratory_analyses.json, results/ben_additional_exploratory.json
  - results/wave3_results.json, results/continuous_count_results.json

Writes:
  - results/cluster_cf_and_28trial_summary.json
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT.parent / "system-three-sealed"
R = ROOT / "results"
sys.path.insert(0, str(ROOT / "src"))

spec_mr = importlib.util.spec_from_file_location("mr", ROOT / "scripts" / "07_meta_regression.py")
mr = importlib.util.module_from_spec(spec_mr)
sys.modules["mr"] = mr
spec_mr.loader.exec_module(mr)

spec_s50 = importlib.util.spec_from_file_location("s50", ROOT / "scripts" / "50_ben_additional_exploratory.py")
s50 = importlib.util.module_from_spec(spec_s50)
spec_s50.loader.exec_module(s50)

from system_three.estimators.dr import aipw_scores  # noqa: E402


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    fr = json.loads((R / "final_results.json").read_text())
    b2 = json.loads((R / "ben_additional_exploratory.json").read_text())
    w3 = json.loads((R / "wave3_results.json").read_text())
    man = json.loads((R / "cforest_cluster_manifest.json").read_text())

    w12 = fr["trials"]["wave1"] + fr["trials"]["wave2"]
    w3_ts = w3["trials"]
    all28 = w12 + w3_ts

    cf_per_trial = {}
    for t in all28:
        p_cf = ROOT / "predictions_cforest_cluster" / f"{t}.parquet"
        assert sha256_file(p_cf) == man[t]["sha256"], f"Hash mismatch for {t}"
        cf_df = pd.read_parquet(p_cf)
        sealed = pd.read_parquet(SEALED / f"{t}_test.parquet")
        smeta = json.loads((SEALED / f"{t}_meta.json").read_text())
        two = sealed[sealed["action"].isin([smeta["action"], smeta["reference"]])]
        d = two.merge(cf_df, on="row_id", how="inner", validate="1:1").reset_index(drop=True)
        w = (d["action"] == smeta["action"]).to_numpy(int)
        if smeta.get("p_action") is not None:
            e = np.full(len(d), smeta["p_action"] / (smeta["p_action"] + smeta["p_reference"]))
        else:
            pr = d["propensity"].to_numpy(float)
            e = np.where(w == 1, pr, 1.0 - pr)
        g = aipw_scores(d["outcome"].to_numpy(float), w, e, d["m0"].to_numpy(float), d["m1"].to_numpy(float), smeta["better"])
        cl = d["cluster_id"].to_numpy()
        resp = d["m1"].to_numpy(float) if smeta["better"] == "higher" else (1.0 - d["m1"].to_numpy(float))
        scores = {
            "risk": d["risk"].to_numpy(float),
            "scale": d["scale"].to_numpy(float),
            "response": resp,
            "cf_cl": d["benefit"].to_numpy(float),
        }
        res = s50.fast_boot_contrasts(
            g, scores, cl,
            pairs=[("cf_cl", "risk"), ("cf_cl", "scale"), ("cf_cl", "response")],
            q=0.2,
            include_vs_ate=["cf_cl"],
            plugin_scores={"cf_cl": scores["cf_cl"]},
        )
        cf_per_trial[t] = res

    def pool_cf(ts, key):
        es = [cf_per_trial[t]["pairs"][key]["estimate"] for t in ts]
        sds = [cf_per_trial[t]["pairs"][key]["boot_sd"] for t in ts]
        return s50.re_pool(es, sds, floor=mr.V_FLOOR)

    def pool_cf_opt(ts):
        es = [cf_per_trial[t]["optimism"]["cf_cl"]["overstatement"] for t in ts]
        sds = [cf_per_trial[t]["optimism"]["cf_cl"]["boot_sd"] for t in ts]
        p = s50.re_pool(es, sds, floor=mr.V_FLOOR)
        p["mean_plugin"] = float(np.mean([cf_per_trial[t]["optimism"]["cf_cl"]["plugin"] for t in ts]))
        p["mean_heldout"] = float(np.mean([cf_per_trial[t]["optimism"]["cf_cl"]["heldout"] for t in ts]))
        p["n_overstated"] = int(sum(x > 0 for x in es))
        p["n_trials"] = len(ts)
        return p

    cf_summary = {}
    for nm, ts in [
        ("wave12_22", w12),
        ("wave3_6", w3_ts),
        ("all28", all28),
        ("clinical6", ["gusto", "ist", "synbiotic", "elaia2", "w3_ist3", "w3_kat_aki"]),
        ("health_uptake4", ["w3_campos_vaccine", "w3_mtics_crc_fit", "w3_mtics_crc_remind", "w3_alsan_covid"]),
    ]:
        cf_summary[nm] = {
            "cf_over_risk": pool_cf(ts, "cf_cl_minus_risk"),
            "cf_over_scale": pool_cf(ts, "cf_cl_minus_scale"),
            "cf_over_response": pool_cf(ts, "cf_cl_minus_response"),
            "optimism": pool_cf_opt(ts),
        }

    # Build 28-trial pooled circular optimism & 6-trial clinical pool (DR, scale, response, dr_strong)
    def get_pair_any(t, kind):
        if t in w12:
            if kind == "dr_over_risk":
                r = fr["Q3"]["dr"]["per_trial"][t]
                # boot_sd from b2 or scorer
                return r["estimate"], b2["per_trial"][t]["optimism"]["dr"]["boot_sd"]
            elif kind == "scale_over_risk":
                # from scorer
                import glob
                ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"predictions__{t}.json")))
                tt = next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values()))
                c = next(x for x in tt["table"] if x["ranking"] == "scale-minus-risk" and x["budget"] == 0.2)
                return c["estimate"], c["boot_sd"]
            elif kind == "response_over_risk":
                r = fr["Q3"]["response_over_risk_paired"]["per_trial"][t]
                return r["estimate"], (r["hi95"] - r["lo95"]) / 3.92
            elif kind == "dr_over_scale":
                import glob
                ps = sorted(glob.glob(str(ROOT / "scoring" / "*" / f"predictions__{t}.json")))
                tt = next(iter(json.loads(Path(ps[-1]).read_text())["trials"].values()))
                c = next(x for x in tt["table"] if x["ranking"] == "benefit-minus-scale" and x["budget"] == 0.2)
                return c["estimate"], c["boot_sd"]
            elif kind == "dr_strong_over_scale":
                r = b2["per_trial"][t]["pairs"]["dr_strong_minus_scale"]
                return r["estimate"], r["boot_sd"]
            elif kind == "dr_opt":
                r = b2["per_trial"][t]["optimism"]["dr"]
                return r["overstatement"], r["boot_sd"]
            elif kind == "scale_opt":
                r = b2["per_trial"][t]["optimism"]["scale"]
                return r["overstatement"], r["boot_sd"]
            elif kind == "rose_opt":
                r = b2["per_trial"][t]["optimism"]["rose"]
                return r["overstatement"], r["boot_sd"]
        else:
            sc = w3["per_trial"][t]["scored"]
            mp = {
                "dr_over_risk": ("pairs", "benefit_minus_risk"),
                "scale_over_risk": ("pairs", "scale_minus_risk"),
                "response_over_risk": ("pairs", "response_minus_risk"),
                "dr_over_scale": ("pairs", "benefit_minus_scale"),
                "dr_strong_over_scale": ("pairs", "dr_strong_minus_scale"),
                "dr_opt": ("optimism", "benefit"),
                "scale_opt": ("optimism", "scale"),
                "rose_opt": ("optimism", "rose"),
            }[kind]
            r = sc[mp[0]][mp[1]]
            val = r["overstatement"] if mp[0] == "optimism" else r["estimate"]
            return val, r["boot_sd"]

    def pool_any(ts, kind, skip_zero=True):
        es, sds = [], []
        for t in ts:
            e_, s_ = get_pair_any(t, kind)
            if skip_zero and s_ < 1e-9 and abs(e_) < 1e-9:
                continue
            es.append(e_); sds.append(s_)
        p = s50.re_pool(es, sds, floor=mr.V_FLOOR)
        p["n_pos"] = int(sum(x > 0 for x in es))
        return p

    pooled_28 = {}
    for grp, ts in [
        ("all28", all28),
        ("clinical6", ["gusto", "ist", "synbiotic", "elaia2", "w3_ist3", "w3_kat_aki"]),
        ("health_uptake4", ["w3_campos_vaccine", "w3_mtics_crc_fit", "w3_mtics_crc_remind", "w3_alsan_covid"]),
    ]:
        pooled_28[grp] = {
            k: pool_any(ts, k, skip_zero=(k in ("scale_over_risk", "scale_opt")))
            for k in ("scale_over_risk", "response_over_risk", "dr_over_risk", "dr_over_scale", "dr_strong_over_scale", "dr_opt", "rose_opt", "scale_opt")
        }

    # Total randomized & test units across 28 trials
    dt = json.loads((R / "dataset_table.json").read_text())
    n_rand_28 = sum(dt[t]["rows"] for t in w12) + sum(w3["per_trial"][t]["meta"]["n_total"] for t in w3_ts)
    n_test_28 = fr["n_test_units"]["wave1"] + fr["n_test_units"]["wave2"] + sum(w3["per_trial"][t]["meta"]["n_test"] for t in w3_ts)

    out = {
        "n_trials_28": 28,
        "n_individuals_28": int(n_rand_28),
        "n_test_units_28": int(n_test_28),
        "cluster_cf_summary": cf_summary,
        "pooled_28": pooled_28,
        "cluster_cf_per_trial": cf_per_trial,
    }
    (R / "cluster_cf_and_28trial_summary.json").write_text(json.dumps(out, indent=1))
    print("WROTE", R / "cluster_cf_and_28trial_summary.json")
    print(f"28 trials: {n_rand_28:,} randomized, {n_test_28:,} sealed test units")

    print("\n=== CLUSTER-RESPECTING CAUSAL FOREST (per 10,000 reached) ===")
    for nm, v in cf_summary.items():
        r1, r2, r3, opt = v["cf_over_risk"], v["cf_over_scale"], v["cf_over_response"], v["optimism"]
        print(f"  {nm:15s}: vs_risk={10*r1['mean']:+5.0f} ({10*r1['lo95']:+5.0f} to {10*r1['hi95']:+5.0f}) | "
              f"vs_scale={10*r2['mean']:+5.0f} ({10*r2['lo95']:+5.0f} to {10*r2['hi95']:+5.0f}) | "
              f"vs_resp={10*r3['mean']:+5.0f} ({10*r3['lo95']:+5.0f} to {10*r3['hi95']:+5.0f}) | "
              f"opt={10*opt['mean']:+5.0f} [{opt['n_overstated']}/{opt['n_trials']}]")

    print("\n=== 28-TRIAL & 6-CLINICAL POOLS (per 10,000 reached) ===")
    for grp, d_ in pooled_28.items():
        print(f"\n  [{grp}]")
        for k, v in d_.items():
            print(f"    {k:22s} (k={v['k']:2d}, pos={v['n_pos']:2d}/{v['k']:2d}): {10*v['mean']:+5.0f} ({10*v['lo95']:+5.0f} to {10*v['hi95']:+5.0f})")


if __name__ == "__main__":
    main()
