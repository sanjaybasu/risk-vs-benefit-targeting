"""Validate each loader against a published aggregate, assign splits, and seal the test split.

This is the only script that reads full outcomes after splitting, and it does so once, here:
1. Load and validate the common table.
2. Reproduce the published aggregate on the full sample (before the seal).
3. Assign splits by cluster (splits.assign).
4. Write data/processed/<ds>/{context,dev}.parquet and test_state.parquet (no actions, no outcomes).
5. Write test actions, outcomes, propensities, exact truth, and costs to the sealed folder outside
   the repository; record SHA-256 hashes in results/seal_manifest.json.
6. Write results/dataset_table.json: size, treated share, outcome rate (context and dev only),
   license, reproduction, and the smallest detectable difference in policy value on the test split.

Usage: uv run python scripts/03_validate_and_seal.py [dataset ...]
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from system_three.estimators.dr import binarize, exact_scores, ipw_scores
from system_three.loaders import registry
from system_three.splits import PROCESSED, TEST_PUBLIC_COLUMNS, assign
from system_three.table import aux, features, validate

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT.parent / "system-three-sealed"
RESULTS = ROOT / "results"
SEALED_COLS = ["row_id", "action", "outcome", "propensity", "y_ref", "y_act", "truth_exact"]
Q = 0.20  # budget used for the power column


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def reproduce(df: pd.DataFrame, info, pub: dict) -> dict:
    metric, arm = pub["metric"], pub["arm"]
    out = {"source": pub["source"], "metric": metric, "published": pub["value"], "tol": pub["tol"]}
    pair = df[df["action"].isin([arm, info.reference])]
    t = pair["action"] == arm
    if metric == "diff":
        got = pair.loc[t, "outcome"].mean() - pair.loc[~t, "outcome"].mean()
    elif metric == "diff_strata":
        g = pair.assign(t=t).groupby("stratum")
        per = g.apply(lambda s: pd.Series({
            "d": s.loc[s.t, "outcome"].mean() - s.loc[~s.t, "outcome"].mean(), "n": len(s)}))
        per = per.dropna()
        got = float(np.average(per["d"], weights=per["n"]))
    elif metric == "diff_aux_weight":
        w = pair["aux_weight"]
        got = (np.average(pair.loc[t, "outcome"], weights=w[t])
               - np.average(pair.loc[~t, "outcome"], weights=w[~t]))
    elif metric == "treated_share":
        got = t.mean()
    elif metric in ("exact_gap", "ihdp_ate"):
        got = float((df["y_act"] - df["y_ref"]).mean())
    else:
        raise ValueError(metric)
    out["reproduced"] = float(got)
    if pub["value"] is None or pub["tol"] is None:
        out["status"] = "no external target"
    else:
        out["status"] = "match" if abs(got - pub["value"]) <= pub["tol"] else "MISMATCH"
    return out


def unit_scores(df: pd.DataFrame, info) -> np.ndarray:
    """Per-unit scores G for the first non-reference action (IPW or exact), for the power column."""
    act = [a for a in info.actions if a != info.reference][0]
    if df["truth_exact"].all():
        return exact_scores(df["y_act"], df["y_ref"], info.better)
    b = binarize(df, act)
    return ipw_scores(b["outcome"].to_numpy(), b["W"].to_numpy(), b["e"].to_numpy(), info.better)


def run(name: str, info, pub, load) -> dict:
    raw = load()
    attrs = dict(raw.attrs)
    df = validate(raw, info)
    df.attrs.update(attrs)
    rep = reproduce(df, info, pub)
    df["split"] = assign(df).to_numpy()

    out = PROCESSED / name
    out.mkdir(parents=True, exist_ok=True)
    for s in ("context", "dev"):
        part = df[df["split"] == s].drop(columns=["split"])
        part.attrs.clear()
        part.to_parquet(out / f"{s}.parquet", index=False)
    test = df[df["split"] == "test"]
    test[TEST_PUBLIC_COLUMNS + features(test)].to_parquet(out / "test_state.parquet", index=False)
    if attrs:
        (out / "attrs.json").write_text(json.dumps(attrs, default=str))

    SEALED.mkdir(parents=True, exist_ok=True)
    sealed_path = SEALED / f"{name}_test.parquet"
    test[SEALED_COLS + aux(test)].to_parquet(sealed_path, index=False)

    dev_ctx = df[df["split"] != "test"]
    dev_ctx.attrs.update(attrs)
    g = unit_scores(dev_ctx, info)
    n_test_units = int((df["split"] == "test").sum())
    # Detectable difference (80% power, two-sided 0.05) between two rankings' mean effect among
    # the top 20%, treating the two selected sets as independent (conservative; AIPW and overlap
    # between the sets both shrink it). Estimated from context and dev IPW scores only.
    se_diff = np.sqrt(2) * np.std(g) / np.sqrt(max(1, Q * n_test_units))
    mde = 2.8 * se_diff
    binary = set(np.unique(dev_ctx["outcome"].dropna())) <= {0.0, 1.0}
    t = df["action"] != info.reference
    return {
        "dataset": name, "tier": info.tier, "outcome": info.outcome_name, "better": info.better,
        "rows": int(len(df)), "clusters": int(df["cluster_id"].nunique()),
        "context": int((df["split"] == "context").sum()), "dev": int((df["split"] == "dev").sum()),
        "test": n_test_units, "treated_share": float(t.mean()),
        "outcome_mean_context_dev": float(dev_ctx["outcome"].mean()),
        "truth_exact": bool(df["truth_exact"].all()),
        "license": info.license, "commercial_use": info.commercial_use,
        "reproduction": rep,
        "binary_outcome": bool(binary),
        "mde_per_action_at_q20": float(mde),
        "mde_per_1000_at_q20": float(1000 * mde) if binary else None,
        "mde_in_outcome_sd": float(mde / dev_ctx["outcome"].std()),
        "sealed_sha256": sha256(sealed_path),
    }


def main(argv: list[str]) -> int:
    reg = registry()
    names = argv or list(reg)
    RESULTS.mkdir(exist_ok=True)
    table_path = RESULTS / "dataset_table.json"
    table = json.loads(table_path.read_text()) if table_path.exists() else {}
    for name in names:
        info, pub, load = reg[name]
        try:
            row = run(name, info, pub, load)
            table[name] = row
            r = row["reproduction"]
            print(f"{name:32s} rows={row['rows']:>10,} test={row['test']:>9,} "
                  f"repro={r['reproduced']:.4f} vs {r['published']} [{r['status']}] "
                  f"MDE={row['mde_per_action_at_q20']:.4g}/action ({row['mde_in_outcome_sd']:.2f} SD)",
                  flush=True)
        except Exception as e:
            print(f"{name:32s} FAIL {type(e).__name__}: {e}", flush=True)
            table[name] = {"dataset": name, "error": f"{type(e).__name__}: {e}"}
    table_path.write_text(json.dumps(table, indent=2))
    manifest = {k: v.get("sealed_sha256") for k, v in table.items() if "sealed_sha256" in v}
    (RESULTS / "seal_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all("error" not in v for v in table.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
