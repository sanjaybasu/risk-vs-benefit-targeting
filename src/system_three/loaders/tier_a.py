"""Tier A loaders: agent decisions where both potential outcomes are observed.

RouterBench: every one of 11 models answered every prompt, with a score and a cost. For a pair
(smaller model = reference, larger model = action), y_ref and y_act are the two scores, so each
prompt's effect of escalating is known exactly. A logged action is simulated by a fair coin per
prompt (seeded hash), so context records look like logged data with propensity 0.5.

SWE-bench Verified: resolved or not for each task, for each submitted system. Same construction
for a weaker system (reference) and a stronger one (action). No cost per task is published.
"""

import hashlib
import json

import numpy as np
import pandas as pd

from system_three.loaders._base import RAW, frame
from system_three.table import DatasetInfo

RB_PAIRS = {
    "mistral7b_to_gpt4": ("mistralai/mistral-7b-chat", "gpt-4-1106-preview"),
    "gpt35_to_gpt4": ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
    "llama70b_to_claude2": ("meta/llama-2-70b-chat", "claude-v2"),
    "mixtral_to_gpt4": ("mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"),
}


def coin(key: str) -> int:
    return hashlib.sha256(f"logged-action|{key}".encode()).digest()[0] & 1


def rb_info(pair: str) -> DatasetInfo:
    return DatasetInfo(
        name=f"routerbench__{pair}", tier="A", outcome_name="answer_score", better="higher",
        reference="call_small", actions=("call_small", "call_large"),
        license="MIT (code); data card states no license", commercial_use=False,
        citation="Hu QJ, Bieker J, Li X, et al. RouterBench: a benchmark for multi-LLM routing "
        "system. arXiv:2403.12031 (2024).",
    )


RB_PUBLISHED = {"source": "exact truth; check = mean(y_act - y_ref) equals model accuracy gap",
                "arm": "call_large", "metric": "exact_gap", "value": None, "tol": 1e-9}


def load_routerbench(pair: str, shots: str = "0shot") -> pd.DataFrame:
    small, large = RB_PAIRS[pair]
    d = pd.read_parquet(RAW / "routerbench" / f"routerbench_{shots}.parquet")
    for m in (small, large):
        d[m] = pd.to_numeric(d[m], errors="coerce")
    d = d[d[small].notna() & d[large].notna()].reset_index(drop=True)
    w = d["sample_id"].map(coin)
    y = np.where(w == 1, d[large], d[small])
    text = pd.Series([json.dumps({"eval": e, "prompt": p}) for e, p in zip(d["eval_name"], d["prompt"])])
    info = rb_info(pair)
    df = frame(info.name, pd.DataFrame(index=d.index), w.map({1: "call_large", 0: "call_small"}),
               pd.Series(y), info, propensity=0.5, row_id=d["sample_id"], cluster=d["sample_id"],
               stratum=d["eval_name"], state_text=text, y_ref=d[small], y_act=d[large])
    df["aux_cost_ref"] = pd.to_numeric(d[f"{small}|total_cost"], errors="coerce").to_numpy()
    df["aux_cost_act"] = pd.to_numeric(d[f"{large}|total_cost"], errors="coerce").to_numpy()
    return df


# ---------------------------------------------------------------- SWE-bench Verified
SWE_INFO_BASE = dict(tier="A", outcome_name="resolved", better="higher", reference="weaker_agent",
                     actions=("weaker_agent", "stronger_agent"), license="no license stated",
                     commercial_use=False,
                     citation="Jimenez CE, Yang J, Wettig A, et al. SWE-bench: can language models "
                     "resolve real-world GitHub issues? ICLR 2024. Experiments repository.")


def swebench_systems() -> pd.DataFrame:
    root = RAW / "swebench" / "repo" / "evaluation" / "verified"
    rows = []
    for res in sorted(root.glob("*/results/results.json")):
        r = json.loads(res.read_text())
        rows.append({"system": res.parents[1].name, "resolved": set(r.get("resolved", [])),
                     "n_resolved": len(r.get("resolved", []))})
    return pd.DataFrame(rows).sort_values("n_resolved").reset_index(drop=True)


def swebench_pairs() -> dict[str, tuple[str, str]]:
    """Three fixed pairs by resolve-rate rank: 25th to 75th, 50th to 90th, 10th to 50th centile."""
    s = swebench_systems()
    q = lambda p: s["system"].iat[int(round(p * (len(s) - 1)))]  # noqa: E731
    return {"p25_to_p75": (q(0.25), q(0.75)), "p50_to_p90": (q(0.50), q(0.90)),
            "p10_to_p50": (q(0.10), q(0.50))}


def load_swebench(pair: str) -> pd.DataFrame:
    s = swebench_systems().set_index("system")
    weak, strong = swebench_pairs()[pair]
    tasks = sorted(set().union(*s["resolved"]))  # tasks any system resolved
    # Full task list: every id that appears in any results file field, so unresolved tasks count.
    root = RAW / "swebench" / "repo" / "evaluation" / "verified"
    allids = set(tasks)
    for res in root.glob("*/results/results.json"):
        for v in json.loads(res.read_text()).values():
            if isinstance(v, list):
                allids |= {t for t in v if isinstance(t, str) and "__" in t}
    ids = pd.Series(sorted(allids))
    y_ref = ids.isin(s.at[weak, "resolved"]).astype(float)
    y_act = ids.isin(s.at[strong, "resolved"]).astype(float)
    w = ids.map(coin)
    info = DatasetInfo(name=f"swebench__{pair}", **SWE_INFO_BASE)
    text = ids.map(lambda t: json.dumps({"instance_id": t, "repo": t.split("__")[0]}))
    df = frame(info.name, pd.DataFrame(index=ids.index), w.map({1: "stronger_agent", 0: "weaker_agent"}),
               pd.Series(np.where(w == 1, y_act, y_ref)), info, propensity=0.5, row_id=ids,
               stratum=ids.str.split("__").str[0], state_text=text, y_ref=y_ref, y_act=y_act)
    df.attrs["systems"] = {"reference": weak, "action": strong}
    return df
