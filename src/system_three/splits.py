"""Split assignment and the test-access guard.

Splits are assigned per cluster by a salted SHA-256 hash, so a household, a prompt, or a
headline test never spans two splits. Default shares: 50% context, 20% dev, 30% test.
A dataset with a published native split (IHDP, Upworthy) keeps it; see NATIVE_SPLIT.

Processed tables:
  data/processed/<dataset>/context.parquet   full rows
  data/processed/<dataset>/dev.parquet       full rows
  data/processed/<dataset>/test_state.parquet  row_id, cluster_id, stratum, state only

Test actions and outcomes exist only in the sealed folder, which sits outside the repository
and is written once by scripts/03_validate_and_seal.py. No module under src/ reads it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
SALT = "system-three-v0.1-2026-11-12"
SHARES = {"context": 0.50, "dev": 0.20, "test": 0.30}
TEST_PUBLIC_COLUMNS = ["dataset", "row_id", "cluster_id", "stratum", "state"]
NATIVE_SPLIT = {"ihdp", "upworthy"}


def hash_unit(dataset: str, cluster_id: str) -> float:
    h = hashlib.sha256(f"{SALT}|{dataset}|{cluster_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def family(dataset: str) -> str:
    """Datasets that share units (RouterBench pairs share prompts) share one split key."""
    return dataset.split("__")[0]


def assign(df: pd.DataFrame) -> pd.Series:
    """Return the split for each row, by cluster. Native splits are kept where published."""
    fam = family(df["dataset"].iat[0])
    u = df["cluster_id"].map(lambda c: hash_unit(fam, c))
    cuts = np.cumsum(list(SHARES.values()))
    split = pd.Series(
        np.select([u < cuts[0], u < cuts[1]], ["context", "dev"], "test"), index=df.index
    )
    if "native_split" in df.columns:  # IHDP: published train/test; train divides 5:2
        train = df["native_split"] == "train"
        split[~train] = "test"
        split[train] = np.where(u[train] < 5 / 7, "context", "dev")
    return split


class SealedSplitError(PermissionError):
    pass


def load(dataset: str, split: str) -> pd.DataFrame:
    """Load a processed split. Test returns state only; its outcomes are sealed."""
    if split == "test":
        return pd.read_parquet(PROCESSED / dataset / "test_state.parquet")
    if split not in ("context", "dev"):
        raise ValueError(f"unknown split {split!r}")
    return pd.read_parquet(PROCESSED / dataset / f"{split}.parquet")


def load_test_outcomes(*_args, **_kwargs):
    raise SealedSplitError(
        "Test outcomes are sealed and held by a third party. Score the test split with the "
        "third party's runner; never read it during development."
    )
