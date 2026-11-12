"""Test outcomes are sealed: no development code path can read them."""

import re
from pathlib import Path

import pytest

from system_three import splits

ROOT = Path(__file__).resolve().parents[1]
SEAL_WRITER = "03_validate_and_seal.py"


def test_load_test_outcomes_raises():
    with pytest.raises(splits.SealedSplitError):
        splits.load_test_outcomes("camden")


def test_only_the_seal_writer_names_the_sealed_folder():
    pat = re.compile(r"system-three-sealed")
    offenders = []
    for p in list((ROOT / "src").rglob("*.py")) + list((ROOT / "scripts").rglob("*.py")):
        if p.name != SEAL_WRITER and pat.search(p.read_text()):
            offenders.append(str(p.relative_to(ROOT)))
    assert not offenders, offenders


def test_split_hash_is_stable_and_cluster_level():
    a = splits.hash_unit("camden", "25264")
    assert a == splits.hash_unit("camden", "25264")
    assert 0 <= a < 1
