"""Guard against text covariates silently coercing to all-missing (pandas 3 str dtype)."""

import pytest

from system_three.loaders import registry
from system_three.table import validate


@pytest.mark.parametrize("name", ["ist", "synbiotic", "gusto", "elaia1", "elaia2"])
def test_no_all_missing_covariates(name):
    info, _, load = registry()[name]
    df = validate(load(), info)
    dead = [c for c in df.columns if c.startswith("x_") and df[c].isna().all()]
    assert not dead, dead
