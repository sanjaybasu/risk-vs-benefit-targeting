import importlib.util
from pathlib import Path

import pytest
from pydantic import ValidationError

from system_three.contract import Request, Response

ROOT = Path(__file__).resolve().parents[1]

# Example from page 6 of the release plan (illustrative values).
PLAN_REQUEST = {
    "state": {"task": "refactor billing module", "tests_failing": 2, "turns_used": 14},
    "question": {
        "type": "effect",
        "actions": ["escalate_model", "ask_user", "none"],
        "reference": "none",
        "outcome": {"name": "task_success", "better": "higher"},
    },
    "context": {"dataset_id": "agent-runs-2026"},
}
PLAN_RESPONSE = {
    "model": "rose-0.1",
    "reference": "none",
    "effects": [
        {"action": "escalate_model", "mean": 0.18, "interval_90": [0.09, 0.27], "p_harm": 0.02, "support": 0.88},
        {"action": "ask_user", "mean": 0.03, "interval_90": [-0.05, 0.11], "p_harm": 0.27, "support": 0.91},
    ],
    "abstain": False,
}


def test_plan_example_validates():
    Request.model_validate(PLAN_REQUEST)
    Response.model_validate(PLAN_RESPONSE)


@pytest.mark.parametrize(
    "patch",
    [
        {"question": {**PLAN_REQUEST["question"], "reference": "missing"}},
        {"question": {**PLAN_REQUEST["question"], "actions": ["none"]}},
        {"question": {**PLAN_REQUEST["question"], "type": "wait"}},  # removed in v0.2
        {"context": {}},
        {"context": {"dataset_id": "x", "records": []}},
        {"extra_field": 1},
    ],
)
def test_bad_requests_fail(patch):
    with pytest.raises(ValidationError):
        Request.model_validate({**PLAN_REQUEST, **patch})


def test_bad_interval_fails():
    bad = {**PLAN_RESPONSE, "effects": [{**PLAN_RESPONSE["effects"][0], "interval_90": [0.3, 0.1]}]}
    with pytest.raises(ValidationError):
        Response.model_validate(bad)


def test_committed_openapi_matches_models():
    spec = importlib.util.spec_from_file_location("b", ROOT / "scripts" / "00_build_openapi.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert (ROOT / "contract" / "openapi.yaml").read_text() == mod.render()
