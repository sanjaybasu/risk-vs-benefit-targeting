"""System Three contract, version 0.1.

A System Three model answers "what changes if we act?". A request carries a state, a question
(a menu of actions that includes a reference action, a named outcome, and the better direction),
and context (logged records of state, action, outcome, or a registered dataset id). The response
gives, per non-reference action, the expected change against the reference, a 90% interval,
the probability of harm, and a support score, plus an abstain flag. It never returns free text.

`contract/openapi.yaml` is generated from these models by `build_openapi()`; a test fails if the
committed file differs from the generated one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "0.1.0"

State = dict[str, Any] | str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Outcome(_Strict):
    name: str = Field(min_length=1, description="Name of the outcome, e.g. 'readmit_180d'.")
    better: Literal["higher", "lower"] = Field(description="Which direction of the outcome is better.")


class Question(_Strict):
    type: Literal["effect", "choose", "wait"]
    actions: list[str] = Field(min_length=2, description="Menu of actions; includes the reference.")
    reference: str = Field(description="Reference action, such as 'none' or 'call_small_model'.")
    outcome: Outcome
    horizon_days: Annotated[int, Field(gt=0)] | None = None
    delays_days: list[Annotated[int, Field(ge=0)]] | None = Field(
        default=None, description="Required for type 'wait': candidate delays, 0 means act now."
    )

    @model_validator(mode="after")
    def _check(self) -> Question:
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if self.reference not in self.actions:
            raise ValueError("reference must be one of actions")
        if self.type == "wait" and not self.delays_days:
            raise ValueError("type 'wait' needs delays_days")
        return self


class Record(_Strict):
    state: State
    action: str
    outcome: float
    propensity: Annotated[float, Field(gt=0, lt=1)] | None = Field(
        default=None, description="P(action | state) under the logging policy, if known."
    )


class Context(_Strict):
    dataset_id: str | None = None
    records: list[Record] | None = None

    @model_validator(mode="after")
    def _one(self) -> Context:
        if (self.dataset_id is None) == (self.records is None):
            raise ValueError("give exactly one of dataset_id or records")
        return self


class Request(_Strict):
    model: str = "rose-latest"
    state: State
    question: Question
    context: Context


class Effect(_Strict):
    action: str
    delay_days: int | None = Field(default=None, description="Set for type 'wait'.")
    mean: float = Field(description="Expected outcome change against the reference action.")
    interval_90: tuple[float, float]
    p_harm: Annotated[float, Field(ge=0, le=1)] = Field(
        description="Probability the change is in the worse direction for this case."
    )
    p_best: Annotated[float, Field(ge=0, le=1)] | None = Field(
        default=None, description="Set for type 'choose': probability this action is best."
    )
    support: Annotated[float, Field(ge=0, le=1)] = Field(
        description="How well the context covers this state and action (overlap)."
    )

    @model_validator(mode="after")
    def _interval(self) -> Effect:
        lo, hi = self.interval_90
        if lo > hi:
            raise ValueError("interval_90 must be (low, high) with low <= high")
        return self


class Response(_Strict):
    model: str
    contract_version: str = CONTRACT_VERSION
    reference: str
    effects: list[Effect]
    abstain: bool
    abstain_reason: Literal["low_support", "no_context", "outcome_unknown"] | None = None


def build_openapi() -> dict[str, Any]:
    """Return the OpenAPI 3.1 document for the contract."""
    defs: dict[str, Any] = {}
    for m in (Request, Response):
        schema = m.model_json_schema(ref_template="#/components/schemas/{model}")
        defs.update(schema.pop("$defs", {}))
        defs[m.__name__] = schema
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "System Three",
            "version": CONTRACT_VERSION,
            "summary": "Effect of each action against a reference action, with uncertainty.",
            "license": {"name": "MIT", "identifier": "MIT"},
        },
        "paths": {
            "/v1/systemthree": {
                "post": {
                    "operationId": "estimate",
                    "requestBody": {
                        "required": True,
                        "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Request"}}},
                    },
                    "responses": {
                        "200": {
                            "description": "Per-action effects against the reference.",
                            "content": {
                                "application/json": {"schema": {"$ref": "#/components/schemas/Response"}}
                            },
                        }
                    },
                }
            }
        },
        "components": {"schemas": dict(sorted(defs.items()))},
    }
