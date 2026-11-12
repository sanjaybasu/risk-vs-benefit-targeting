"""System Three contract, version 0.1.

A System Three model answers "what changes if we act?". A request carries a state, a question
(a menu of actions that includes a reference action, a named outcome, and the better direction),
and context (logged records of state, action, outcome, or a registered dataset id). The response
gives, per non-reference action, the expected change against the reference, a 90% interval,
the probability of harm, and a support score, plus an abstain flag. It never returns free text.

v0.2 (panel C37, C38, C67): the effect, interval, and p_harm refer to the conditional average
effect at this state, CATE(x); individual-level effects are not identified from randomized data
without further assumptions, so unit-level coverage is claimed only where both potential outcomes
are observed. Optional Frechet-Hoeffding bounds give the identified range of the probability that
an individual is harmed. The "wait" question type is removed: no benchmark dataset identifies
delay effects.

`contract/openapi.yaml` is generated from these models by `build_openapi()`; a test fails if the
committed file differs from the generated one.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_VERSION = "0.2.0"

State = dict[str, Any] | str


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Outcome(_Strict):
    name: str = Field(min_length=1, description="Name of the outcome, e.g. 'readmit_180d'.")
    better: Literal["higher", "lower"] = Field(description="Which direction of the outcome is better.")


class Question(_Strict):
    type: Literal["effect", "choose"]
    actions: list[str] = Field(min_length=2, description="Menu of actions; includes the reference.")
    reference: str = Field(description="Reference action, such as 'none' or 'call_small_model'.")
    outcome: Outcome
    horizon_days: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _check(self) -> Question:
        if len(set(self.actions)) != len(self.actions):
            raise ValueError("actions must be unique")
        if self.reference not in self.actions:
            raise ValueError("reference must be one of actions")
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
    mean: float = Field(description="CATE(x): expected outcome change against the reference "
                                    "for cases with this state.")
    interval_90: tuple[float, float] = Field(description="90% interval for CATE(x), not for an "
                                                         "individual's effect.")
    p_harm: Annotated[float, Field(ge=0, le=1)] = Field(
        description="P(CATE(x) is in the worse direction | data). Not the probability that an "
                    "individual is harmed, which is not identified; see p_harm_individual_bounds."
    )
    p_harm_individual_bounds: tuple[float, float] | None = Field(
        default=None, description="Frechet-Hoeffding bounds on P(an individual with this state is "
                                  "harmed), for binary outcomes; optional."
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
