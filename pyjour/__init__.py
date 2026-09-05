"""Python bindings for conservative greeting-name inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import version
from typing import Any, cast

from . import _native

__all__ = ["Inference", "__version__", "infer", "infer_detailed"]

__version__ = version("pyjour")


@dataclass(frozen=True, slots=True)
class Inference:
    """Production C5 inference over one display name."""

    best_candidate: str | None
    greeting_name: str | None
    decision_score: float
    emission_source: str
    gender_hint: str | None
    gender_confidence: float


def infer(
    display_name: str,
    country_hint: str | None = None,
    locale_hint: str | None = None,
    gender_hint: str | None = None,
) -> Inference:
    """Infer a conservative greeting name using frozen production C5."""

    summary = cast(
        dict[str, Any],
        json.loads(
            _native.infer_json(display_name, country_hint, locale_hint, gender_hint)
        ),
    )
    return Inference(
        best_candidate=_optional_string(summary["best_candidate"]),
        greeting_name=_optional_string(summary["greeting_name"]),
        decision_score=float(summary["decision_score"]),
        emission_source=cast(str, summary["emission_source"]),
        gender_hint=_optional_string(summary["gender_hint"]),
        gender_confidence=float(summary["gender_confidence"]),
    )


def infer_detailed(
    display_name: str,
    country_hint: str | None = None,
    locale_hint: str | None = None,
    gender_hint: str | None = None,
) -> dict[str, Any]:
    """Return the detailed Rust diagnostic as JSON-compatible values."""

    return cast(
        dict[str, Any],
        json.loads(
            _native.infer_detailed_json(
                display_name, country_hint, locale_hint, gender_hint
            )
        ),
    )


def _optional_string(value: object) -> str | None:
    assert value is None or isinstance(value, str)
    return value
