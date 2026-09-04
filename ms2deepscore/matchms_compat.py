"""Compatibility helpers for matchms pre-1.0 and >=1.0 APIs.

The matchms 1.0 refactor renamed the BaseSimilarity module and changed the
``matrix`` contract from returning a NumPy array to returning ``matchms.Scores``.
Keep all version-dependent imports/return wrapping in one place so that the
actual similarity implementations stay readable.
"""

from __future__ import annotations
from typing import Iterable, Tuple

try:  # matchms >= 1.0
    from matchms import Scores
    from matchms.similarity.base_similarity import BaseSimilarity

    MATCHMS_V1_API = True
except ImportError:  # matchms <= 0.33.x
    from matchms.similarity.BaseSimilarity import BaseSimilarity

    Scores = None  # type: ignore[assignment]
    MATCHMS_V1_API = False


def normalize_score_fields(
    score_fields: Iterable[str] | str | None,
    available_fields: Tuple[str, ...],
) -> Tuple[str, ...]:
    """Validate/normalize requested score fields for the matchms >=1.0 API."""
    if score_fields is None:
        return available_fields
    if isinstance(score_fields, str):
        fields = (score_fields,)
    else:
        fields = tuple(score_fields)

    unknown = tuple(field for field in fields if field not in available_fields)
    if unknown:
        raise ValueError(
            f"Unknown score field(s): {unknown}. Available fields are {available_fields}."
        )
    if len(fields) == 0:
        raise ValueError("score_fields must contain at least one field.")
    return fields


def as_matchms_scores(score_arrays: dict[str, object]):
    """Wrap score matrices in the matchms >=1.0 Scores container."""
    if not MATCHMS_V1_API or Scores is None:
        raise RuntimeError("matchms.Scores wrapping is only available with the matchms >=1.0 API.")
    return Scores(score_arrays)


def assert_legacy_symmetric_inputs(references, queries) -> None:
    """Preserve the <=0.33 ``is_symmetric=True`` validation behavior."""
    if references is queries:
        return
    try:
        equal = len(references) == len(queries) and all(
            reference == query for reference, query in zip(references, queries)
        )
    except Exception:
        equal = False
    assert equal, "Expected references to be equal to queries for is_symmetric=True"
