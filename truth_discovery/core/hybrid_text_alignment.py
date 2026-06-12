
from __future__ import annotations

import re
from typing import Iterable, Optional

import numpy as np

DEFAULT_NUMERIC_MATCH_TOL = 0.02
DEFAULT_LASO_NUMERIC_ALPHA = 0.5

def extract_numeric_values(text: str) -> list[float]:

    values: list[float] = []
    for match in re.finditer(r"-?\d+(?:\.\d+)?", str(text)):
        try:
            values.append(float(match.group()))
        except (TypeError, ValueError):
            continue
    return values

def numeric_match_ratio(
    reference: Iterable[float],
    candidate: Iterable[float],
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:

    reference_values = [float(value) for value in reference]
    candidate_values = [float(value) for value in candidate]
    if not reference_values:
        return 1.0

    matched = 0
    for ref_value in reference_values:
        denom = max(abs(ref_value), 1e-8)
        if any(abs(ref_value - cand_value) / denom <= tol for cand_value in candidate_values):
            matched += 1
    return matched / len(reference_values)

def symmetric_numeric_match_ratio(
    left: Iterable[float],
    right: Iterable[float],
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:

    return 0.5 * (
        numeric_match_ratio(left, right, tol=tol)
        + numeric_match_ratio(right, left, tol=tol)
    )

def gaussian_numeric_similarity(
    left: Iterable[float],
    right: Iterable[float],
    tol: float = DEFAULT_NUMERIC_MATCH_TOL,
) -> float:

    left_vals = [float(v) for v in left]
    right_vals = [float(v) for v in right]

    def _directional(ref: list[float], cand: list[float]) -> float:
        if not ref:
            return 1.0
        if not cand:
            return 0.0
        scores: list[float] = []
        for r in ref:
            denom = max(abs(r), 1e-8)
            best_rel = min(abs(r - c) / denom for c in cand)
            scores.append(float(np.exp(-0.5 * (best_rel / max(tol, 1e-12)) ** 2)))
        return float(np.mean(scores))

    return 0.5 * (_directional(left_vals, right_vals) + _directional(right_vals, left_vals))

def blend_LASO_numeric_scores(
    sem_score: float,
    num_score: Optional[float],
    alpha: float = DEFAULT_LASO_NUMERIC_ALPHA,
) -> float:

    sem = float(np.clip(sem_score, 0.0, 1.0))
    if num_score is None:
        return sem

    alpha = float(np.clip(alpha, 0.0, 1.0))
    num = float(np.clip(num_score, 0.0, 1.0))
    return float(alpha * sem + (1.0 - alpha) * num)
