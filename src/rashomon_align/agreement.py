from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

AgreementFn = Callable[[object, object], float]


def exact_match(a: object, b: object) -> float:
    return float(a == b)


def elementwise(predictions_a: Sequence, predictions_b: Sequence, agree: AgreementFn = exact_match) -> np.ndarray:
    left, right = list(predictions_a), list(predictions_b)
    if len(left) != len(right):
        raise ValueError(f"prediction counts differ: {len(left)} and {len(right)}")
    if not left:
        return np.empty(0, dtype=float)
    return np.asarray([agree(a, b) for a, b in zip(left, right, strict=True)], dtype=float)
