from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from .agreement import AgreementFn, elementwise, exact_match
from .space import InstanceSpace

Predictor = Callable[[object], Sequence]


@dataclass(frozen=True)
class AlignmentResult:
    value: float
    instances: int
    reference: str

    def __float__(self) -> float:
        return self.value


def _predict(model: Predictor, X) -> Sequence:
    if hasattr(model, "predict"):
        return model.predict(X)
    return model(X)


def alignment(model_a: Predictor, model_b: Predictor, X, agree: AgreementFn = exact_match) -> float:
    matches = elementwise(_predict(model_a, X), _predict(model_b, X), agree)
    return float(matches.mean()) if matches.size else 1.0


def distributional_alignment(
    model_a: Predictor, model_b: Predictor, X, agree: AgreementFn = exact_match
) -> AlignmentResult:
    value = alignment(model_a, model_b, X, agree)
    return AlignmentResult(value, len(np.asarray(X)), "P(X)")


def geometric_alignment(
    model_a: Predictor,
    model_b: Predictor,
    space: InstanceSpace,
    count: int = 1000,
    seed: int = 42,
    agree: AgreementFn = exact_match,
) -> AlignmentResult:
    samples = space.sample(count, seed)
    return AlignmentResult(alignment(model_a, model_b, samples, agree), count, "U(F)")


dra = distributional_alignment
gra = geometric_alignment
