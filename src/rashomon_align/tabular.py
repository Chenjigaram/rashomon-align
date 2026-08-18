from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .measure import distributional_alignment, geometric_alignment
from .space import InstanceSpace


@dataclass(frozen=True)
class FoldResult:
    accuracy_a: float
    accuracy_b: float
    dra: float
    gra: float

    @property
    def accuracy_difference(self) -> float:
        return self.accuracy_b - self.accuracy_a

    @property
    def absolute_accuracy_difference(self) -> float:
        return abs(self.accuracy_a - self.accuracy_b)


def compare_on_fold(
    model_a, model_b, X_train, y_train, X_test, y_test, count: int = 1000, seed: int = 42
) -> FoldResult:
    model_a.fit(X_train, y_train)
    model_b.fit(X_train, y_train)
    space = InstanceSpace.from_data(X_train)
    return FoldResult(
        accuracy_a=float((model_a.predict(X_test) == np.asarray(y_test)).mean()),
        accuracy_b=float((model_b.predict(X_test) == np.asarray(y_test)).mean()),
        dra=distributional_alignment(model_a, model_b, X_test).value,
        gra=geometric_alignment(model_a, model_b, space, count=count, seed=seed).value,
    )


def cross_validated_comparison(make_a, make_b, X, y, folds: int = 5, seed: int = 42, count: int = 1000):
    from sklearn.model_selection import StratifiedKFold

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = []
    array_X, array_y = np.asarray(X, dtype=float), np.asarray(y)
    for train_index, test_index in splitter.split(array_X, array_y):
        results.append(
            compare_on_fold(
                make_a(),
                make_b(),
                array_X[train_index],
                array_y[train_index],
                array_X[test_index],
                array_y[test_index],
                count=count,
                seed=seed,
            )
        )
    return results
