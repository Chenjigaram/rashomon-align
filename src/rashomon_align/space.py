from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InstanceSpace:
    lower: np.ndarray
    upper: np.ndarray
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if self.lower.ndim != 1:
            raise ValueError("bounds must be one-dimensional")
        if np.any(self.upper < self.lower):
            raise ValueError("upper bound is below lower bound for at least one attribute")

    @property
    def dimensions(self) -> int:
        return int(self.lower.shape[0])

    @classmethod
    def from_data(cls, X, names: tuple[str, ...] | None = None) -> InstanceSpace:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("expected a two-dimensional feature matrix")
        if names is None and hasattr(X, "columns"):
            names = tuple(str(c) for c in X.columns)
        return cls(array.min(axis=0), array.max(axis=0), names)

    def sample(self, count: int, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform(self.lower, self.upper, size=(count, self.dimensions))

    def contains(self, X) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        return np.all((array >= self.lower) & (array <= self.upper), axis=1)

    def volume_is_degenerate(self) -> bool:
        return bool(np.any(self.upper == self.lower))
