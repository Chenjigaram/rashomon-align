from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier


class MaximallyPrunedTree(ClassifierMixin, BaseEstimator):
    def __init__(self, min_samples_split: int = 10, random_state: int = 42, allow_stump: bool = False):
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.allow_stump = allow_stump

    def _tree(self, alpha: float) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            min_samples_split=self.min_samples_split,
            ccp_alpha=alpha,
            random_state=self.random_state,
        )

    def fit(self, X, y):
        probe = DecisionTreeClassifier(min_samples_split=self.min_samples_split, random_state=self.random_state)
        alphas = np.asarray([a for a in probe.cost_complexity_pruning_path(X, y).ccp_alphas if a >= 0.0])
        chosen = float(alphas[-1]) if alphas.size else 0.0
        fitted = self._tree(chosen).fit(X, y)
        if not self.allow_stump and fitted.tree_.node_count <= 1 and alphas.size > 1:
            chosen = float(alphas[-2])
            fitted = self._tree(chosen).fit(X, y)
        self.alpha_ = chosen
        self.tree_ = fitted
        self.classes_ = fitted.classes_
        return self

    def predict(self, X):
        return self.tree_.predict(X)
