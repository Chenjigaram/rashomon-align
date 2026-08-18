import numpy as np
import pytest

pytest.importorskip("sklearn")

from sklearn.tree import DecisionTreeClassifier  # noqa: E402

from rashomon_align.tabular import compare_on_fold, cross_validated_comparison  # noqa: E402


def data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1, size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    return X, y


def unpruned():
    return DecisionTreeClassifier(min_samples_split=2, random_state=42)


def pruned():
    return DecisionTreeClassifier(min_samples_split=10, ccp_alpha=0.01, random_state=42)


def test_fold_result_has_both_measures():
    X, y = data()
    result = compare_on_fold(unpruned(), pruned(), X[:150], y[:150], X[150:], y[150:], count=200)
    assert 0.0 <= result.dra <= 1.0
    assert 0.0 <= result.gra <= 1.0


def test_accuracy_difference_is_absolute():
    X, y = data()
    result = compare_on_fold(unpruned(), pruned(), X[:150], y[:150], X[150:], y[150:], count=100)
    assert result.accuracy_difference >= 0


def test_cross_validation_returns_one_result_per_fold():
    X, y = data()
    assert len(cross_validated_comparison(unpruned, pruned, X, y, folds=5, count=100)) == 5


def test_identical_models_align_perfectly():
    X, y = data()
    results = cross_validated_comparison(unpruned, unpruned, X, y, folds=3, count=200)
    assert all(r.dra == 1.0 and r.gra == 1.0 for r in results)


def test_comparison_is_reproducible():
    X, y = data()
    first = cross_validated_comparison(unpruned, pruned, X, y, folds=3, count=200)
    second = cross_validated_comparison(unpruned, pruned, X, y, folds=3, count=200)
    assert [r.gra for r in first] == [r.gra for r in second]
