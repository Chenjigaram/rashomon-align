import numpy as np
import pytest

from rashomon_align import InstanceSpace, alignment, dra, exact_match, gra

X = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.9, 0.1]])


class Threshold:
    def __init__(self, index, cut):
        self.index, self.cut = index, cut

    def predict(self, X):
        return (np.asarray(X)[:, self.index] > self.cut).astype(int)


def test_model_is_perfectly_aligned_with_itself():
    model = Threshold(0, 0.5)
    assert alignment(model, model, X) == 1.0


def test_opposite_models_never_agree():
    class Inverted(Threshold):
        def predict(self, X):
            return 1 - super().predict(X)

    assert alignment(Threshold(0, 0.5), Inverted(0, 0.5), X) == 0.0


def test_plain_callables_are_accepted():
    assert alignment(lambda X: [0] * len(X), lambda X: [0] * len(X), X) == 1.0


def test_distributional_alignment_reports_its_reference():
    result = dra(Threshold(0, 0.5), Threshold(0, 0.5), X)
    assert result.reference == "P(X)" and result.instances == 4


def test_geometric_alignment_reports_its_reference():
    space = InstanceSpace.from_data(X)
    result = gra(Threshold(0, 0.5), Threshold(0, 0.5), space, count=100)
    assert result.reference == "U(F)" and result.instances == 100


def test_alignment_result_converts_to_float():
    assert float(dra(Threshold(0, 0.5), Threshold(0, 0.5), X)) == 1.0


def test_high_distributional_but_low_geometric_alignment():
    space = InstanceSpace(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    a, b = Threshold(0, 0.5), Threshold(1, 0.5)
    data = np.array([[0.1, 0.1], [0.9, 0.9], [0.2, 0.2], [0.8, 0.8]])
    assert dra(a, b, data).value == 1.0
    assert gra(a, b, space, count=2000).value < 0.6


def test_mismatched_prediction_counts_are_rejected():
    with pytest.raises(ValueError):
        alignment(lambda X: [0, 0], lambda X: [0], X)


def test_empty_input_is_treated_as_agreement():
    assert alignment(lambda X: [], lambda X: [], np.empty((0, 2))) == 1.0


def test_custom_agreement_function_is_used():
    assert alignment(lambda X: [1] * len(X), lambda X: [2] * len(X), X, agree=lambda a, b: 1.0) == 1.0


def test_exact_match_helper():
    assert exact_match(3, 3) == 1.0 and exact_match(3, 4) == 0.0
