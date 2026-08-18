import numpy as np
import pytest

from rashomon_align import InstanceSpace

X = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 15.0]])


def test_bounding_box_comes_from_the_data():
    space = InstanceSpace.from_data(X)
    assert space.lower.tolist() == [0.0, 10.0]
    assert space.upper.tolist() == [1.0, 20.0]


def test_dimensions_match_the_feature_count():
    assert InstanceSpace.from_data(X).dimensions == 2


def test_samples_fall_inside_the_box():
    space = InstanceSpace.from_data(X)
    assert space.contains(space.sample(500, seed=1)).all()


def test_sampling_is_reproducible():
    space = InstanceSpace.from_data(X)
    assert np.array_equal(space.sample(50, seed=7), space.sample(50, seed=7))


def test_different_seeds_give_different_samples():
    space = InstanceSpace.from_data(X)
    assert not np.array_equal(space.sample(50, seed=1), space.sample(50, seed=2))


def test_sample_shape():
    assert InstanceSpace.from_data(X).sample(37, seed=3).shape == (37, 2)


def test_constant_column_is_flagged_as_degenerate():
    constant = np.array([[1.0, 5.0], [1.0, 6.0]])
    assert InstanceSpace.from_data(constant).volume_is_degenerate()


def test_inverted_bounds_are_rejected():
    with pytest.raises(ValueError):
        InstanceSpace(np.array([1.0]), np.array([0.0]))


def test_one_dimensional_input_is_rejected():
    with pytest.raises(ValueError):
        InstanceSpace.from_data(np.array([1.0, 2.0]))


def test_column_names_are_picked_up_from_a_frame():
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame(X, columns=["a", "b"])
    assert InstanceSpace.from_data(frame).names == ("a", "b")
