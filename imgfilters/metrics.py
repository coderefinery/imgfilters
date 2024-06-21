import numpy as np
from pytest import approx


def mean_squared_error(image1, image2):
    if image1.shape != image2.shape:
        raise AssertionError("Input images must have the same dimensions.")

    return np.mean((image1 - image2) ** 2.0)


def test_mean_squared_error():
    image1 = np.array(
        [
            [[0.933, 0.936, 0.742], [0.946, 0.68, 0.869], [0.494, 0.517, 0.387]],
            [[0.61, 0.019, 0.448], [0.491, 0.999, 0.238], [0.576, 0.342, 0.692]],
            [[0.905, 0.46, 0.191], [0.83, 0.095, 0.584], [0.629, 0.536, 0.919]],
        ]
    )

    image2 = np.array(
        [
            [[0.325, 0.063, 0.267], [0.712, 0.578, 0.506], [0.389, 0.87, 0.112]],
            [[0.585, 0.356, 0.859], [0.502, 0.463, 0.479], [0.62, 0.215, 0.895]],
            [[0.577, 0.612, 0.569], [0.556, 0.072, 0.939], [0.992, 0.671, 0.737]],
        ]
    )

    result = mean_squared_error(image1, image2)

    assert result == approx(0.1152523)


def test_mean_squared_error_different_dimensions():
    image1 = np.array(
        [
            [[0.933, 0.936, 0.742], [0.946, 0.68, 0.869], [0.494, 0.517, 0.387]],
            [[0.61, 0.019, 0.448], [0.491, 0.999, 0.238], [0.576, 0.342, 0.692]],
            [[0.905, 0.46, 0.191], [0.83, 0.095, 0.584], [0.629, 0.536, 0.919]],
        ]
    )

    image2 = np.array(
        [
            [[0.325, 0.063, 0.267], [0.712, 0.578, 0.506], [0.389, 0.87, 0.112]],
            [[0.585, 0.356, 0.859], [0.502, 0.463, 0.479], [0.62, 0.215, 0.895]],
        ]
    )

    try:
        mean_squared_error(image1, image2)
    except AssertionError as e:
        assert str(e) == "Input images must have the same dimensions."
    else:
        assert False, "Exception not raised"
