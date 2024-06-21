import numpy as np


def mean_squared_error(image1, image2):
    if image1.shape != image2.shape:
        raise AssertionError("Input images must have the same dimensions.")

    return np.mean((image1 - image2) ** 2.0)
