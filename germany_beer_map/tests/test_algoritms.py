import numpy as np
import pytest
from germany_beer_map.src.algorithms.scaling import scale_contour, calculate_centroid, translate_contour, calculate_mse

def test_scale_contour():
    contour = np.array([[1, 1], [2, 2]])
    scale_factor = 2
    expected = np.array([[2, 2], [4, 4]])
    assert np.array_equal(scale_contour(contour, scale_factor), expected)

def test_calculate_centroid():
    contour = np.array([[1, 1], [23, 23], [3, 3]])
    expected = np.array([9, 9])
    assert np.array_equal(calculate_centroid(contour), expected)

def test_translate_contour():
    contour1 = np.array([[1, 1], [23, 23], [3, 3]])
    contour2 = np.array([[5, 5], [25, 25], [3, 3]])
    expected = np.array([[[3,  3], [23, 23], [1,  1]]])
    assert np.array_equal(translate_contour(contour1, contour2), expected)

def test_calculate_mse():
    contour1 = np.array([[1, 1], [2, 2]])
    contour2 = np.array([[3, 3], [4, 4]])
    expected = 4.0
    assert calculate_mse(contour1, contour2) == expected