import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from germany_beer_map.src.algorithms.scaling import scale_contour, calculate_centroid, translate_contour, calculate_mse
from germany_beer_map.src.algorithms.spatial_dist_min import haversine
from germany_beer_map.src.algorithms.geocode import convert_address_to_coords

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

def test_geocode():
	# Test with a single address, Plenarbereich Reichstagsgebäude.
	# Of known coordinates: 51.5186111, 13.3761111 (52°31′07″N 13°22′34″E)
	address = "Platz der Republik 1, 11011 Berlin, Germany"
	expected = (13.376296140954109, 52.518671200312156)
	result = tuple(convert_address_to_coords(address=address)[-1])
	print("result: ", *result, "expected: ", *expected)
	# Check if the distance is within 50 meters
	assert haversine(*expected, *result) * 1000 <= 50