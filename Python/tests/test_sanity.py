import math
import os

import pytest

from Python.kmeans import read_date_from_file, get_centroids_from_data_points, DEFAULT_ITERATIONS_NUMBER, EPSILON

test_scenarios = [
    (i, 7, DEFAULT_ITERATIONS_NUMBER) for i in range(1, 21)
]


def compare_data_points(data_point, other_data_point):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=EPSILON)


@pytest.mark.parametrize("i,k,iterations", test_scenarios)
def test_sanity(k, iterations, i):
    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'input_{i}.txt')
    data_points = read_date_from_file(filepath)
    centroids = get_centroids_from_data_points(data_points, k, iterations)
    expected_centroids = read_date_from_file(f'output_{i}.txt')
    assert len(centroids) == len(expected_centroids)
    for i, centroid in enumerate(centroids):
        compare_data_points(centroid, expected_centroids[i])
