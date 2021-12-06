import math

import pytest

from Python.kmeans import read_date_from_file, get_centroids_from_data_points, DEFAULT_ITERATIONS_NUMBER

kuku = [
    (1, 3, 600),
    (2, 7, DEFAULT_ITERATIONS_NUMBER),
    (3, 15, 300),
]


def compare_data_points(data_point, other_data_point):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=0.0001)


@pytest.mark.parametrize("i,k,iterations", kuku)
def test_sanity(k, iterations, i):
    data_points = read_date_from_file(f'input_{i}.txt')
    centroids = get_centroids_from_data_points(data_points, k, iterations)
    expected_centroids = read_date_from_file(f'output_{i}.txt')
    assert len(centroids) == len(expected_centroids)
    for i, centroid in enumerate(centroids):
        compare_data_points(centroid, expected_centroids[i])
