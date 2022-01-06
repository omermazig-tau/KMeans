import itertools
import math
import os
from typing import List

import pytest

from Python.kmeans import read_date_from_file, get_centroids_from_data_points, DEFAULT_ITERATIONS_NUMBER, \
    DEFAULT_EPSILON

import kmeans_c_api

test_scenarios = [
    (i, 7, DEFAULT_ITERATIONS_NUMBER) for i in range(1, 21)
]


def compare_data_points(data_point, other_data_point, epsilon):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=epsilon)


def get_matrix_from_flattened_list(k, cols, flatten_centroids: List[float]):
    matrix = []
    for i in range(k):
        temp = []
        for j in range(cols):
            temp.append(flatten_centroids[j + i * cols])
        matrix.append(temp)
    return matrix


def _get_centroids_from_c(data_points, initial_centroids, iterations, k, epsilon):
    rows = len(data_points)
    cols = len(data_points[0])
    flatten_initial_centroids = tuple(itertools.chain.from_iterable(initial_centroids))
    flatten_data_points = tuple(itertools.chain.from_iterable(data_points))
    flatten_centroids = kmeans_c_api.get_new_centroids_api([iterations, rows, cols, k, epsilon,
                                                            flatten_initial_centroids, flatten_data_points])

    return get_matrix_from_flattened_list(k, cols, flatten_centroids)


# `k` param is necessary for callback compatibility
# noinspection PyUnusedLocal
def _get_centroids_from_python(data_points, initial_centroids, iterations, k, epsilon):
    return get_centroids_from_data_points(data_points, initial_centroids, iterations, epsilon)


@pytest.mark.parametrize("get_centroids_callback", [
    _get_centroids_from_c,
    _get_centroids_from_python
])
@pytest.mark.parametrize("i,k,iterations", test_scenarios)
def test_sanity(k, iterations, i, get_centroids_callback):
    epsilon = DEFAULT_EPSILON

    filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'input_{i}.txt')
    data_points = read_date_from_file(filepath)
    initial_centroids = data_points[:k]
    centroids = get_centroids_callback(data_points, initial_centroids, iterations, k, epsilon)
    expected_centroids = read_date_from_file(f'output_{i}.txt')
    assert len(centroids) == len(expected_centroids)
    for i, centroid in enumerate(centroids):
        compare_data_points(centroid, expected_centroids[i], epsilon)
