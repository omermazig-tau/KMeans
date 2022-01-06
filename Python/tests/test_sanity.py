import math
import os
import pytest

from Python.kmeans import read_date_from_file, get_centroids_from_data_points, DEFAULT_ITERATIONS_NUMBER, \
    DEFAULT_EPSILON
from Python.kmeans_pp import _get_centroids_from_c

test_scenarios = [
    (i, 7, DEFAULT_ITERATIONS_NUMBER) for i in range(1, 21)
]


def compare_data_points(data_point, other_data_point, epsilon):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=epsilon)


# `k` param is necessary for callback compatibility
# noinspection PyUnusedLocal
def _get_centroids_from_python(data_points, initial_centroids, iterations, k, epsilon):
    return get_centroids_from_data_points(data_points, initial_centroids, iterations, epsilon)


@pytest.mark.parametrize("get_centroids_callback", [
    _get_centroids_from_c,
    _get_centroids_from_python
])
@pytest.mark.parametrize("i,k,iterations", [(1, 3, 100), (2, 7, None), (3, 15, 300)])
def test_sanity_original_examples(k, iterations, i, get_centroids_callback):
    epsilon = DEFAULT_EPSILON
    if iterations is None:
        iterations = DEFAULT_ITERATIONS_NUMBER

    base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kmeans_files")
    input_filepath = os.path.join(base_folder, f"input_{i}_orig.txt")
    output_filepath = os.path.join(base_folder, f"output_{i}_orig.txt")
    get_and_assert_new_centroids(k, iterations, epsilon, get_centroids_callback, input_filepath, output_filepath)


@pytest.mark.parametrize("get_centroids_callback", [
    _get_centroids_from_c,
    _get_centroids_from_python
])
@pytest.mark.parametrize("i,k,iterations", test_scenarios)
def test_sanity(k, iterations, i, get_centroids_callback):
    epsilon = DEFAULT_EPSILON

    base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kmeans_files")
    input_filepath = os.path.join(base_folder, f"input_{i}.txt")
    output_filepath = os.path.join(base_folder, f"output_{i}.txt")
    get_and_assert_new_centroids(k, iterations, epsilon, get_centroids_callback, input_filepath, output_filepath)


def get_and_assert_new_centroids(k, iterations, epsilon, get_centroids_callback, input_filepath, output_filepath):
    data_points = read_date_from_file(input_filepath)
    initial_centroids = data_points[:k]
    centroids = get_centroids_callback(data_points, initial_centroids, iterations, k, epsilon)
    expected_centroids = read_date_from_file(output_filepath)
    assert len(centroids) == len(expected_centroids)
    for i, centroid in enumerate(centroids):
        compare_data_points(centroid, expected_centroids[i], epsilon)
