import math
import os
from typing import Tuple, List

import pytest

from Python.kmeans_pp import DEFAULT_ITERATIONS_NUMBER, apply_kmeans_pp


def read_date_from_file(filepath: str) -> List[Tuple[float]]:
    with open(filepath, 'r') as file1:
        data_points = file1.read().splitlines()
    data_points = [tuple([float(i) for i in line.split(",")]) for line in data_points]
    return data_points


def compare_data_points(data_point, other_data_point, epsilon):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=epsilon)


@pytest.mark.parametrize("i,k,iterations,epsilon", [(1, 3, 333, 0), (2, 7, None, 0), (3, 15, 750, 0)])
def test_sanity_original_examples(k, iterations, i, epsilon):
    if iterations is None:
        iterations = DEFAULT_ITERATIONS_NUMBER

    base_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kmeans_pp_files")
    input_path_1 = os.path.join(base_folder, f"input_{i}_db_1.txt")
    input_path_2 = os.path.join(base_folder, f"input_{i}_db_2.txt")
    output_path = os.path.join(base_folder, f"output_{i}.txt")
    centroids, initial_centroids_indexes = apply_kmeans_pp(k, iterations, epsilon, input_path_1, input_path_2)
    get_and_assert_new_centroids(centroids, initial_centroids_indexes, output_path, epsilon)


def get_and_assert_new_centroids(centroids, initial_centroids_indexes, output_path, epsilon):
    expected_output = read_date_from_file(output_path)
    expected_indexes = expected_output[0]
    expected_centroids = expected_output[1:]

    assert initial_centroids_indexes == expected_indexes

    assert len(centroids) == len(expected_centroids)
    for i, centroid in enumerate(centroids):
        compare_data_points(centroid, expected_centroids[i], epsilon)
