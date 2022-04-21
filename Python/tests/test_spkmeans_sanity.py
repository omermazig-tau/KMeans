import itertools
import math
import os
import pytest

from Python.kmeans import read_date_from_file
from Python.spkmeans import preform_specific_goal, spk
from Python.tests import test_kmeans_pp_sanity

DEFAULT_EPSILON = 0.0001


def compare_data_points(data_point, other_data_point, epsilon):
    for a, b in zip(data_point, other_data_point):
        assert math.isclose(a, b, abs_tol=epsilon)


@pytest.mark.parametrize("i", range(10))
@pytest.mark.parametrize("goal", ['wam', 'ddg', 'lnorm'])
def test_specific_goal_sanity(i, goal):
    input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spkmeans_files")
    output_folder = os.path.join(input_folder, 'outputs', 'py', goal)
    file_prefix = 'jacobi' if goal == 'jacobi' else 'spk'
    file_name = f"{file_prefix}_{i}.txt"
    input_filepath = os.path.join(input_folder, file_name)
    output_filepath = os.path.join(output_folder, file_name)
    get_and_assert_matrix_result(goal, input_filepath, output_filepath)


@pytest.mark.parametrize("i", range(22))
def test_jacobi_sanity(i):
    goal = 'jacobi'
    input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spkmeans_files")
    output_folder = os.path.join(input_folder, 'outputs', 'py', goal)
    file_prefix = 'jacobi' if goal == 'jacobi' else 'spk'
    file_name = f"{file_prefix}_{i}.txt"
    input_filepath = os.path.join(input_folder, file_name)
    output_filepath = os.path.join(output_folder, file_name)
    get_and_assert_matrix_result(goal, input_filepath, output_filepath)


@pytest.mark.parametrize("i", range(10))
def test_spk_sanity(i):
    k = 0
    input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spkmeans_files")
    output_folder = os.path.join(input_folder, 'outputs', 'py', 'spk')
    file_name = f"spk_{i}.txt"
    input_filepath = os.path.join(input_folder, file_name)
    output_filepath = os.path.join(output_folder, file_name)

    matrix = read_date_from_file(input_filepath)
    rows = len(matrix)
    cols = len(matrix[0])
    flatten_matrix = tuple(itertools.chain.from_iterable(matrix))

    centroids, initial_centroids_indexes = spk(flatten_matrix, rows, cols, k)
    test_kmeans_pp_sanity.get_and_assert_new_centroids(centroids, initial_centroids_indexes, output_filepath, DEFAULT_EPSILON)


def get_and_assert_matrix_result(goal, input_filepath, output_filepath):
    matrix = read_date_from_file(input_filepath)
    rows = len(matrix)
    cols = len(matrix[0])
    flatten_matrix = tuple(itertools.chain.from_iterable(matrix))

    matrix_result = preform_specific_goal(flatten_matrix, rows, cols, goal)
    expected_output = read_date_from_file(output_filepath)

    assert len(matrix_result) == len(expected_output)
    for i, row in enumerate(matrix_result):
        compare_data_points(row, expected_output[i], DEFAULT_EPSILON)
