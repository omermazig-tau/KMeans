import itertools
import sys
import pandas as pd

from Python.common import print_matrix, get_matrix_from_flattened_list
from Python.kmeans_pp import apply_kmeans_pp, print_output, DEFAULT_ITERATIONS_NUMBER
from Python.kmeans import read_date_from_file
from Python import spkmeans_api


def parse_command_line():
    if len(sys.argv) == 4:
        k, goal, file_name = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
    else:
        raise ValueError

    if k < 0 or goal not in ['spk', 'wam', 'ddg', 'lnorm', 'jacobi']:
        raise ValueError

    return k, goal, file_name


def spk(flatten_matrix, rows, cols, k):
    flatten_matrix_result = spkmeans_api.get_spk_matrix(rows, cols, k, flatten_matrix)
    k = len(flatten_matrix_result) / rows
    if k == int(k):
        k = int(k)
    else:
        raise ValueError("K is somehow a fraction, meaning we fucked up")

    matrix_result = get_matrix_from_flattened_list(rows, k, flatten_matrix_result)
    matrix_result = pd.DataFrame(matrix_result)
    centroids, initial_centroids_indexes = apply_kmeans_pp(k, DEFAULT_ITERATIONS_NUMBER, 0.0, matrix_result)
    return centroids, initial_centroids_indexes


def preform_specific_goal(flatten_matrix, rows, cols, goal):
    if goal == 'wam':
        flatten_matrix_result = spkmeans_api.get_weight_adjacency_matrix(rows, cols, flatten_matrix)
    elif goal == 'ddg':
        flatten_matrix_result = spkmeans_api.get_diagonal_degree_matrix(rows, cols, flatten_matrix)
    elif goal == 'lnorm':
        flatten_matrix_result = spkmeans_api.get_normalized_graph_laplacian(rows, cols, flatten_matrix)
    elif goal == 'jacobi':
        flatten_matrix_result = spkmeans_api.get_jacobi_matrix(rows, cols, flatten_matrix)
        rows += 1
    else:
        # Impossible to get here
        raise ValueError

    matrix_result = get_matrix_from_flattened_list(rows, cols, flatten_matrix_result)
    return matrix_result


def main():
    try:
        try:
            k, goal, file_path = parse_command_line()
        except ValueError:
            print("Invalid Input!")
            return

        matrix = read_date_from_file(file_path)
        rows = len(matrix)
        cols = len(matrix[0])
        flatten_matrix = tuple(itertools.chain.from_iterable(matrix))

        if goal == 'spk':
            centroids, initial_centroids_indexes = spk(flatten_matrix, rows, cols, k)
            print_output(centroids, initial_centroids_indexes)
        else:
            matrix_result = preform_specific_goal(flatten_matrix, rows, cols, goal)
            print_matrix(matrix_result)

    except Exception as e:
        print("An Error Has Occurred")
        # TODO - remove the raise before submission
        raise e


if __name__ == '__main__':
    main()
