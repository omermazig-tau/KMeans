import itertools
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple

import spkmeans_api
import mykmeanssp

DEFAULT_ITERATIONS_NUMBER = 300


# KMMEANS++ Functions
def get_list_of_initial_centroids(k, data_points):
    index_initial_centroids = list()
    data_points.sort_index(inplace=True)
    cols = data_points.shape[1]

    np.random.seed(0)
    next_centroid = np.random.choice(data_points.shape[0], 1)
    index_initial_centroids.append(next_centroid)

    centroids = data_points.iloc[next_centroid]

    for i in range(1, k):
        data_points['Distance' + str(i)] = (pow((data_points - centroids.iloc[i - 1]), 2)).sum(
            axis=1)  # adding new column to data_points of distance between each vector to i centroid
        data_points['D'] = data_points.iloc[:, -i:].min(axis=1)  # adding new column of the minimum distance squared
        data_points['D'] = data_points['D'] / data_points['D'].sum()  # calculating probability for each line
        next_centroid = np.random.choice(data_points.shape[0], 1, p=np.array(
            data_points['D'].values.tolist()))  # according to probabilities choosing new centroid
        index_initial_centroids.append(next_centroid)
        centroids = pd.merge(centroids, data_points.iloc[next_centroid, 0:cols],
                             how='outer')  # adding new centroid to data of centroids
        data_points.drop(columns=['D'], inplace=True)  # remove the column D on dataPoints

    return [array[0] for array in index_initial_centroids], centroids.values.tolist()


def _get_centroids_from_c(data_points, initial_centroids, iterations, k, epsilon):
    rows = len(data_points)
    cols = len(data_points[0])
    flatten_initial_centroids = tuple(itertools.chain.from_iterable(initial_centroids))
    flatten_data_points = tuple(itertools.chain.from_iterable(data_points))
    flatten_centroids = mykmeanssp.fit(iterations, rows, cols, k, epsilon,
                                       flatten_initial_centroids, flatten_data_points)

    return get_matrix_from_flattened_list(k, cols, flatten_centroids)


def apply_kmeans_pp(k, iterations, epsilon, data_points):
    initial_centroids_indexes, initial_centroids = get_list_of_initial_centroids(k, data_points.copy())
    if iterations == 0:
        return initial_centroids, initial_centroids_indexes
    else:
        data_points = data_points.to_numpy().tolist()
        centroids = _get_centroids_from_c(data_points, initial_centroids, iterations, k, epsilon)
        return centroids, initial_centroids_indexes


# Print functions
def print_jacobi_matrix(matrix):
    print_eigen_values(matrix[0])
    print_matrix(matrix[1:])


def print_eigen_values(eigen_values: List[float]):
    def minus_zero_to_zero(val):
        return val if not -0.0001 < val < 0 else 0

    print(*[f"{minus_zero_to_zero(i):.4f}" for i in eigen_values], sep=",")


def print_matrix(matrix: List[List[float]]):
    for row in matrix:
        print(*[f"{i:.4f}" for i in row], sep=",")


def print_output(list_centroids, list_index):
    print(*list_index, sep=",")
    print_matrix(list_centroids)


def get_matrix_from_flattened_list(rows, cols, flatten_centroids: List[float]) -> List[List[float]]:
    matrix = []
    for i in range(rows):
        temp = []
        for j in range(cols):
            temp.append(flatten_centroids[j + i * cols])
        matrix.append(temp)
    return matrix


# Parsing functions
def parse_command_line():
    if len(sys.argv) == 4:
        k, goal, file_name = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
    else:
        raise ValueError

    if k < 0 or k == 1 or goal not in ['spk', 'wam', 'ddg', 'lnorm', 'jacobi']:
        raise ValueError

    return k, goal, file_name


def read_date_from_file(filepath: str) -> List[Tuple[float]]:
    with open(filepath, 'r') as file1:
        data_points = file1.read().splitlines()
    data_points = [tuple([float(i) for i in line.split(",")]) for line in data_points]
    return data_points


# API Functions
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
        cols = rows
    elif goal == 'ddg':
        flatten_matrix_result = spkmeans_api.get_diagonal_degree_matrix(rows, cols, flatten_matrix)
        cols = rows
    elif goal == 'lnorm':
        flatten_matrix_result = spkmeans_api.get_normalized_graph_laplacian(rows, cols, flatten_matrix)
        cols = rows
    elif goal == 'jacobi':
        flatten_matrix_result = spkmeans_api.get_jacobi_matrix(rows, cols, flatten_matrix)
        cols = rows
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
            try:
                centroids, initial_centroids_indexes = spk(flatten_matrix, rows, cols, k)
            except SystemError:
                # That means that K was 1
                print('An Error Has Occurred')
            else:
                print_output(centroids, initial_centroids_indexes)
        else:
            matrix_result = preform_specific_goal(flatten_matrix, rows, cols, goal)
            if goal == 'jacobi':
                print_jacobi_matrix(matrix_result)
            else:
                print_matrix(matrix_result)

    except Exception as e:
        print("An Error Has Occurred")
        # TODO - remove the raise before submission
        raise e


if __name__ == '__main__':
    main()
