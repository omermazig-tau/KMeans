import itertools
import sys
from typing import List

import numpy as np
import pandas as pd
import os

from Python import mykmeanssp
from Python.common import print_matrix, get_matrix_from_flattened_list

DEFAULT_ITERATIONS_NUMBER = 300


def parse_command_line():
    if len(sys.argv) == 6:
        k, iterations, epsilon, file_input_1, file_input_2 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), \
                                                             sys.argv[4], sys.argv[5]
    elif len(sys.argv) == 5:
        k, iterations, epsilon, file_input_1, file_input_2 = int(sys.argv[1]), DEFAULT_ITERATIONS_NUMBER, \
                                                             float(sys.argv[2]), sys.argv[3], sys.argv[4]
    else:
        raise ValueError

    if k < 0 or iterations < 0 or epsilon < 0:
        raise ValueError

    return k, iterations, epsilon, file_input_1, file_input_2


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


def get_data_points_from_two_files(filepath1, filepath2):
    data_points_1 = pd.read_csv(filepath1, header=None, index_col=0)
    data_points_1.index.names = ['INDEX']
    data_points_2 = pd.read_csv(filepath2, header=None, index_col=0)
    data_points_2.index.names = ['INDEX']

    data_points = pd.merge(data_points_1, data_points_2, on='INDEX', how='inner')
    return data_points


def main():
    try:
        try:
            k, iterations, epsilon, file_name_1, file_name_2 = parse_command_line()
        except ValueError:
            print("Invalid Input!")
            return

        # Read the data
        filepath1 = os.path.realpath(file_name_1)
        filepath2 = os.path.realpath(file_name_2)

        if k == 0:
            centroids, initial_centroids_indexes = [], []
        else:
            data_points = get_data_points_from_two_files(filepath1, filepath2)
            if k > len(data_points):
                raise ValueError("Number of clusters can't be higher than number of points")
            centroids, initial_centroids_indexes = apply_kmeans_pp(k, iterations, epsilon, data_points)
        print_output(centroids, initial_centroids_indexes)

    except:
        print("An Error Has Occurred")


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


def print_output(list_centroids, list_index):
    print(*list_index, sep=",")
    print_matrix(list_centroids)


if __name__ == '__main__':
    main()
