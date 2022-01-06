import sys

import numpy as np
import pandas as pd
import os

from Python.kmeans import DEFAULT_ITERATIONS_NUMBER


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


def main():
    try:
        try:
            k, iterations, epsilon, file_name_1, file_name_2 = parse_command_line()
        except ValueError:
            print("Invalid Input!")
            return

        # Read the data
        filepath1 = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name_1)
        filepath2 = os.path.join(os.path.dirname(os.path.realpath(__file__)), file_name_2)

        data_points_1 = pd.read_csv(filepath1, header=None)
        data_points_2 = pd.read_csv(filepath2, header=None)

        if k > len(data_points_1) + len(data_points_2):
            raise ValueError("Number of clusters can't be higher than number of points")

        list_index, list_centroids = get_list_of_initial_centroids(k, data_points_1, data_points_2)
        print_output(list_centroids, list_index)  # TODO list_centroids should be the updated centroids

    except:
        print("An Error Has Occurred")
        return


def get_list_of_initial_centroids(k, data_points_1, data_points_2):
    data_points = pd.merge(data_points_1, data_points_2, on=0, how='inner')
    cols = data_points.shape[1]

    np.random.seed(0)
    centroids = data_points.iloc[np.random.choice(data_points.shape[0], 1)]

    for i in range(1, k):
        data_points['Distance' + str(i)] = (pow((data_points - centroids.iloc[i - 1][1:]), 2)).sum(
            axis=1)  # adding new column to data_points of distance between each vector to i centroid
        data_points['D'] = data_points.iloc[:, -i:].min(axis=1)  # adding new column of the minimum distance squared
        data_points['D'] = data_points['D'] / data_points['D'].sum()  # calculating probability for each line
        next_centroid = np.random.choice(data_points.shape[0], 1, p=data_points['D'].to_numpy())  # according to probabilities choosing new centroid
        centroids = pd.merge(centroids, data_points.iloc[next_centroid, 0:cols],
                             how='outer')  # adding new centroid to data of centroids
        data_points.drop(columns=['D'], inplace=True)  # remove the column D on dataPoints

    return centroids[0].to_numpy().tolist(), centroids.drop(columns=[0], axis=1).to_numpy().tolist()


def print_output(list_centroids, list_index):
    print(*[int(i) for i in list_index], sep=",")
    for centroid in list_centroids:
        print(*[format(i, ".4f") for i in centroid], sep=",")


if __name__ == '__main__':
    main()
