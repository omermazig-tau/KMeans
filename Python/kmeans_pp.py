import numpy as np
import pandas as pd
import os
import sys
from kmeans import DEFAULT_ITERATIONS_NUMBER


def parse_command_line():
    if len(sys.argv) == 5:
        k, iterations, file_name_1, file_name_2 = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    elif len(sys.argv) == 4:
        k, iterations, file_name_1, file_name_2 = int(sys.argv[1]), DEFAULT_ITERATIONS_NUMBER, sys.argv[2], sys.argv[3]
    else:
        raise ValueError

    if k < 0 or iterations < 0:
        raise ValueError

    return k, iterations, file_name_1, file_name_2


def main():
    try:
        try:
            k, iterations, file_name_1, file_name_2 = parse_command_line()
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

        list_centroids = get_list_of_initial_centroids(k, data_points_1, data_points_2)
        print(list_centroids)

    except:
        print("An Error Has Occurred")
        return


def get_list_of_initial_centroids(k, data_points_1, data_points_2):
    data_points = pd.merge(data_points_1, data_points_2, how='outer')  # Merged data points
    cols = data_points.shape[1]

    np.random.seed(0)
    centroids = data_points.iloc[np.random.choice(data_points.shape[0], 1)].copy()

    for i in range(1, k):
        data_points['Distance' + str(i)] = (pow((data_points - centroids.iloc[i - 1]), 2)).sum(axis=1)
        data_points['D'] = data_points.iloc[:, -i:].min(axis=1)
        data_points['D'] = data_points['D'] / data_points['D'].sum()
        next_centroid = np.random.choice(data_points.shape[0], 1, p=data_points['D'].to_numpy())
        centroids = pd.merge(centroids, data_points.iloc[next_centroid, 0:cols], how='outer')
        data_points.drop(columns=['D'], inplace=True)

    return centroids.to_numpy().tolist()


if __name__ == '__main__':
    main()
