import math
import os
from collections import defaultdict
from typing import List, Tuple


def read_date_from_file(filename: str) -> List[Tuple[float]]:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(path, 'r') as file1:
        data_points = file1.read().splitlines()
    data_points = [tuple([float(i) for i in line.split(",")]) for line in data_points]
    return data_points


def get_distance_between_points(point_1, point_2):
    distance = 0
    for i in range(len(point_1)):
        distance += (point_1[i] - point_2[i]) ** 2
    return distance


def get_centroids(bins):
    centroids = []
    for data_points in bins.values():
        sum_dimensions = [0] * len(data_points[0])
        for vector in data_points:
            for i in range(len(vector)):
                sum_dimensions[i] += vector[i]
        centroids.append(tuple([i / len(data_points) for i in sum_dimensions]))
    return centroids


def main():
    k = 7
    epsilon = 0.001
    iterations = 200
    # Read the data
    data_points = read_date_from_file(filename="input_2.txt")
    centroids = data_points[:k]
    iteration_number = 0
    epsilon_condition = True
    while epsilon_condition and iteration_number < iterations:
        epsilon_condition = False
        bins = defaultdict(list)
        for data_point in data_points:
            nearest_distance = get_distance_between_points(centroids[0], data_point)
            nearest_centroid_id = 0
            for centroid_id, centroid in enumerate(centroids):
                distance_to_centroid = get_distance_between_points(centroid, data_point)
                if distance_to_centroid < nearest_distance:
                    nearest_distance = distance_to_centroid
                    nearest_centroid_id = centroid_id
            bins[nearest_centroid_id].append(data_point)
        new_centroids = get_centroids(bins)
        for i in range(len(new_centroids)):
            centroid = centroids[i]
            new_centroid = new_centroids[i]
            distance_between_centroids = math.sqrt(sum([(x0 - x1) ** 2 for x0, x1 in zip(centroid, new_centroid)]))
            if distance_between_centroids >= epsilon:
                epsilon_condition = True
        centroids = new_centroids
        iteration_number += 1

    for centroid in centroids:
        print(centroid)


if __name__ == '__main__':
    main()
