import math
import os
import sys
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
    for bin_id in range(len(bins)):
        data_points = bins[bin_id]
        sum_dimensions = [0] * len(data_points[0])
        for vector in data_points:
            for i in range(len(vector)):
                sum_dimensions[i] += vector[i]
        centroids.append(tuple([i / len(data_points) for i in sum_dimensions]))
    return centroids


def parse_command_line():
    if len(sys.argv) == 5:
        k, iterations, file_input, file_output = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
    elif len(sys.argv) == 4:
        k, iterations, file_input, file_output = int(sys.argv[1]), 200, sys.argv[2], sys.argv[3]
    else:
        raise ValueError

    return k, iterations, file_input, file_output


def main():
    try:
        try:
            k, iterations, file_input, file_output = parse_command_line()
        except ValueError:
            print("Invalid Input!")
            return

        epsilon = 0.001

        # Read the data
        data_points = read_date_from_file(filename=file_input)
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

        write_centroids_to_file(file_output, centroids)

    except:
        print("An Error Has Occurred")


def write_centroids_to_file(file_output, centroids):
    with open(file_output, "w") as f:
        for centroid in centroids:
            for element in centroid[:-1:]:
                f.write(str("%.4f" % element) + ",")
            f.write(str("%.4f" % centroid[-1]) + "\n")


if __name__ == '__main__':
    main()
