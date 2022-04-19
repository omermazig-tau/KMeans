import itertools
import os
import sys

from Python.common import print_matrix, get_matrix_from_flattened_list
from Python.kmeans_pp import apply_kmeans_pp
from Python import spkmeans_api


def parse_command_line():
    if len(sys.argv) == 4:
        k, goal, file_name = int(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3])
    else:
        raise ValueError

    if k < 0 or goal not in ['spk', 'wam', 'ddg', 'lnorm', 'jacobi']:
        raise ValueError

    return k, goal, file_name


def main():
    try:
        try:
            k, goal, file_name = parse_command_line()
        except ValueError:
            print("Invalid Input!")
            return

        with open(file_name, 'r') as _file:
            matrix = [[float(num) for num in line.split(',')] for line in _file]
        rows = len(matrix)
        cols = len(matrix[0])
        flatten_matrix = tuple(itertools.chain.from_iterable(matrix))

        if goal == 'wam':
            flatten_matrix_result = spkmeans_api.get_weight_adjacency(rows, cols, flatten_matrix)
        else:
            raise ValueError

        matrix_result = get_matrix_from_flattened_list(rows, cols, flatten_matrix_result)
        print_matrix(matrix_result)

    except Exception as e:
        print("An Error Has Occurred")
        # TODO - remove the raise before submission
        raise e


if __name__ == '__main__':
    main()
