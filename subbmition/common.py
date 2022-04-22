from typing import List


def minus_zero_to_zero(val):
    return val if not -0.0001 < val < 0 else 0


def print_eigen_values(eigen_values: List[float]):
    print(*[f"{minus_zero_to_zero(i):.4f}" for i in eigen_values], sep=",")


def print_matrix(matrix: List[List[float]]):
    for row in matrix:
        print(*[f"{i:.4f}" for i in row], sep=",")


def get_matrix_from_flattened_list(rows, cols, flatten_centroids: List[float]) -> List[List[float]]:
    matrix = []
    for i in range(rows):
        temp = []
        for j in range(cols):
            temp.append(flatten_centroids[j + i * cols])
        matrix.append(temp)
    return matrix
