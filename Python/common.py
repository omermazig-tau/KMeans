from typing import List


def print_matrix(matrix: List[List[float]]):
    for row in matrix:
        print(*[f"{i:.4f}".rstrip('0') for i in row], sep=",")


def get_matrix_from_flattened_list(rows, cols, flatten_centroids: List[float]) -> List[List[float]]:
    matrix = []
    for i in range(rows):
        temp = []
        for j in range(cols):
            temp.append(flatten_centroids[j + i * cols])
        matrix.append(temp)
    return matrix
