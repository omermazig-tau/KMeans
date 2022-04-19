
def print_matrix(matrix):
    for row in matrix:
        print(*[f"{i:.4f}".rstrip('0') for i in row], sep=",")
