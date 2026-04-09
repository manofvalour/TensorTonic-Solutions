import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    row = len(A)
    column = len(A[0])

    transposed = [[0 for _ in range(row)] for _ in range(column)]

    for i in range(row):
        for j in range(column):
            transposed[j][i]= A[i][j]
    
    return np.array(transposed)
