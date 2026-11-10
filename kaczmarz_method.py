
import numpy as np


def kaczmarz(A, b, N):
    """
    Solve a linear system Ax = b using the Kaczmarz method.

    Parameters:
        A: Coefficient matrix
        b: Right-hand-side vector
        N: Number of iterations

    Returns:
        p: Approximate solution
        iterations: Number of iterations performed
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()

    if A.ndim != 2 or A.shape[0] != len(b):
        raise ValueError("Incompatible dimensions for A and b.")

    m, n = A.shape
    p = np.zeros(n)

    iterations = 0
    k = 0

    while iterations < N:
        i = k % m
        a = A[i]

        if np.dot(a, a) == 0:
            raise ValueError("A contains a zero row.")

        p = p + ((b[i] - a @ p) / (a @ a)) * a

        k += 1
        iterations += 1

    return p, iterations


# Example matrix and vector from original research
A = np.array([
    [1.0, 0.0, 2.0],
    [-1.0, 1.0, 2.0],
    [1.0, 2.0, 3.0],
    [1.0, 2.0, 3.0]
])

b = np.array([2.0, 2.0, 2.0, -5.0])

solution, number_of_iterations = kaczmarz(A, b, 10000)

print("# of iterations:", number_of_iterations)
print("The solution is:", solution)

# Evaluate residual
print("Norm of residual:", np.linalg.norm(b - A @ solution))
