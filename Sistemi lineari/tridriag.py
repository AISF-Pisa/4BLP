import time
import numpy as np

def solve_tridiagonal(diag, sup_diag, inf_diag, b):
    '''
    Function to solve a tridiagonal system using Gaussian elimination.

    Parameters
    ----------
    diag : 1darray
        A(i, i) = diag
    sup_diag : 1darray
        A(i, i + 1) = sup_diag
    inf_diag : 1darray
        A(i - 1, i) = inf_diag
    b : array
        RHS of the equation

    Returns
    -------
    x : 1darray
        solution of the sistem
    '''

    N    = len(diag)
    d_ii = np.copy(diag)
    d_up = np.copy(sup_diag)
    d_lo = np.copy(inf_diag)
    b    = np.copy(b)
    x    = np.zeros(N)

    # Forward elimination
    for i in range(1, N):
        if d_ii[i-1] == 0.0:
            raise ValueError("Division by zero, non-invertible matrix")
        factor = d_lo[i-1] / d_ii[i-1]
        d_ii[i] -= factor * d_up[i-1]
        b[i] -= factor * b[i-1]

    # Back substitution
    if d_ii[-1] == 0.0:
        raise ValueError("Division by zero, non-invertible matrix")
    
    x[-1] = b[-1] / d_ii[-1]
    for i in range(N-2, -1, -1):
        b[i] -= d_up[i] * x[i+1]
        if d_ii[i] == 0.0:
            raise ValueError("Division by zero, non-invertible matrix")
        x[i] = b[i] / d_ii[i]
    
    return x


if __name__ == "__main__":
    np.random.seed(69420)
    N  = 1000
    d1 = np.random.normal(size=[N])
    d2 = np.random.normal(size=[N-1])
    d3 = np.random.normal(size=[N-1])
    A = np.diag(d2, -1) + np.diag(d1, 0) + np.diag(d3, 1)
    b = np.random.normal(size=[N])

    start = time.time()
    x1 = solve_tridiagonal(d1, d3, d2, b)
    print(f"Elapsed time = {time.time()-start}")

    start = time.time()
    x2 = np.linalg.solve(A, b)
    print(f"Elapsed time = {time.time()-start}")

    d = np.sqrt(np.sum((x1 - x2)**2))
    print(f'difference with numpy = {d}')