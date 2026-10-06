import numpy as np
from core import ConjugateGradientSolver

def main():
    A = np.random.rand(10, 10)
    b = np.random.rand(10)
    solver = ConjugateGradientSolver(A, b)
    result = solver.solve()
    np.testing.assert_allclose(result, np.linalg.solve(A, b))

if __name__ == '__main__':
    main()
