import numpy as np

class ConjugateGradientSolver:
    def __init__(self, A: np.array, b: np.array) -> None:
        self.A = A
        self.b = b

    def solve(self) -> np.array:
        n = self.A.shape[0]
        r = self.b - np.dot(self.A, self.solve_rhs())
        alpha = 0
        x = np.zeros((n,))

        while r.norm() > 1e-8:
            r_dot_p = np.dot(r, x)
            p = r - alpha * x
            r -= np.dot(p, self.A) * x
            alpha *= 2 / (r_dot_p**2)
            x = p / r_dot_p

        return x
