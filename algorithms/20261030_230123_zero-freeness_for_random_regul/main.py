import unittest
import math
import cmath
from typing import List
from core import (
    Graph,
    generate_random_regular_graph,
    independence_polynomial,
    evaluate_polynomial,
    durand_kerner,
    is_zero_free,
)

class TestIndependencePolynomial(unittest.TestCase):
    def test_two_isolated_vertices(self):
        g = Graph()
        v0 = g.add_vertex()
        v1 = g.add_vertex()
        coeffs = independence_polynomial(g)
        self.assertEqual(coeffs, [1, 2, 1])  # (1 + x)^2

    def test_path_three_vertices(self):
        g = Graph()
        v0 = g.add_vertex()
        v1 = g.add_vertex()
        v2 = g.add_vertex()
        g.add_edge(v0, v1)
        g.add_edge(v1, v2)
        coeffs = independence_polynomial(g)
        self.assertEqual(coeffs, [1, 3, 1])  # empty, three singles, ends together

    def test_random_regular_small(self):
        n, d = 6, 3
        g = generate_random_regular_graph(n, d, seed=42)
        coeffs = independence_polynomial(g)
        # Polynomial evaluated at λ=0 must be 1 (empty set)
        self.assertAlmostEqual(evaluate_polynomial(coeffs, 0), 1.0)

    def test_durand_kerner_simple(self):
        # Polynomial x^2 - 1 => coeffs [-1, 0, 1]
        coeffs = [-1, 0, 1]
        roots = durand_kerner(coeffs, max_iter=200, tol=1e-14)
        # Sort by real part for deterministic comparison
        roots.sort(key=lambda z: (z.real, z.imag))
        self.assertTrue(any(abs(r - 1) < 1e-10 for r in roots))
        self.assertTrue(any(abs(r + 1) < 1e-10 for r in roots))

    def test_zero_freeness_region(self):
        # Polynomial 1 + x has root at -1
        coeffs = [1, 1]
        region = lambda z: abs(z) < 0.5  # disc of radius 0.5 around 0
        self.assertTrue(is_zero_free(coeffs, region))
        # Larger disc contains the root
        region2 = lambda z: abs(z) < 1.5
        self.assertFalse(is_zero_free(coeffs, region2))

    def test_evaluate_complex(self):
        coeffs = [2, -3, 1]  # 1*x^2 -3*x +2 = (x-1)(x-2)
        val = evaluate_polynomial(coeffs, 1+1j)
        expected = (1+1j - 1) * (1+1j - 2)
        self.assertAlmostEqual(val.real, expected.real, places=10)
        self.assertAlmostEqual(val.imag, expected.imag, places=10)

if __name__ == '__main__':
    unittest.main()
