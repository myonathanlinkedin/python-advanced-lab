import unittest
from typing import List

from core import Vec2, Mat3, transform_vertices, Canvas, Rasterizer


class TestVec2(unittest.TestCase):
    def test_arithmetic(self) -> None:
        a = Vec2(1.0, 2.0)
        b = Vec2(3.0, -1.0)
        self.assertEqual(a + b, Vec2(4.0, 1.0))
        self.assertEqual(a - b, Vec2(-2.0, 3.0))
        self.assertEqual(a * 2, Vec2(2.0, 4.0))
        self.assertEqual(3 * b, Vec2(9.0, -3.0))

    def test_normalization(self) -> None:
        v = Vec2(3.0, 4.0)
        n = v.normalized()
        self.assertAlmostEqual(n.length(), 1.0)
        self.assertAlmostEqual(n.x, 0.6)
        self.assertAlmostEqual(n.y, 0.8)


class TestMat3(unittest.TestCase):
    def test_identity(self) -> None:
        m = Mat3.identity()
        v = Vec2(5.0, -7.0)
        self.assertEqual(m.apply(v), v)

    def test_translation(self) -> None:
        m = Mat3.translation(2.0, -3.0)
        v = Vec2(1.0, 1.0)
        self.assertEqual(m.apply(v), Vec2(3.0, -2.0))

    def test_rotation(self) -> None:
        rad = math.pi / 2
        m = Mat3.rotation(rad)
        v = Vec2(1.0, 0.0)
        r = m.apply(v)
        self.assertAlmostEqual(r.x, 0.0, places=7)
        self.assertAlmostEqual(r.y, 1.0, places=7)

    def test_combination(self) -> None:
        m = Mat3.translation(1, 2) @ Mat3.rotation(math.pi) @ Mat3.scaling(2, 2)
        v = Vec2(1, 0)
        res = m.apply(v)
        # Scale by 2 -> (2,0), rotate 180° -> (-2,0), translate -> (-1,2)
        self.assertAlmostEqual(res.x, -1.0)
        self.assertAlmostEqual(res.y, 2.0)


class TestTransform(unittest.TestCase):
    def test_batch_transform(self) -> None:
        verts = [Vec2(0, 0), Vec2(1, 0), Vec2(0, 1)]
        mat = Mat3.translation(5, 5)
        out = transform_vertices(verts, mat)
        expected = [Vec2(5, 5), Vec2(6, 5), Vec2(5, 6)]
        self.assertEqual(out, expected)


class TestRasterizer(unittest.TestCase):
    def test_line_horizontal(self) -> None:
        canvas = Canvas(10, 2)
        Rasterizer.draw_line(canvas, Vec2(1, 0), Vec2(8, 0))
        for x in range(1, 9):
            self.assertEqual(canvas.get_pixel(x, 0), 1)
        self.assertEqual(canvas.get_pixel(0, 0), 0)

    def test_line_diagonal(self) -> None:
        canvas = Canvas(5, 5)
        Rasterizer.draw_line(canvas, Vec2(0, 0), Vec2(4, 4))
        for i in range(5):
            self.assertEqual(canvas.get_pixel(i, i), 1)

    def test_triangle_fill(self) -> None:
        canvas = Canvas(6, 6)
        v0 = Vec2(1, 1)
        v1 = Vec2(4, 1)
        v2 = Vec2(1, 4)
        Rasterizer.draw_triangle(canvas, v0, v1, v2)
        # Count filled pixels; the right‑angled triangle should cover 6 pixels (including edges)
        filled = sum(1 for c in canvas.buffer if c == 1)
        self.assertEqual(filled, 6)

    def test_degenerate_triangle(self) -> None:
        canvas = Canvas(3, 3)
        v = Vec2(1, 1)
        Rasterizer.draw_triangle(canvas, v, v, v)  # zero area
        self.assertTrue(all(p == 0 for p in canvas.buffer))


def demo() -> None:
    """Simple demo rendering a rotated square."""
    size = 20
    canvas = Canvas(size, size)
    # Square vertices centred at (10,10)
    half = 5
    verts = [
        Vec2(-half, -half),
        Vec2( half, -half),
        Vec2( half,  half),
        Vec2(-half,  half),
    ]
    # Rotate 45°
    rot = Mat3.rotation(math.radians(45))
    trans = Mat3.translation(size / 2, size / 2)
    model = trans @ rot
    screen = transform_vertices(verts, model)

    # Draw edges
    for i in range(4):
        Rasterizer.draw_line(canvas, screen[i], screen[(i + 1) % 4])

    print(canvas.to_ascii())


if __name__ == "__main__":
    # Run unit tests; if they all pass, show demo.
    unittest.main(exit=False)
    print("\nDemo rendering:\n")
    demo()
