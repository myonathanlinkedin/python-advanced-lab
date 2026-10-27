from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def dot(self, other: Vec2) -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> Vec2:
        l = self.length()
        if l == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)


class Mat3:
    """3×3 matrix for 2‑D affine transformations (row‑major)."""

    def __init__(self, rows: Tuple[Tuple[float, float, float],
                                 Tuple[float, float, float],
                                 Tuple[float, float, float]]) -> None:
        self._m = rows

    @staticmethod
    def identity() -> Mat3:
        return Mat3(((1.0, 0.0, 0.0),
                     (0.0, 1.0, 0.0),
                     (0.0, 0.0, 1.0)))

    @staticmethod
    def translation(tx: float, ty: float) -> Mat3:
        return Mat3(((1.0, 0.0, tx),
                     (0.0, 1.0, ty),
                     (0.0, 0.0, 1.0)))

    @staticmethod
    def rotation(rad: float) -> Mat3:
        c, s = math.cos(rad), math.sin(rad)
        return Mat3(((c, -s, 0.0),
                     (s,  c, 0.0),
                     (0.0, 0.0, 1.0)))

    @staticmethod
    def scaling(sx: float, sy: float) -> Mat3:
        return Mat3(((sx, 0.0, 0.0),
                     (0.0, sy, 0.0),
                     (0.0, 0.0, 1.0)))

    def __matmul__(self, other: Mat3) -> Mat3:
        """Matrix multiplication (self * other)."""
        result = []
        for i in range(3):
            row = []
            for j in range(3):
                val = sum(self._m[i][k] * other._m[k][j] for k in range(3))
                row.append(val)
            result.append(tuple(row))
        return Mat3(tuple(result))  # type: ignore[arg-type]

    def apply(self, v: Vec2) -> Vec2:
        """Apply affine transform to a Vec2."""
        x = self._m[0][0] * v.x + self._m[0][1] * v.y + self._m[0][2]
        y = self._m[1][0] * v.x + self._m[1][1] * v.y + self._m[1][2]
        return Vec2(x, y)


def transform_vertices(vertices: List[Vec2], mat: Mat3) -> List[Vec2]:
    return [mat.apply(v) for v in vertices]


class Canvas:
    """Simple integer pixel buffer."""

    def __init__(self, width: int, height: int, background: int = 0) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Canvas dimensions must be positive")
        self.width = width
        self.height = height
        self._bg = background
        self.buffer: List[int] = [background] * (width * height)

    def _index(self, x: int, y: int) -> int:
        return y * self.width + x

    def clear(self, color: int = 0) -> None:
        self.buffer = [color] * (self.width * self.height)

    def set_pixel(self, x: int, y: int, color: int = 1) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buffer[self._index(x, y)] = color

    def get_pixel(self, x: int, y: int) -> int:
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.buffer[self._index(x, y)]
        raise IndexError("Pixel out of bounds")

    def to_ascii(self) -> str:
        rows = []
        for y in range(self.height):
            row = ''.join('#' if self.get_pixel(x, y) else '.' for x in range(self.width))
            rows.append(row)
        return '\n'.join(rows)


class Rasterizer:
    """Software rasterizer mimicking a minimal GPU pipeline."""

    @staticmethod
    def draw_line(canvas: Canvas, p0: Vec2, p1: Vec2, color: int = 1) -> None:
        x0, y0 = int(round(p0.x)), int(round(p0.y))
        x1, y1 = int(round(p1.x)), int(round(p1.y))
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            canvas.set_pixel(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    @staticmethod
    def draw_triangle(canvas: Canvas,
                      v0: Vec2, v1: Vec2, v2: Vec2,
                      color: int = 1) -> None:
        # Bounding box in integer pixel space
        min_x = max(0, int(math.floor(min(v0.x, v1.x, v2.x))))
        max_x = min(canvas.width - 1, int(math.ceil(max(v0.x, v1.x, v2.x))))
        min_y = max(0, int(math.floor(min(v0.y, v1.y, v2.y))))
        max_y = min(canvas.height - 1, int(math.ceil(max(v0.y, v1.y, v2.y))))

        # Edge function
        def edge(a: Vec2, b: Vec2, c: Vec2) -> float:
            return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x)

        area = edge(v0, v1, v2)
        if area == 0:
            return  # Degenerate triangle

        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = Vec2(x + 0.5, y + 0.5)  # pixel centre
                w0 = edge(v1, v2, p)
                w1 = edge(v2, v0, p)
                w2 = edge(v0, v1, p)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) if area > 0 else (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    canvas.set_pixel(x, y, color)


__all__ = [
    "Vec2",
    "Mat3",
    "transform_vertices",
    "Canvas",
    "Rasterizer",
]
