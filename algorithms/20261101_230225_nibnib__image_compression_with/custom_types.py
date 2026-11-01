from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

Pixel = Tuple[int, int, int]  # RGB pixel with values 0-255

@dataclass(frozen=True)
class Image:
    width: int
    height: int
    data: List[List[Pixel]]  # data[y][x] gives Pixel at (x, y)

    def __post_init__(self):
        if self.width < 0 or self.height < 0:
            raise ValueError("Image dimensions must be non-negative")
        if len(self.data) != self.height:
            raise ValueError("Data height does not match image height")
        for row in self.data:
            if len(row) != self.width:
                raise ValueError("Data width does not match image width")
