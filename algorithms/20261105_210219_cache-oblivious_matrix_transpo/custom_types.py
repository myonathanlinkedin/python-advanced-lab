from __future__ import annotations

from typing import Generic, List, TypeVar

T = TypeVar('T')

# A matrix is a list of rows, each row is a list of elements of type T.
Matrix = List[List[T]]
