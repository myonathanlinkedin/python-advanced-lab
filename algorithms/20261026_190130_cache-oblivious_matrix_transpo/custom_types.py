from __future__ import annotations
from typing import List, TypeVar, Generic

T = TypeVar('T')

# A simple matrix type alias using a list of rows.
Matrix = List[List[T]]
