"""
Domain models and memory structures for the Cooley-Tukey Radix-2 FFT.

This module defines the core data structures used throughout the FFT
implementation:
  - ComplexSample: A single complex-valued signal sample.
  - Signal: An immutable sequence of ComplexSample values.
  - FFTResult: The output of an FFT computation, including the
    transformed spectrum and metadata.

All types are pure data containers with no algorithmic logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass(frozen=True)
class ComplexSample:
    """
    A single complex-valued sample: z = real + imag * i.

    Invariant: real and imag are finite floats (no NaN/Inf).
    """
    real: float
    imag: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.real):
            raise ValueError(f"real part must be finite, got {self.real}")
        if not math.isfinite(self.imag):
            raise ValueError(f"imag part must be finite, got {self.imag}")

    def __add__(self, other: "ComplexSample") -> "ComplexSample":
        return ComplexSample(self.real + other.real, self.imag + other.imag)

    def __sub__(self, other: "ComplexSample") -> "ComplexSample":
        return ComplexSample(self.real - other.real, self.imag - other.imag)

    def __mul__(self, other: "ComplexSample") -> "ComplexSample":
        # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        return ComplexSample(
            self.real * other.real - self.imag * other.imag,
            self.real * other.imag + self.imag * other.real,
        )

    def __neg__(self) -> "ComplexSample":
        return ComplexSample(-self.real, -self.imag)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComplexSample):
            return NotImplemented
        return (
            math.isclose(self.real, other.real, abs_tol=1e-12)
            and math.isclose(self.imag, other.imag, abs_tol=1e-12)
        )

    def __repr__(self) -> str:
        if self.imag == 0.0:
            return f"ComplexSample({self.real:.6f})"
        if self.real == 0.0:
            return f"ComplexSample({self.imag:.6f}j)"
        sign = "+" if self.imag >= 0 else "-"
        return f"ComplexSample({self.real:.6f}{sign}{abs(self.imag):.6f}j)"

    @property
    def magnitude(self) -> float:
        """|z| = sqrt(real^2 + imag^2)."""
        return math.hypot(self.real, self.imag)

    @property
    def phase(self) -> float:
        """arg(z) in [-pi, pi]."""
        return math.atan2(self.imag, self.real)


@dataclass(frozen=True)
class Signal:
    """
    An immutable sequence of complex samples representing a discrete signal.

    Invariant: length >= 0. For FFT use, length must be a power of 2.
    """
    samples: Tuple[ComplexSample, ...]

    def __post_init__(self) -> None:
        # Validate all samples are finite
        for i, s in enumerate(self.samples):
            if not math.isfinite(s.real) or not math.isfinite(s.imag):
                raise ValueError(f"Sample at index {i} contains non-finite values")

    @property
    def length(self) -> int:
        return len(self.samples)

    @property
    def is_power_of_two(self) -> bool:
        """True if length is a power of 2 (required for radix-2 FFT)."""
        n = self.length
        if n == 0:
            return False
        return (n & (n - 1)) == 0

    def __getitem__(self, index: int) -> ComplexSample:
        return self.samples[index]

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        return iter(self.samples)

    def __repr__(self) -> str:
        preview = ", ".join(repr(s) for s in self.samples[:4])
        suffix = ", ..." if self.length > 4 else ""
        return f"Signal(n={self.length}, [{preview}{suffix}])"


@dataclass(frozen=True)
class FFTResult:
    """
    Result of a forward or inverse FFT computation.

    Attributes:
        spectrum: The transformed complex samples (frequency domain).
        is_forward: True if this is a forward FFT, False if inverse.
        n: The transform length (power of 2).
        log2_n: log2(n), the number of butterfly stages.
    """
    spectrum: Tuple[ComplexSample, ...]
    is_forward: bool
    n: int
    log2_n: int

    def __post_init__(self) -> None:
        if self.n != len(self.spectrum):
            raise ValueError(
                f"spectrum length {len(self.spectrum)} != declared n={self.n}"
            )
        if self.n <= 0 or (self.n & (self.n - 1)) != 0:
            raise ValueError(f"n must be a positive power of 2, got {self.n}")
        expected_log2 = int(math.log2(self.n))
        if self.log2_n != expected_log2:
            raise ValueError(
                f"log2_n={self.log2_n} inconsistent with n={self.n}"
            )

    def __getitem__(self, index: int) -> ComplexSample:
        return self.spectrum[index]

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        direction = "forward" if self.is_forward else "inverse"
        preview = ", ".join(repr(s) for s in self.spectrum[:4])
        suffix = ", ..." if self.n > 4 else ""
        return f"FFTResult({direction}, n={self.n}, [{preview}{suffix}])"


def is_power_of_two(n: int) -> bool:
    """
    Check if n is a positive power of 2.

    Time: O(1)
    Space: O(1)
    """
    if n <= 0:
        return False
    return (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """
    Return the smallest power of 2 >= n.

    Time: O(1)
    Space: O(1)
    """
    if n <= 1:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1
