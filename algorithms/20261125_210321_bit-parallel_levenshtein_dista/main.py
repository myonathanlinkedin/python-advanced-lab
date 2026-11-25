from __future__ import annotations
import random
import string
from typing import List

from .engine import BitParallelLevenshtein


def _levenshtein_dp(a: str, b: str) -> int:
    """
    Classic dynamic‑programming reference implementation.
    Used only for verification in unit tests.
    """
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m]


def _random_string(length: int) -> str:
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))


def run_unit_tests() -> None:
    # Edge cases
    assert BitParallelLevenshtein("").distance("") == 0
    assert BitParallelLevenshtein("").distance("abc") == 3
    assert BitParallelLevenshtein("abc").distance("") == 3

    # Identical strings
    for s in ["a", "abc", "hello", ""]:
        assert BitParallelLevenshtein(s).distance(s) == 0

    # Small exhaustive tests (length <= 6)
    alphabet = "ab"
    for n in range(0, 5):
        for m in range(0, 5):
            for a in (''.join(p) for p in __import__('itertools').product(alphabet, repeat=n)):
                for b in (''.join(p) for p in __import__('itertools').product(alphabet, repeat=m)):
                    expected = _levenshtein_dp(a, b)
                    got = BitParallelLevenshtein(a).distance(b)
                    assert got == expected, f"Mismatch for a={a!r}, b={b!r}: {got} != {expected}"

    # Randomised tests up to length 30 (covers > word size)
    random.seed(0)
    for _ in range(200):
        a_len = random.randint(0, 30)
        b_len = random.randint(0, 30)
        a = _random_string(a_len)
        b = _random_string(b_len)
        expected = _levenshtein_dp(a, b)
        got = BitParallelLevenshtein(a).distance(b)
        assert got == expected, f"Random test failed for a={a!r}, b={b!r}: {got} != {expected}"

    print("All unit tests passed.")


def demo() -> None:
    """
    Simple interactive demonstration.
    """
    examples = [
        ("kitten", "sitting"),
        ("flaw", "lawn"),
        ("intention", "execution"),
        ("", "nonempty"),
        ("short", ""),
    ]
    for pat, txt in examples:
        dist = BitParallelLevenshtein(pat).distance(txt)
        print(f"Levenshtein('{pat}', '{txt}') = {dist}")


if __name__ == "__main__":
    run_unit_tests()
    demo()
