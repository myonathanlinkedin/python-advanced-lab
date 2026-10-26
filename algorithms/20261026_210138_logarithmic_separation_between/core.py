"""
Core algorithms for LZ77, LZ-End, and Re-Pair grammar compression.
All implementations are deterministic, pure‑Python and use only the standard library.
"""

from __future__ import annotations
from collections import Counter
from typing import List, Tuple, Dict


# ----------------------------------------------------------------------
# LZ77 factorization (naïve O(n^2) implementation)
# Each factor is a triple (distance, length, next_char).
# distance = 0 and length = 0 for a literal character.
# ----------------------------------------------------------------------
def lz77_factorization(text: str) -> List[Tuple[int, int, str]]:
    n = len(text)
    i = 0
    factors: List[Tuple[int, int, str]] = []
    while i < n:
        best_len = 0
        best_dist = 0
        # search all previous positions for longest match
        for j in range(i):
            l = 0
            while i + l < n and text[j + l] == text[i + l]:
                l += 1
                if j + l >= i:  # cannot cross current position
                    break
            if l > best_len:
                best_len = l
                best_dist = i - j
        if best_len == 0:
            # literal
            factors.append((0, 0, text[i]))
            i += 1
        else:
            # copy + following character (if any)
            next_char = text[i + best_len] if i + best_len < n else ""
            factors.append((best_dist, best_len, next_char))
            i += best_len + (1 if next_char else 0)
    return factors


def lz77_reconstruct(factors: List[Tuple[int, int, str]]) -> str:
    out: List[str] = []
    for dist, length, nxt in factors:
        if length == 0:
            out.append(nxt)
        else:
            start = len(out) - dist
            for k in range(length):
                out.append(out[start + k])
            if nxt:
                out.append(nxt)
    return "".join(out)


# ----------------------------------------------------------------------
# LZ-End factorization (naïve O(n^3) implementation)
# Each factor is a triple (distance, length, next_char) similar to LZ77,
# but the source substring must end at a previous phrase boundary.
# ----------------------------------------------------------------------
def lz_end_factorization(text: str) -> List[Tuple[int, int, str]]:
    n = len(text)
    i = 0
    factors: List[Tuple[int, int, str]] = []
    phrase_ends: List[int] = []          # positions (inclusive) where a phrase ends
    while i < n:
        max_len = 0
        best_dist = 0
        # try all previous phrase ends as possible source terminators
        for end in phrase_ends:
            max_possible = i - (end - (i - 1))  # not needed, we just bound by remaining length
            # try decreasing lengths
            max_l = min(end + 1, n - i)       # source cannot be longer than text left
            for l in range(max_l, 0, -1):
                src_start = end - l + 1
                if src_start < 0:
                    continue
                if text[src_start:src_start + l] == text[i:i + l]:
                    if l > max_len:
                        max_len = l
                        best_dist = i - src_start
                    break  # longer lengths already checked
        if max_len == 0:
            # literal
            factors.append((0, 0, text[i]))
            i += 1
            phrase_ends.append(i - 1)
        else:
            nxt = text[i + max_len] if i + max_len < n else ""
            factors.append((best_dist, max_len, nxt))
            i += max_len + (1 if nxt else 0)
            phrase_ends.append(i - 1)
    return factors


def lz_end_reconstruct(factors: List[Tuple[int, int, str]]) -> str:
    out: List[str] = []
    for dist, length, nxt in factors:
        if length == 0:
            out.append(nxt)
        else:
            start = len(out) - dist
            for k in range(length):
                out.append(out[start + k])
            if nxt:
                out.append(nxt)
    return "".join(out)


# ----------------------------------------------------------------------
# Re-Pair grammar compression (naïve implementation)
# Returns a list of production rules and the final compressed string.
# Non‑terminals are named X1, X2, …
# ----------------------------------------------------------------------
def repar_factorization(text: str) -> Tuple[Dict[str, str], str]:
    """
    Perform Re-Pair on `text`.
    Returns (rules, compressed_text) where `rules` maps non‑terminal to its expansion.
    """
    # work on a list of symbols (characters or non‑terminals)
    symbols: List[str] = list(text)
    rules: Dict[str, str] = {}
    next_id = 1

    while True:
        # count all adjacent pairs
        pairs = Counter()
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += 1
        # find most frequent pair occurring at least twice
        most_common = pairs.most_common(1)
        if not most_common or most_common[0][1] < 2:
            break
        pair, _ = most_common[0]
        a, b = pair
        new_nt = f"X{next_id}"
        next_id += 1
        rules[new_nt] = a + b

        # replace all non‑overlapping occurrences of the pair
        i = 0
        new_symbols: List[str] = []
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                new_symbols.append(new_nt)
                i += 2
            else:
                new_symbols.append(symbols[i])
                i += 1
        symbols = new_symbols

    compressed = "".join(symbols)
    return rules, compressed


def repar_expand(rules: Dict[str, str], compressed: str) -> str:
    """
    Fully expand a Re-Pair grammar.
    """
    # iterative expansion until no non‑terminal remains
    changed = True
    result = compressed
    while changed:
        changed = False
        for nt, rhs in rules.items():
            if nt in result:
                result = result.replace(nt, rhs)
                changed = True
    return result


# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def factor_count(factors: List[Tuple[int, int, str]]) -> int:
    """Number of phrases (factors) in a factorization."""
    return len(factors)


def grammar_size(rules: Dict[str, str], compressed: str) -> int:
    """Total number of symbols (right‑hand side length + compressed length)."""
    rhs_len = sum(len(v) for v in rules.values())
    return rhs_len + len(compressed)


__all__ = [
    "lz77_factorization",
    "lz77_reconstruct",
    "lz_end_factorization",
    "lz_end_reconstruct",
    "repar_factorization",
    "repar_expand",
    "factor_count",
    "grammar_size",
]
