from __future__ import annotations

from typing import List

from engine import SuffixAutomaton


def _run_basic_tests() -> None:
    # Test 1: simple string
    s = "ababa"
    sam = SuffixAutomaton(s)

    # Substring existence
    assert sam.contains("a")
    assert sam.contains("ab")
    assert sam.contains("aba")
    assert sam.contains("baba")
    assert sam.contains("")
    assert not sam.contains("c")
    assert not sam.contains("abb")

    # Occurrence counting
    assert sam.count_occurrences("a") == 3
    assert sam.count_occurrences("b") == 2
    assert sam.count_occurrences("ab") == 2
    assert sam.count_occurrences("aba") == 2
    assert sam.count_occurrences("bab") == 1
    assert sam.count_occurrences("") == len(s) + 1  # empty string occurs n+1 times

    # Test 2: repeated single character
    s2 = "aaaaa"
    sam2 = SuffixAutomaton(s2)
    for length in range(1, len(s2) + 1):
        sub = "a" * length
        assert sam2.contains(sub)
        # Number of occurrences of "a"*k in "a"*n is n - k + 1
        assert sam2.count_occurrences(sub) == len(s2) - length + 1
    assert not sam2.contains("aaab")
    assert sam2.count_occurrences("aaab") == 0

    # Test 3: empty source string
    empty_sam = SuffixAutomaton("")
    assert empty_sam.contains("")
    assert not empty_sam.contains("a")
    assert empty_sam.count_occurrences("") == 1  # only the empty suffix
    assert empty_sam.count_occurrences("a") == 0

    # Test 4: all distinct characters
    s3 = "abcdefghijklmnopqrstuvwxyz"
    sam3 = SuffixAutomaton(s3)
    assert sam3.contains("abc")
    assert sam3.contains("xyz")
    assert not sam3.contains("abca")
    assert sam3.count_occurrences("a") == 1
    assert sam3.count_occurrences("z") == 1
    assert sam3.count_occurrences("ab") == 1

    # Test 5: long random-like string (deterministic)
    s4 = "thequickbrownfoxjumpsoverthelazydog"
    sam4 = SuffixAutomaton(s4)
    queries: List[str] = ["quick", "brown", "fox", "lazy", "dog", "the", "thequick", "over", "jumps", "cat"]
    expected: List[bool] = [True, True, True, True, True, True, True, True, True, False]
    for q, exp in zip(queries, expected):
        assert sam4.contains(q) == exp
        # occurrence count should be 1 for present substrings, 0 otherwise
        assert sam4.count_occurrences(q) == (1 if exp else 0)

    # Test 6: verify state count invariant (2 * |s| - 1 upper bound)
    for test_str in ["", "a", "ab", "abcab", "aaaa", "ababa", s4]:
        sam = SuffixAutomaton(test_str)
        assert len(sam) <= 2 * len(test_str) - 1 if test_str else len(sam) == 1


def _demo() -> None:
    """
    Demonstration of interactive usage.
    """
    text = "abracadabra"
    sam = SuffixAutomaton(text)
    print(f"Indexed text: {text!r}")
    for query in ["abra", "cad", "dab", "ra", ""]:
        print(f"contains({query!r}) -> {sam.contains(query)}; occurrences -> {sam.count_occurrences(query)}")


if __name__ == "__main__":
    _run_basic_tests()
    _demo()
