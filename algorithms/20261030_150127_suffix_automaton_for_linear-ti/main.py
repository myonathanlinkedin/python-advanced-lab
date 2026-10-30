from __future__ import annotations

from typing import List

from .engine import SuffixAutomaton


def _test_basic_operations() -> None:
    # Simple known example
    s = "ababa"
    sam = SuffixAutomaton(s)

    # Substring existence checks
    assert sam.contains("a")
    assert sam.contains("b")
    assert sam.contains("ab")
    assert sam.contains("ba")
    assert sam.contains("aba")
    assert sam.contains("bab")
    assert sam.contains("abab")
    assert sam.contains("baba")
    assert sam.contains("ababa")
    assert not sam.contains("c")
    assert not sam.contains("aa")
    assert not sam.contains("abb")

    # Distinct substring count: known to be 9 for "ababa"
    assert sam.count_distinct_substrings() == 9

def _test_edge_cases() -> None:
    # Empty string
    empty = SuffixAutomaton("")
    assert empty.count_distinct_substrings() == 0
    assert not empty.contains("a")

    # Single character
    single = SuffixAutomaton("x")
    assert single.count_distinct_substrings() == 1
    assert single.contains("x")
    assert not single.contains("xx")
    assert not single.contains("")

    # Repeated character
    rep = SuffixAutomaton("aaaa")
    # Distinct substrings are "a", "aa", "aaa", "aaaa" => 4
    assert rep.count_distinct_substrings() == 4
    for length in range(1, 5):
        assert rep.contains("a" * length)
    assert not rep.contains("a" * 5)

def _test_random_consistency() -> None:
    import random
    import string

    for _ in range(20):
        length = random.randint(0, 30)
        txt = "".join(random.choice(string.ascii_lowercase) for _ in range(length))
        sam = SuffixAutomaton(txt)

        # Exhaustively verify substring existence for all substrings (O(n^2) but n ≤ 30)
        substrings = {txt[i:j] for i in range(len(txt)) for j in range(i + 1, len(txt) + 1)}
        for sub in substrings:
            assert sam.contains(sub), f"Missing substring {sub!r} in {txt!r}"
        # Also verify that no non‑existent substring is reported true
        for _ in range(10):
            fake_len = random.randint(1, 5)
            fake = "".join(random.choice(string.ascii_lowercase) for _ in range(fake_len))
            if fake not in substrings:
                assert not sam.contains(fake), f"False positive for {fake!r} in {txt!r}"

        # Distinct substring count must equal size of the set
        assert sam.count_distinct_substrings() == len(substrings)

def _run_all_tests() -> None:
    _test_basic_operations()
    _test_edge_cases()
    _test_random_consistency()
    print("All tests passed.")


if __name__ == "__main__":
    _run_all_tests()
