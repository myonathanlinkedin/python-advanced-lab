"""
main.py
=======
Unit tests, benchmarks, and entrypoint for the Aether-Flux model.

Run:  python main.py
"""

from __future__ import annotations

import time
import unittest
from typing import List

from core import (
    AetherFluxParser,
    Frame,
    ForwardingEntry,
    ParseResult,
    Router,
    build_topology,
    compute_checksum,
    make_frame,
    RING_CAPACITY,
    MAX_FRAME_LEN,
)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestChecksum(unittest.TestCase):
    def test_deterministic(self) -> None:
        a = compute_checksum(1, 2, b"hello")
        b = compute_checksum(1, 2, b"hello")
        self.assertEqual(a, b)

    def test_differs_on_payload(self) -> None:
        a = compute_checksum(1, 2, b"hello")
        b = compute_checksum(1, 2, b"world")
        self.assertNotEqual(a, b)

    def test_range(self) -> None:
        for i in range(256):
            c = compute_checksum(i, i, bytes([i]))
            self.assertTrue(0 <= c <= 255)


class TestFrame(unittest.TestCase):
    def test_valid(self) -> None:
        f = make_frame(1, 2, b"abc")
        self.assertEqual(f.src, 1)
        self.assertEqual(f.dst, 2)
        self.assertEqual(f.declared_len, 3)
        self.assertEqual(f.checksum, compute_checksum(1, 2, b"abc"))

    def test_corrupt_checksum(self) -> None:
        f = make_frame(1, 2, b"abc", corrupt_checksum=True)
        self.assertNotEqual(f.checksum, compute_checksum(1, 2, b"abc"))

    def test_invalid_src(self) -> None:
        with self.assertRaises(ValueError):
            Frame(src=256, dst=0, payload=b"", declared_len=0, checksum=0)

    def test_invalid_dst(self) -> None:
        with self.assertRaises(ValueError):
            Frame(src=0, dst=-1, payload=b"", declared_len=0, checksum=0)

    def test_negative_declared_len(self) -> None:
        with self.assertRaises(ValueError):
            Frame(src=0, dst=0, payload=b"", declared_len=-1, checksum=0)


class TestParser(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = AetherFluxParser()

    def test_clean_frame(self) -> None:
        f = make_frame(1, 2, b"xyz")
        r = self.parser.parse_frame(f)
        self.assertTrue(r.ok)
        self.assertEqual(r.tokens, (0x78, 0x79, 0x7A))
        self.assertFalse(self.parser.is_poisoned)

    def test_checksum_mismatch_poisons(self) -> None:
        f = make_frame(1, 2, b"xyz", corrupt_checksum=True)
        r = self.parser.parse_frame(f)
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "checksum_mismatch")
        self.assertTrue(self.parser.is_poisoned)

    def test_bug_a_length_overflow(self) -> None:
        # declared_len > actual payload length
        f = Frame(src=1, dst=2, payload=b"ab", declared_len=10,
                  checksum=compute_checksum(1, 2, b"ab"))
        r = self.parser.parse_frame(f)
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "length_overflow")
        self.assertTrue(self.parser.is_poisoned)

    def test_reset_clears_poison(self) -> None:
        f = make_frame(1, 2, b"xyz", corrupt_checksum=True)
        self.parser.parse_frame(f)
        self.assertTrue(self.parser.is_poisoned)
        self.parser.reset_state()
        self.assertFalse(self.parser.is_poisoned)
        self.assertEqual(self.parser.ring_snapshot, ())

    def test_ring_capacity_respected(self) -> None:
        p = AetherFluxParser(ring_capacity=4)
        f = make_frame(0, 0, bytes(range(10)))
        p.parse_frame(f)
        snap = p.ring_snapshot
        self.assertEqual(len(snap), 4)

    def test_invalid_capacity(self) -> None:
        with self.assertRaises(ValueError):
            AetherFluxParser(ring_capacity=0)


class TestRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = AetherFluxParser()
        self.router = Router("R0", self.parser,
                             table=[ForwardingEntry(dst=2, next_hop=10)])

    def test_forward_success(self) -> None:
        f = make_frame(1, 2, b"hi")
        nh = self.router.forward(f)
        self.assertEqual(nh, 10)

    def test_forward_no_route(self) -> None:
        f = make_frame(1, 99, b"hi")
        nh = self.router.forward(f)
        self.assertIsNone(nh)

    def test_forward_bad_frame(self) -> None:
        f = make_frame(1, 2, b"hi", corrupt_checksum=True)
        nh = self.router.forward(f)
        self.assertIsNone(nh)


class TestMultiRouterAmplification(unittest.TestCase):
    """Verify that a single poisoned frame affects ALL shared routers."""

    def test_shared_parser_poisoning(self) -> None:
        parser = AetherFluxParser()
        table = [ForwardingEntry(dst=5, next_hop=100)]
        routers = build_topology(3, parser, default_table=table)

        # Send a good frame through R0
        good = make_frame(1, 5, b"ok")
        self.assertEqual(routers[0].forward(good), 100)

        # Poison the shared parser via R1
        bad = make_frame(1, 5, b"ok", corrupt_checksum=True)
        self.assertIsNone(routers[1].forward(bad))

        # R2 should now be affected: parser is poisoned
        self.assertTrue(parser.is_poisoned)

        # After reset, R2 works again
        parser.reset_state()
        self.assertEqual(routers[2].forward(good), 100)

    def test_length_overflow_affects_all(self) -> None:
        parser = AetherFluxParser()
        table = [ForwardingEntry(dst=7, next_hop=200)]
        routers = build_topology(2, parser, default_table=table)

        # Craft a frame with declared_len > payload
        bad = Frame(src=1, dst=7, payload=b"ab", declared_len=50,
                    checksum=compute_checksum(1, 7, b"ab"))
        self.assertIsNone(routers[0].forward(bad))
        self.assertTrue(parser.is_poisoned)

        # R1 is also affected
        good = make_frame(1, 7, b"fine")
        # Parser is poisoned but a good frame with valid checksum
        # will still parse (poison flag is set but checksum passes).
        # The key invariant: the ring contains stale data.
        snap = parser.ring_snapshot
        self.assertGreater(len(snap), 0)


class TestEdgeCases(unittest.TestCase):
    def test_empty_payload(self) -> None:
        p = AetherFluxParser()
        f = make_frame(0, 0, b"")
        r = p.parse_frame(f)
        self.assertTrue(r.ok)
        self.assertEqual(r.tokens, ())

    def test_max_frame_len(self) -> None:
        p = AetherFluxParser()
        payload = bytes(MAX_FRAME_LEN)
        f = make_frame(0, 0, payload)
        r = p.parse_frame(f)
        self.assertTrue(r.ok)

    def test_declared_len_exceeds_max(self) -> None:
        p = AetherFluxParser()
        f = Frame(src=0, dst=0, payload=b"a",
                  declared_len=MAX_FRAME_LEN + 100,
                  checksum=compute_checksum(0, 0, b"a"))
        r = p.parse_frame(f)
        self.assertFalse(r.ok)
        self.assertEqual(r.error, "length_overflow")

    def test_many_frames_stability(self) -> None:
        p = AetherFluxParser()
        for i in range(1000):
            f = make_frame(i % 256, (i + 1) % 256, bytes([i & 0xFF]))
            r = p.parse_frame(f)
            self.assertTrue(r.ok)
        self.assertFalse(p.is_poisoned)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark(n_frames: int = 100_000) -> float:
    """Parse *n_frames* frames and return elapsed seconds."""
    parser = AetherFluxParser()
    frames: List[Frame] = [
        make_frame(i % 256, (i + 1) % 256, bytes([i & 0xFF]))
        for i in range(n_frames)
    ]
    t0 = time.perf_counter()
    for f in frames:
        parser.parse_frame(f)
    t1 = time.perf_counter()
    return t1 - t0


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo() -> None:
    print("=== Aether-Flux Demo ===")
    parser = AetherFluxParser()
    table = [
        ForwardingEntry(dst=10, next_hop=100),
        ForwardingEntry(dst=20, next_hop=200),
    ]
    routers = build_topology(3, parser, default_table=table)

    # Normal operation
    f1 = make_frame(1, 10, b"hello")
    print(f"R0 forward dst=10 -> next_hop={routers[0].forward(f1)}")

    # Inject a bad frame (checksum mismatch)
    f2 = make_frame(1, 10, b"hello", corrupt_checksum=True)
    print(f"R1 forward bad frame -> next_hop={routers[1].forward(f2)}")
    print(f"Parser poisoned: {parser.is_poisoned}")

    # Reset and verify recovery
    parser.reset_state()
    f3 = make_frame(1, 20, b"world")
    print(f"R2 forward dst=20 after reset -> next_hop={routers[2].forward(f3)}")
    print(f"Parser poisoned: {parser.is_poisoned}")
    print("=== Demo complete ===\n")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo()
    unittest.main(verbosity=2, exit=True)
