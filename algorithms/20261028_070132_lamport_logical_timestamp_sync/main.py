import unittest
from typing import List

from core import LamportEngine, Process, Message


class TestLamportEngine(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = LamportEngine()
        for pid in ("A", "B", "C"):
            self.engine.add_process(pid)

    def test_internal_events(self) -> None:
        proc_a = self.engine.get_process("A")
        ts1 = proc_a.internal_event()
        ts2 = proc_a.internal_event()
        self.assertEqual(ts1, 1)
        self.assertEqual(ts2, 2)
        self.assertEqual(proc_a.clock.value(), 2)

    def test_send_receive_order(self) -> None:
        msg = self.engine.send_message("A", "B")
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.sender_id, "A")
        self.assertEqual(msg.receiver_id, "B")
        # After delivery, B's clock must be > msg.timestamp
        proc_b = self.engine.get_process("B")
        self.assertGreater(proc_b.clock.value(), msg.timestamp)
        # Verify causality invariant
        self.assertTrue(self.engine.verify_causality())

    def test_multiple_messages(self) -> None:
        # A sends two messages to B
        m1 = self.engine.send_message("A", "B")
        m2 = self.engine.send_message("A", "B")
        self.assertLess(m1.timestamp, m2.timestamp)
        proc_b = self.engine.get_process("B")
        self.assertGreater(proc_b.clock.value(), m2.timestamp)
        self.assertTrue(self.engine.verify_causality())

    def test_cross_communication(self) -> None:
        # A -> B, B -> C, C -> A forming a cycle
        m_ab = self.engine.send_message("A", "B")
        m_bc = self.engine.send_message("B", "C")
        m_ca = self.engine.send_message("C", "A")
        # All timestamps must be strictly increasing along the cycle
        self.assertLess(m_ab.timestamp, m_bc.timestamp)
        self.assertLess(m_bc.timestamp, m_ca.timestamp)
        self.assertTrue(self.engine.verify_causality())

    def test_receive_with_older_timestamp(self) -> None:
        # Manually craft a message with older timestamp
        proc_a = self.engine.get_process("A")
        proc_b = self.engine.get_process("B")
        # A performs two internal events, reaching ts=2
        proc_a.internal_event()
        proc_a.internal_event()
        # B is still at ts=0, we send a message with ts=1 (older than B's future ts)
        old_msg = Message(sender_id="A", receiver_id="B", timestamp=1)
        self.engine.deliver_message(old_msg)
        # B's clock should become max(0,1)+1 = 2
        self.assertEqual(proc_b.clock.value(), 2)
        self.assertTrue(self.engine.verify_causality())

    def test_invalid_receiver(self) -> None:
        proc_a = self.engine.get_process("A")
        msg = proc_a.send("B")
        # Tamper receiver id
        bad_msg = Message(sender_id=msg.sender_id, receiver_id="X", timestamp=msg.timestamp)
        with self.assertRaises(ValueError):
            self.engine.deliver_message(bad_msg)

    def test_snapshot(self) -> None:
        self.engine.send_message("A", "B")
        self.engine.send_message("B", "C")
        snap = self.engine.snapshot()
        self.assertIsInstance(snap, dict)
        self.assertSetEqual(set(snap.keys()), {"A", "B", "C"})
        # All clocks must be >=1 after communication
        for v in snap.values():
            self.assertGreaterEqual(v, 1)


def run_demo() -> None:
    """Simple demonstration of LamportEngine usage."""
    eng = LamportEngine()
    for pid in ("P1", "P2"):
        eng.add_process(pid)

    print("Initial snapshot:", eng.snapshot())
    eng.send_message("P1", "P2")
    eng.send_message("P2", "P1")
    eng.get_process("P1").internal_event()
    eng.get_process("P2").internal_event()
    print("Final snapshot:", eng.snapshot())
    print("Causality holds:", eng.verify_causality())
    for pid, proc in eng.processes.items():
        print(f"Log for {pid}:")
        for entry in proc.get_log():
            print("  ", entry)


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo Run ---")
    run_demo()
