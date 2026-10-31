import unittest
from typing import List

from core import LamportClockEngine, Message


class TestLamportClockEngine(unittest.TestCase):
    def setUp(self) -> None:
        # Create an engine with three processes: 0, 1, 2
        self.engine = LamportClockEngine(process_ids=[0, 1, 2])

    def test_initial_state(self) -> None:
        for pid in (0, 1, 2):
            self.assertEqual(self.engine.get_timestamp(pid), 0)

    def test_internal_event(self) -> None:
        ts = self.engine.internal_event(0)
        self.assertEqual(ts, 1)
        self.assertEqual(self.engine.get_timestamp(0), 1)

        # Multiple internal events accumulate
        for i in range(5):
            ts = self.engine.internal_event(0)
        self.assertEqual(self.engine.get_timestamp(0), 7)  # 1 + 5 + 1

    def test_send_receive_sequence(self) -> None:
        # Process 0 sends to 1
        msg = self.engine.send_event(0, 1)
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.sender, 0)
        self.assertEqual(msg.receiver, 1)
        self.assertEqual(msg.timestamp, 1)
        self.assertEqual(self.engine.get_timestamp(0), 1)

        # Process 1 receives the message
        recv_ts = self.engine.receive_event(1, msg)
        self.assertEqual(recv_ts, 2)
        self.assertEqual(self.engine.get_timestamp(1), 2)

        # Process 1 sends back to 0
        reply = self.engine.send_event(1, 0)
        self.assertEqual(reply.timestamp, 3)  # 2 + 1
        self.assertEqual(self.engine.get_timestamp(1), 3)

        # Process 0 receives reply
        recv_ts2 = self.engine.receive_event(0, reply)
        self.assertEqual(recv_ts2, 4)  # max(1,3)+1
        self.assertEqual(self.engine.get_timestamp(0), 4)

    def test_concurrent_events(self) -> None:
        # Simulate interleaved internal events before communication
        self.engine.internal_event(0)  # ts 1
        self.engine.internal_event(1)  # ts 1
        self.engine.internal_event(0)  # ts 2

        msg = self.engine.send_event(0, 2)  # ts 3
        self.assertEqual(msg.timestamp, 3)

        # Process 2 had no prior events
        recv_ts = self.engine.receive_event(2, msg)
        self.assertEqual(recv_ts, 4)  # max(0,3)+1
        self.assertEqual(self.engine.get_timestamp(2), 4)

    def test_error_handling_unknown_process(self) -> None:
        with self.assertRaises(ValueError):
            self.engine.internal_event(99)

        with self.assertRaises(ValueError):
            self.engine.send_event(0, 99)

        with self.assertRaises(ValueError):
            self.engine.receive_event(99, Message(0, 99, 1))

    def test_error_handling_mismatched_receiver(self) -> None:
        msg = self.engine.send_event(0, 1)
        with self.assertRaises(ValueError):
            self.engine.receive_event(2, msg)  # msg intended for 1

    def test_multiple_processes_consistency(self) -> None:
        # Chain of messages 0->1->2->0
        m01 = self.engine.send_event(0, 1)  # ts 1
        r1 = self.engine.receive_event(1, m01)  # ts 2
        m12 = self.engine.send_event(1, 2)  # ts 3
        r2 = self.engine.receive_event(2, m12)  # ts 4
        m20 = self.engine.send_event(2, 0)  # ts 5
        r0 = self.engine.receive_event(0, m20)  # ts 6

        self.assertEqual(self.engine.get_timestamp(0), 6)
        self.assertEqual(self.engine.get_timestamp(1), 3)
        self.assertEqual(self.engine.get_timestamp(2), 5)

    def test_timestamp_monotonicity(self) -> None:
        # Ensure timestamps never decrease for any process
        timestamps: List[int] = []
        for _ in range(10):
            ts = self.engine.internal_event(0)
            timestamps.append(ts)
        for earlier, later in zip(timestamps, timestamps[1:]):
            self.assertLessEqual(earlier, later)

    def test_repr(self) -> None:
        rep = repr(self.engine)
        self.assertIn("LamportClockEngine", rep)
        self.assertIn("{0: 0", rep)  # initial counters


if __name__ == "__main__":
    unittest.main()
