import unittest
from typing import List

# Import core components
from core import LamportEngine, Process, Message


class TestLamportClock(unittest.TestCase):
    def test_initial_state(self):
        proc = Process("P1")
        self.assertEqual(proc.clock.time, 0)
        self.assertEqual(proc.log, [])

    def test_internal_event(self):
        proc = Process("P1")
        ts1 = proc.internal_event()
        self.assertEqual(ts1, 1)
        ts2 = proc.internal_event()
        self.assertEqual(ts2, 2)
        self.assertEqual(proc.log, [("internal", 1), ("internal", 2)])

    def test_send_and_receive(self):
        engine = LamportEngine()
        p1 = engine.add_process("A")
        p2 = engine.add_process("B")

        # A sends to B
        msg = engine.send_message("A", "B")
        self.assertIsInstance(msg, Message)
        self.assertEqual(msg.sender_id, "A")
        self.assertEqual(msg.timestamp, 1)  # A ticked once for send

        # Verify logs
        self.assertEqual(p1.log, [("send", 1)])
        self.assertEqual(p2.log, [("receive", 2)])  # B updates to max(0,1)+1

        # B sends back to A
        msg2 = engine.send_message("B", "A")
        self.assertEqual(msg2.timestamp, 3)  # B ticked from 2 to 3
        self.assertEqual(p2.log, [("receive", 2), ("send", 3)])
        self.assertEqual(p1.log, [("send", 1), ("receive", 4)])  # A updates to max(1,3)+1

    def test_receive_older_timestamp(self):
        engine = LamportEngine()
        p1 = engine.add_process("X")
        p2 = engine.add_process("Y")

        # X performs two internal events: clock = 2
        p1.internal_event()
        p1.internal_event()
        self.assertEqual(p1.clock.time, 2)

        # Y sends a message (timestamp = 1)
        msg = engine.send_message("Y", "X")
        self.assertEqual(msg.timestamp, 1)

        # X receives older timestamp; should become 3 (max(2,1)+1)
        self.assertEqual(p1.clock.time, 3)
        self.assertEqual(p1.log[-1], ("receive", 3))

    def test_broadcast(self):
        engine = LamportEngine()
        for pid in ("A", "B", "C"):
            engine.add_process(pid)

        msgs = engine.broadcast("A", payload="hello")
        self.assertEqual(len(msgs), 2)  # sent to B and C
        timestamps = [m.timestamp for m in msgs]
        # A's clock should have ticked once for each send (total 2)
        self.assertEqual(engine.get_process("A").clock.time, 2)
        # Receivers should have timestamps 2 (max(0,1)+1) and 2 (max(0,1)+1)
        for pid in ("B", "C"):
            self.assertEqual(engine.get_process(pid).clock.time, 2)

    def test_error_conditions(self):
        engine = LamportEngine()
        engine.add_process("P")
        with self.assertRaises(KeyError):
            engine.add_process("P")  # duplicate
        with self.assertRaises(KeyError):
            engine.send_message("P", "Q")  # receiver missing
        with self.assertRaises(KeyError):
            engine.send_message("Q", "P")  # sender missing


def run_demo() -> None:
    """
    Simple demonstration of Lamport timestamp synchronization.
    """
    engine = LamportEngine()
    p1 = engine.add_process("Node1")
    p2 = engine.add_process("Node2")
    p3 = engine.add_process("Node3")

    # Sequence of events
    p1.internal_event()          # t=1
    engine.send_message("Node1", "Node2")  # Node1 t=2, Node2 t=3
    p2.internal_event()          # Node2 t=4
    engine.broadcast("Node2")    # Node2 t=6, Node1 t=7, Node3 t=7
    p3.internal_event()          # Node3 t=8

    # Print final logical times
    for pid in sorted(engine.processes):
        proc = engine.get_process(pid)
        print(f"{pid}: logical time = {proc.clock.time}, log = {proc.log}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo Run ---")
    run_demo()
