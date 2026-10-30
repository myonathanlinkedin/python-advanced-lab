import unittest
import time
from typing import List

from core import Process, Timestamp, Event, merge_logs


class TestLamportEngine(unittest.TestCase):
    def test_local_event_monotonic(self) -> None:
        p = Process(pid=1)
        ts1 = p.local_event("e1")
        ts2 = p.local_event("e2")
        ts3 = p.local_event("e3")
        self.assertTrue(ts1 < ts2 < ts3)
        self.assertEqual(ts1.counter, 1)
        self.assertEqual(ts2.counter, 2)
        self.assertEqual(ts3.counter, 3)

    def test_send_receive_updates(self) -> None:
        p1 = Process(pid=1)
        p2 = Process(pid=2)

        # p1 sends a message
        send_ts = p1.send("msg")
        self.assertEqual(send_ts.counter, 1)

        # p2 receives the message
        recv_ts = p2.receive(send_ts, "msg_recv")
        self.assertEqual(recv_ts.counter, 2)  # max(0,1)+1

        # p2 sends a reply
        reply_ts = p2.send("reply")
        self.assertEqual(reply_ts.counter, 3)

        # p1 receives the reply
        recv2_ts = p1.receive(reply_ts, "reply_recv")
        self.assertEqual(recv2_ts.counter, 4)

    def test_timestamp_comparison(self) -> None:
        t1 = Timestamp(counter=5, pid=1)
        t2 = Timestamp(counter=5, pid=2)
        t3 = Timestamp(counter=6, pid=1)
        self.assertTrue(t1 < t2)   # tie‑break by pid
        self.assertTrue(t2 < t3)
        self.assertTrue(t1 < t3)

    def test_merge_logs_total_order(self) -> None:
        p1 = Process(pid=1)
        p2 = Process(pid=2)

        p1.local_event("a")          # (1,1)
        p2.local_event("b")          # (1,2)
        ts_send = p1.send("c")       # (2,1)
        p2.receive(ts_send, "d")     # (3,2)
        p2.send("e")                 # (4,2)
        p1.receive(Timestamp(4, 2), "f")  # (5,1)

        merged = merge_logs(p1, p2)
        counters = [e.timestamp.counter for e in merged]
        self.assertEqual(counters, [1, 1, 2, 3, 4, 5])

        # Verify deterministic tie‑break by pid
        first_two = merged[:2]
        self.assertEqual(first_two[0].timestamp.pid, 1)
        self.assertEqual(first_two[1].timestamp.pid, 2)

    def test_receive_older_timestamp(self) -> None:
        p = Process(pid=1)
        p.local_event("init")               # counter=1
        old_ts = Timestamp(counter=0, pid=99)
        p.receive(old_ts, "receive_old")
        # Counter should advance to max(1,0)+1 = 2
        self.assertEqual(p.get_log()[-1].timestamp.counter, 2)

    def test_performance_benchmark(self) -> None:
        p = Process(pid=1)
        N = 100_000
        start = time.perf_counter()
        for i in range(N):
            p.local_event(f"e{i}")
        duration = time.perf_counter() - start
        # Simple sanity check: should complete within a reasonable time (<2 s on typical hardware)
        self.assertLess(duration, 2.0)


def demo() -> None:
    """
    Small demonstration of two processes exchanging messages and printing the global order.
    """
    p1 = Process(pid=1)
    p2 = Process(pid=2)

    p1.local_event("start")
    ts1 = p1.send("msg1")
    p2.receive(ts1, "msg1_recv")
    ts2 = p2.send("reply")
    p1.receive(ts2, "reply_recv")
    p1.local_event("final")
    p2.local_event("final")

    merged: List[Event] = merge_logs(p1, p2)
    print("Global event order:")
    for ev in merged:
        print(f"Process {ev.timestamp.pid} – Counter {ev.timestamp.counter}: {ev.name}")


if __name__ == "__main__":
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    demo()
