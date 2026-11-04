import unittest
from typing import List, Tuple

from core import LamportClock, Event, total_order, causally_related


class TestLamportClock(unittest.TestCase):
    def test_tick_monotonic(self) -> None:
        clk = LamportClock(pid=1)
        times = [clk.tick() for _ in range(5)]
        self.assertEqual(times, [1, 2, 3, 4, 5])
        self.assertTrue(all(earlier < later for earlier, later in zip(times, times[1:])))

    def test_update_respects_received_timestamp(self) -> None:
        p1 = LamportClock(pid=1)
        p2 = LamportClock(pid=2)

        # p1 generates two events
        t1 = p1.tick()          # 1
        t2 = p1.tick()          # 2

        # p2 receives a message from p1 with timestamp t2
        t3 = p2.update(t2)      # max(0,2)+1 = 3
        self.assertEqual(t3, 3)

        # subsequent local event on p2
        t4 = p2.tick()          # 4
        self.assertEqual(t4, 4)

    def test_total_order_is_total(self) -> None:
        e1 = Event(process_id=1, local_counter=3, timestamp=5)
        e2 = Event(process_id=2, local_counter=1, timestamp=5)
        e3 = Event(process_id=1, local_counter=4, timestamp=6)

        # e1 and e2 have equal timestamps, tie‑break by pid
        self.assertEqual(total_order(e1, e2), -1)  # pid 1 < pid 2
        self.assertEqual(total_order(e2, e1), 1)

        # e3 has larger timestamp
        self.assertEqual(total_order(e1, e3), -1)
        self.assertEqual(total_order(e3, e2), 1)

    def test_trap_non_causality(self) -> None:
        """
        Construct a scenario where two events are ordered by timestamps
        but have no causal relationship.
        """
        # Process A generates an event
        clk_a = LamportClock(pid=1)
        ev_a = Event(process_id=1, local_counter=1, timestamp=clk_a.tick())  # ts=1

        # Process B generates an event independently
        clk_b = LamportClock(pid=2)
        ev_b = Event(process_id=2, local_counter=1, timestamp=clk_b.tick())  # ts=1

        # Process B performs another local event, advancing its clock
        ev_b2 = Event(process_id=2, local_counter=2, timestamp=clk_b.tick())  # ts=2

        # According to total order, ev_a (ts=1, pid=1) < ev_b2 (ts=2)
        self.assertEqual(total_order(ev_a, ev_b2), -1)

        # However, there is no message flow from A to B, so they are concurrent.
        history: List[Tuple[Event, List[int]]] = [
            (ev_a, []),   # no outgoing messages
            (ev_b, []),
            (ev_b2, []),
        ]
        self.assertFalse(causally_related(ev_a, ev_b2, history))
        self.assertFalse(causally_related(ev_b2, ev_a, history))

    def test_causal_chain_detection(self) -> None:
        """
        Verify that causally_related correctly identifies a chain of messages.
        """
        # Process 1 event
        p1 = LamportClock(pid=1)
        e1 = Event(process_id=1, local_counter=1, timestamp=p1.tick())  # 1

        # Process 2 receives from 1
        p2 = LamportClock(pid=2)
        p2.update(e1.timestamp)  # becomes 2
        e2 = Event(process_id=2, local_counter=1, timestamp=p2.tick())  # 3

        # Process 3 receives from 2
        p3 = LamportClock(pid=3)
        p3.update(e2.timestamp)  # becomes 4
        e3 = Event(process_id=3, local_counter=1, timestamp=p3.tick())  # 5

        # History encodes the message edges (by timestamps)
        history: List[Tuple[Event, List[int]]] = [
            (e1, [e2.timestamp]),   # e1 -> e2
            (e2, [e3.timestamp]),   # e2 -> e3
            (e3, []),
        ]

        self.assertTrue(causally_related(e1, e3, history))
        self.assertTrue(causally_related(e2, e3, history))
        self.assertFalse(causally_related(e3, e1, history))


def demo_simulation() -> None:
    """
    Simple demonstration:
    - Three processes generate events.
    - Some messages are exchanged.
    - The total order is printed alongside causal relationships.
    """
    clocks = {pid: LamportClock(pid) for pid in (1, 2, 3)}
    events: List[Event] = []
    history: List[Tuple[Event, List[int]]] = []

    # Process 1 creates an event and sends a message to Process 2
    e1 = Event(process_id=1, local_counter=1, timestamp=clocks[1].tick())
    events.append(e1)
    history.append((e1, []))  # placeholder; will fill after receiver updates

    # Process 2 receives the message
    clocks[2].update(e1.timestamp)
    e2 = Event(process_id=2, local_counter=1, timestamp=clocks[2].tick())
    events.append(e2)
    history.append((e2, []))

    # Record the edge e1 -> e2
    history[0] = (e1, [e2.timestamp])

    # Process 3 generates two independent events
    e3 = Event(process_id=3, local_counter=1, timestamp=clocks[3].tick())
    e4 = Event(process_id=3, local_counter=2, timestamp=clocks[3].tick())
    events.extend([e3, e4])
    history.extend([(e3, []), (e4, [])])

    # Print total order
    sorted_events = sorted(events, key=lambda ev: (ev.timestamp, ev.process_id))
    print("Total order (timestamp, pid):")
    for ev in sorted_events:
        print(f"  Process {ev.process_id} – Event {ev.local_counter} – ts={ev.timestamp}")

    # Show causality matrix
    print("\nCausality (True means 'happened before'):")
    for a in events:
        row = []
        for b in events:
            row.append('Y' if causally_related(a, b, history) else 'N')
        print(' '.join(row))


if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Run demo
    print("\n--- Demo Simulation ---")
    demo_simulation()
