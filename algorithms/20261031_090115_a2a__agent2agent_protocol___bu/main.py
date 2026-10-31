"""
Unit‑test suite for the A2A protocol implementation.
"""

import unittest
from typing import Any

from core import Agent, NetworkSimulator, Message


class TestVectorClock(unittest.TestCase):
    def test_increment_and_merge(self):
        a = Agent("A")
        b = Agent("B")
        a.send("B", "msg1")  # A clock: {A:1}
        a.send("B", "msg2")  # A clock: {A:2}
        self.assertEqual(a.clock["A"], 2)
        # B receives first message
        net = NetworkSimulator([a, b])
        msg1 = a.send("B", "msg3")  # A clock: {A:3}
        net.send(msg1)
        net.deliver_all()
        self.assertEqual(b.clock["A"], 3)
        self.assertEqual(b.clock["B"], 1)  # B incremented on receipt

    def test_causal_delivery(self):
        a = Agent("A")
        b = Agent("B")
        net = NetworkSimulator([a, b])

        # A sends two messages to B
        m1 = a.send("B", "first")
        m2 = a.send("B", "second")
        net.send(m1)
        net.send(m2)

        # Deliver only the first message
        net.deliver_all()
        self.assertEqual(len(b.inbox), 2)  # both delivered because ordering satisfied

        # Verify clock values
        self.assertEqual(b.clock["A"], 2)
        self.assertEqual(b.clock["B"], 2)

    def test_causal_violation(self):
        a = Agent("A")
        b = Agent("B")
        net = NetworkSimulator([a, b])

        # A sends a message, then B sends a message back before receiving it
        m1 = a.send("B", "hello")
        net.send(m1)

        # B attempts to send a reply prematurely
        with self.assertRaises(ValueError):
            # B's clock is still empty, cannot satisfy can_deliver for reply
            reply = Message(sender="B", receiver="A", payload="reply", timestamp=b.clock)
            a.receive(reply)

    def test_concurrent_agents(self):
        agents = [Agent(chr(ord('A') + i)) for i in range(3)]  # A, B, C
        net = NetworkSimulator(agents)

        # A -> B, B -> C, C -> A forming a cycle
        m_ab = agents[0].send("B", "a2b")
        m_bc = agents[1].send("C", "b2c")
        m_ca = agents[2].send("A", "c2a")
        net.send(m_ab)
        net.send(m_bc)
        net.send(m_ca)

        net.deliver_all()

        # All agents should have seen all others' increments
        for ag in agents:
            self.assertEqual(ag.clock["A"], 1)
            self.assertEqual(ag.clock["B"], 1)
            self.assertEqual(ag.clock["C"], 1)

    def test_undeliverable_message_stays_pending(self):
        a = Agent("A")
        b = Agent("B")
        net = NetworkSimulator([a, b])

        # Create a message with a future timestamp manually
        future_clock = a.clock.increment("A").increment("A")  # jumps two steps
        msg = Message(sender="A", receiver="B", payload="future", timestamp=future_clock)
        net.send(msg)

        net.deliver_all()
        self.assertEqual(len(net.pending), 1)  # still pending
        self.assertEqual(len(b.inbox), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
