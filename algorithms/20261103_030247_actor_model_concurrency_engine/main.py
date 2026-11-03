import unittest
import time
from typing import List

from core import Actor, ActorSystem, Message

class CounterActor(Actor):
    """
    Simple actor that counts the number of 'increment' messages it receives.
    """
    def __init__(self, system: ActorSystem, actor_id: str):
        super().__init__(system, actor_id)
        self.count = 0

    def receive(self, msg: Message) -> None:
        if msg.type == 'increment':
            self.count += 1
        elif msg.type == 'get':
            if msg.reply_to:
                self.reply(msg.reply_to, 'reply', self.count)

class CollectorActor(Actor):
    """
    Actor that collects all 'reply' messages it receives.
    """
    def __init__(self, system: ActorSystem, actor_id: str):
        super().__init__(system, actor_id)
        self.received: List[Any] = []

    def receive(self, msg: Message) -> None:
        if msg.type == 'reply':
            self.received.append(msg.payload)

def wait_for_condition(cond, timeout=2.0, interval=0.05):
    """
    Utility to wait until a condition becomes True or timeout expires.
    """
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(interval)
    return False

class TestActorSystem(unittest.TestCase):
    def setUp(self):
        self.system = ActorSystem()

    def tearDown(self):
        self.system.stop_all()

    def test_message_order_and_count(self):
        counter_id = self.system.spawn(CounterActor)
        for _ in range(5):
            self.system.send(counter_id, 'increment')
        # Allow actor to process messages
        time.sleep(0.2)
        collector_id = self.system.spawn(CollectorActor)
        self.system.send(counter_id, 'get', reply_to=collector_id)
        self.assertTrue(
            wait_for_condition(lambda: len(self.system._actors[collector_id].received) == 1),
            "Collector did not receive reply in time"
        )
        self.assertEqual(
            self.system._actors[collector_id].received[0],
            5,
            "Counter did not report correct count"
        )

    def test_reply_mechanism(self):
        counter_id = self.system.spawn(CounterActor)
        self.system.send(counter_id, 'increment')
        collector_id = self.system.spawn(CollectorActor)
        self.system.send(counter_id, 'get', reply_to=collector_id)
        self.assertTrue(
            wait_for_condition(lambda: len(self.system._actors[collector_id].received) == 1),
            "Collector did not receive reply"
        )
        self.assertEqual(
            self.system._actors[collector_id].received[0],
            1,
            "Reply payload incorrect"
        )

    def test_actor_stop(self):
        counter_id = self.system.spawn(CounterActor)
        self.system.send(counter_id, 'increment')
        time.sleep(0.1)
        self.system.stop(counter_id)
        # After stopping, sending a message should raise an error
        with self.assertRaises(ValueError):
            self.system.send(counter_id, 'increment')

    def test_invalid_actor(self):
        with self.assertRaises(ValueError):
            self.system.send('nonexistent', 'increment')

    def test_multiple_actors(self):
        counter_ids = [self.system.spawn(CounterActor) for _ in range(3)]
        for cid in counter_ids:
            for _ in range(3):
                self.system.send(cid, 'increment')
        collector_id = self.system.spawn(CollectorActor)
        for cid in counter_ids:
            self.system.send(cid, 'get', reply_to=collector_id)
        self.assertTrue(
            wait_for_condition(lambda: len(self.system._actors[collector_id].received) == 3),
            "Collector did not receive all replies"
        )
        self.assertCountEqual(
            self.system._actors[collector_id].received,
            [3, 3, 3],
            "Counts per actor incorrect"
        )

if __name__ == '__main__':
    unittest.main(verbosity=2)
    # Demo: simple counter actor
    system = ActorSystem()
    counter_id = system.spawn(CounterActor)
    collector_id = system.spawn(CollectorActor)
    for _ in range(10):
        system.send(counter_id, 'increment')
    system.send(counter_id, 'get', reply_to=collector_id)
    # Wait for reply
    wait_for_condition(lambda: len(system._actors[collector_id].received) == 1)
    print(f'Counter value reported: {system._actors[collector_id].received[0]}')
    system.stop_all()
