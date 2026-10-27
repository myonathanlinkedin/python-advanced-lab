import unittest
import time
from core import CounterActor, Increment, GetValue, Actor

class TestActorModel(unittest.TestCase):
    def test_single_actor_sequential(self):
        actor = CounterActor()
        actor.start()
        actor.send(Increment(5))
        actor.send(Increment(3))
        # Allow some time for processing
        time.sleep(0.1)
        self.assertEqual(actor.value, 8)
        actor.stop()

    def test_get_value_reply(self):
        actor = CounterActor()
        actor.start()
        result_container = {}
        def reply(val: int) -> None:
            result_container['value'] = val
        actor.send(Increment(10))
        time.sleep(0.05)
        actor.send(GetValue(reply_to=reply))
        time.sleep(0.05)
        self.assertEqual(result_container.get('value'), 10)
        actor.stop()

    def test_actor_stop(self):
        actor = CounterActor()
        actor.start()
        actor.send(Increment(1))
        time.sleep(0.05)
        actor.stop()
        # After stop, sending should raise
        with self.assertRaises(RuntimeError):
            actor.send(Increment(1))

    def test_concurrent_actors(self):
        actors = [CounterActor() for _ in range(5)]
        for a in actors:
            a.start()
        # Send increments concurrently
        for i, a in enumerate(actors):
            a.send(Increment(i + 1))
        time.sleep(0.1)
        expected = sum(i + 1 for i in range(5))
        total = sum(a.value for a in actors)
        self.assertEqual(total, expected)
        for a in actors:
            a.stop()

    def test_message_order(self):
        actor = CounterActor()
        actor.start()
        order = []
        def reply(val: int) -> None:
            order.append(val)
        # Send multiple GetValue messages
        actor.send(GetValue(reply_to=reply))
        actor.send(Increment(2))
        actor.send(GetValue(reply_to=reply))
        actor.send(Increment(3))
        actor.send(GetValue(reply_to=reply))
        time.sleep(0.2)
        self.assertEqual(order, [0, 2, 5])
        actor.stop()

if __name__ == '__main__':
    # Run unit tests
    unittest.main(exit=False)

    # Demo: simple actor system
    actor = CounterActor()
    actor.start()
    print("Initial counter:", actor.value)
    actor.send(Increment(7))
    time.sleep(0.05)
    print("After increment:", actor.value)
    def print_value(val: int) -> None:
        print("Counter value reported:", val)
    actor.send(GetValue(reply_to=print_value))
    time.sleep(0.05)
    actor.stop()
    print("Actor stopped.")
