import time
import unittest
from typing import List

from core import Actor, ActorSystem, Message


class EchoActor(Actor):
    """
    Simple actor that echoes received messages back to the sender.
    """

    def on_receive(self, message: Message) -> None:
        # Echo back the payload
        if message.sender != self.id:
            self._system.send(message.sender, f"echo:{message.payload}", sender_id=self.id)


class CounterActor(Actor):
    """
    Actor that counts the number of messages received.
    """

    def __init__(self, system: ActorSystem) -> None:
        super().__init__(system)
        self.count: int = 0
        self._received: List[Any] = []

    def on_receive(self, message: Message) -> None:
        self.count += 1
        self._received.append(message.payload)


class TestActorSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.system = ActorSystem()

    def tearDown(self) -> None:
        self.system.shutdown(timeout=1)

    def test_actor_creation_and_shutdown(self) -> None:
        actor = self.system.create_actor(EchoActor)
        self.assertIsNotNone(actor.id)
        self.assertIn(actor.id, self.system._actors)
        actor.stop(timeout=1)
        self.assertFalse(actor._thread.is_alive())

    def test_message_ordering(self) -> None:
        counter = self.system.create_actor(CounterActor)
        for i in range(5):
            self.system.send(counter.id, i)
        time.sleep(0.5)
        self.assertEqual(counter.count, 5)
        self.assertEqual(counter._received, list(range(5)))

    def test_echo_actor(self) -> None:
        echo = self.system.create_actor(EchoActor)
        counter = self.system.create_actor(CounterActor)
        # Send message from counter to echo
        self.system.send(echo.id, "hello", sender_id=counter.id)
        time.sleep(0.5)
        self.assertEqual(counter.count, 1)
        self.assertEqual(counter._received[0], "echo:hello")

    def test_broadcast(self) -> None:
        counter1 = self.system.create_actor(CounterActor)
        counter2 = self.system.create_actor(CounterActor)
        self.system.broadcast("broadcast")
        time.sleep(0.5)
        self.assertEqual(counter1.count, 1)
        self.assertEqual(counter2.count, 1)

    def test_invalid_actor(self) -> None:
        with self.assertRaises(ValueError):
            self.system.send("nonexistent", "msg")

    def test_actor_spawn_from_actor(self) -> None:
        class SpawnerActor(Actor):
            def on_receive(self, message: Message) -> None:
                # Spawn a new CounterActor
                new_actor = self._system.create_actor(CounterActor)
                # Send a message to the new actor
                self._system.send(new_actor.id, "spawned", sender_id=self.id)

        spawner = self.system.create_actor(SpawnerActor)
        self.system.send(spawner.id, "start")
        time.sleep(0.5)
        # There should be two CounterActors now
        counters = [a for a in self.system._actors.values() if isinstance(a, CounterActor)]
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0].count, 1)
        self.assertEqual(counters[0]._received[0], "spawned")

    def test_actor_thread_is_daemon(self) -> None:
        actor = self.system.create_actor(CounterActor)
        self.assertTrue(actor._thread.daemon)


if __name__ == "__main__":
    unittest.main(verbosity=2, exit=False)
    # Demo: simple echo
    system = ActorSystem()
    echo = system.create_actor(EchoActor)
    counter = system.create_actor(CounterActor)
    system.send(echo.id, "demo message", sender_id=counter.id)
    time.sleep(0.5)
    print(f"Counter received: {counter._received}")
    system.shutdown(timeout=1)
