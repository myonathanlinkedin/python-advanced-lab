import threading
import queue
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

@dataclass
class Message:
    type: str
    payload: Any = None
    reply_to: Optional[str] = None

class Actor:
    """
    Base class for all actors. Each actor runs in its own thread and processes
    messages from its mailbox sequentially.
    """
    def __init__(self, system: 'ActorSystem', actor_id: str):
        self.system = system
        self.actor_id = actor_id
        self.mailbox: queue.Queue[Message] = queue.Queue()
        self._running = threading.Event()
        self._running.set()
        self._thread = threading.Thread(
            target=self._run,
            name=f'Actor-{actor_id}',
            daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        while self._running.is_set():
            try:
                msg = self.mailbox.get(timeout=0.1)
            except queue.Empty:
                continue
            self.receive(msg)

    def receive(self, msg: Message) -> None:
        """
        Override this method to handle incoming messages.
        """
        raise NotImplementedError

    def send(self, msg: Message) -> None:
        """
        Enqueue a message to this actor's mailbox.
        """
        self.mailbox.put(msg)

    def reply(self, reply_to: str, msg_type: str, payload: Any = None) -> None:
        """
        Send a reply message to another actor.
        """
        self.system.send(reply_to, msg_type, payload)

    def stop(self) -> None:
        """
        Stop the actor's processing loop and wait for the thread to finish.
        """
        self._running.clear()
        self._thread.join()

class ActorSystem:
    """
    Manages the lifecycle of actors and facilitates message passing between them.
    """
    def __init__(self):
        self._actors: Dict[str, Actor] = {}
        self._lock = threading.Lock()

    def spawn(self, actor_cls: Type[Actor], *args, **kwargs) -> str:
        """
        Create a new actor instance and return its unique identifier.
        """
        actor_id = str(uuid.uuid4())
        actor = actor_cls(self, actor_id, *args, **kwargs)
        with self._lock:
            self._actors[actor_id] = actor
        return actor_id

    def send(
        self,
        actor_id: str,
        msg_type: str,
        payload: Any = None,
        reply_to: Optional[str] = None
    ) -> None:
        """
        Send a message to the specified actor.
        """
        with self._lock:
            actor = self._actors.get(actor_id)
        if actor is None:
            raise ValueError(f'Actor {actor_id} not found')
        actor.send(Message(type=msg_type, payload=payload, reply_to=reply_to))

    def stop(self, actor_id: str) -> None:
        """
        Stop and remove the specified actor from the system.
        """
        with self._lock:
            actor = self._actors.pop(actor_id, None)
        if actor:
            actor.stop()

    def stop_all(self) -> None:
        """
        Stop all actors managed by the system.
        """
        with self._lock:
            actors = list(self._actors.values())
            self._actors.clear()
        for actor in actors:
            actor.stop()
