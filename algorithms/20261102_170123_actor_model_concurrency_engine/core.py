import threading
import queue
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound="Actor")


@dataclass(frozen=True)
class Message:
    sender: str
    payload: Any


class Actor:
    """
    Base class for all actors. Each actor runs in its own thread and processes
    messages from a FIFO mailbox.
    """

    def __init__(self, system: "ActorSystem") -> None:
        self._id: str = str(uuid.uuid4())
        self._system: ActorSystem = system
        self._mailbox: queue.Queue[Message] = queue.Queue()
        self._stop_event: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run, name=f"Actor-{self._id}", daemon=True
        )
        self._lock: threading.Lock = threading.Lock()

    @property
    def id(self) -> str:
        return self._id

    def start(self) -> None:
        """Start the actor's processing thread."""
        self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Signal the actor to stop and wait for the thread to finish."""
        self._stop_event.set()
        self._thread.join(timeout)

    def send(self, message: Message) -> None:
        """Enqueue a message to the actor's mailbox."""
        self._mailbox.put(message)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                msg: Message = self._mailbox.get(timeout=0.1)
                self.on_receive(msg)
            except queue.Empty:
                continue

    def on_receive(self, message: Message) -> None:
        """
        Override this method to handle incoming messages.
        The default implementation does nothing.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self._id}>"


class ActorSystem:
    """
    Manages the lifecycle of actors and provides a simple API for message
    passing between them.
    """

    def __init__(self) -> None:
        self._actors: Dict[str, Actor] = {}
        self._lock: threading.Lock = threading.Lock()

    def create_actor(self, actor_cls: Type[T], *args, **kwargs) -> T:
        """Instantiate and start a new actor."""
        actor: T = actor_cls(self, *args, **kwargs)
        with self._lock:
            self._actors[actor.id] = actor
        actor.start()
        return actor

    def send(self, actor_id: str, payload: Any, sender_id: Optional[str] = None) -> None:
        """Send a message to the specified actor."""
        with self._lock:
            actor = self._actors.get(actor_id)
        if actor is None:
            raise ValueError(f"Actor with id {actor_id} does not exist.")
        message = Message(sender=sender_id or "system", payload=payload)
        actor.send(message)

    def broadcast(self, payload: Any, sender_id: Optional[str] = None) -> None:
        """Send a message to all actors."""
        with self._lock:
            actors = list(self._actors.values())
        for actor in actors:
            actor.send(Message(sender=sender_id or "system", payload=payload))

    def shutdown(self, timeout: Optional[float] = None) -> None:
        """Stop all actors gracefully."""
        with self._lock:
            actors = list(self._actors.values())
        for actor in actors:
            actor.stop(timeout=timeout)

    def get_actor(self, actor_id: str) -> Optional[Actor]:
        """Retrieve an actor by its id."""
        with self._lock:
            return self._actors.get(actor_id)

    def __repr__(self) -> str:
        return f"<ActorSystem actors={len(self._actors)}>"
