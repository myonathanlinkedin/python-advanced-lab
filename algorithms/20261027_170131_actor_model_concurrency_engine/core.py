import threading
import queue
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypeVar, Generic, Protocol

T = TypeVar("T")

class Message(Protocol):
    """Protocol for messages processed by actors."""
    pass

class Actor(Generic[T]):
    """
    Base class for actors in the Actor Model.
    Each actor runs in its own thread and processes messages sequentially.
    """
    def __init__(self) -> None:
        self._mailbox: queue.Queue[Message] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the actor's processing thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the actor and wait for the thread to finish."""
        with self._lock:
            if not self._running:
                return
            self._running = False
            # Send sentinel to unblock queue
            self._mailbox.put(None)
        if self._thread:
            self._thread.join(timeout)

    def send(self, msg: Message) -> None:
        """Send a message to the actor."""
        if not self._running:
            raise RuntimeError("Actor not running")
        self._mailbox.put(msg)

    def _run(self) -> None:
        while self._running:
            msg = self._mailbox.get()
            if msg is None:
                # Sentinel received; exit loop
                break
            try:
                self.process_message(msg)
            except Exception as e:
                # In production, log the exception; here we simply ignore
                pass

    def process_message(self, msg: Message) -> None:
        """Override this method to handle incoming messages."""
        raise NotImplementedError

# Example message types
@dataclass(frozen=True)
class Increment(Message):
    amount: int = 1

@dataclass(frozen=True)
class GetValue(Message):
    reply_to: Callable[[int], None]

# Example actor implementation
class CounterActor(Actor[Message]):
    """
    Actor that maintains an integer counter.
    Supports Increment and GetValue messages.
    """
    def __init__(self) -> None:
        super().__init__()
        self._value: int = 0

    def process_message(self, msg: Message) -> None:
        if isinstance(msg, Increment):
            self._value += msg.amount
        elif isinstance(msg, GetValue):
            msg.reply_to(self._value)
        else:
            raise ValueError(f"Unsupported message type: {type(msg)}")

    @property
    def value(self) -> int:
        return self._value
