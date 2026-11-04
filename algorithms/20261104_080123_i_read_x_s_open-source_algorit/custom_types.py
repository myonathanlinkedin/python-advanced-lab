from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Set, List


class InteractionType(Enum):
    LIKE = auto()
    REPLY = auto()


@dataclass(frozen=True, slots=True)
class User:
    """Represents a user in the social network."""
    user_id: int
    name: str


@dataclass(frozen=True, slots=True)
class Post:
    """Represents a post created by a user."""
    post_id: int
    author: User
    content: str


@dataclass(frozen=True, slots=True)
class Interaction:
    """An interaction (like or reply) on a post."""
    interaction_type: InteractionType
    user: User
    post_id: int
    # For REPLY interactions we could store reply text, but it's unnecessary for scoring.
    # reply_text: str = ""
