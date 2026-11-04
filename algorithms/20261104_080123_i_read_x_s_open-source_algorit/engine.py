from __future__ import annotations
from typing import List, Set, Dict
from collections import defaultdict

from .types import Interaction, InteractionType, User, Post


class ScoringEngine:
    """
    Core algorithm to compute a post's influence score.

    Scoring rules (configurable via class constants):
    - Each LIKE contributes +1 point.
    - Each REPLY contributes +5 points.
    - If a REPLY originates from a mutual friend (i.e., the replying user is in the
      provided mutual set), that REPLY contributes +30 points instead of +5.
    """

    LIKE_SCORE = 1
    REPLY_SCORE = 5
    MUTUAL_REPLY_SCORE = 30

    @classmethod
    def compute_score(
        cls,
        post: Post,
        interactions: List[Interaction],
        mutual_friends: Set[User],
    ) -> int:
        """
        Compute the total score for a given post.

        Parameters
        ----------
        post: Post
            The post whose score is being calculated.
        interactions: List[Interaction]
            All interactions (likes and replies) related to the post.
        mutual_friends: Set[User]
            Set of users considered mutual friends of the post author.

        Returns
        -------
        int
            The total influence score.
        """
        total = 0
        for inter in interactions:
            if inter.post_id != post.post_id:
                # Ignore interactions not belonging to the target post.
                continue

            if inter.interaction_type is InteractionType.LIKE:
                total += cls.LIKE_SCORE
            elif inter.interaction_type is InteractionType.REPLY:
                if inter.user in mutual_friends:
                    total += cls.MUTUAL_REPLY_SCORE
                else:
                    total += cls.REPLY_SCORE
            else:
                # Defensive programming: unknown interaction types are ignored.
                continue
        return total

    @classmethod
    def aggregate_scores(
        cls,
        posts: List[Post],
        interactions: List[Interaction],
        mutual_map: Dict[int, Set[User]],
    ) -> Dict[int, int]:
        """
        Compute scores for multiple posts in a single pass.

        Parameters
        ----------
        posts: List[Post]
            Collection of posts to score.
        interactions: List[Interaction]
            All interactions across the posts.
        mutual_map: Dict[int, Set[User]]
            Mapping from post author user_id to the set of that author's mutual friends.

        Returns
        -------
        Dict[int, int]
            Mapping from post_id to its computed score.
        """
        # Group interactions by post_id for O(N) processing.
        inter_by_post: Dict[int, List[Interaction]] = defaultdict(list)
        for inter in interactions:
            inter_by_post[inter.post_id].append(inter)

        scores: Dict[int, int] = {}
        for post in posts:
            mutuals = mutual_map.get(post.author.user_id, set())
            post_interactions = inter_by_post.get(post.post_id, [])
            scores[post.post_id] = cls.compute_score(post, post_interactions, mutuals)
        return scores
