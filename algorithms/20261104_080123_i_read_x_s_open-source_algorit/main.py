from __future__ import annotations
from typing import List, Set, Dict

from custom_types import User, Post, Interaction, InteractionType
from engine import ScoringEngine


def _sample_data() -> tuple[List[User], List[Post], List[Interaction], Dict[int, Set[User]]]:
    # Users
    alice = User(user_id=1, name="Alice")
    bob = User(user_id=2, name="Bob")
    carol = User(user_id=3, name="Carol")
    dave = User(user_id=4, name="Dave")
    eve = User(user_id=5, name="Eve")

    users = [alice, bob, carol, dave, eve]

    # Posts
    post_a = Post(post_id=101, author=alice, content="Hello world!")
    post_b = Post(post_id=102, author=bob, content="Python is great.")
    posts = [post_a, post_b]

    # Mutual friends mapping (author user_id -> set of mutual friends)
    # Assume Alice's mutual friends are Bob and Carol.
    # Bob's mutual friends are Alice only.
    mutual_map: Dict[int, Set[User]] = {
        alice.user_id: {bob, carol},
        bob.user_id: {alice},
    }

    # Interactions
    interactions = [
        # Likes on Alice's post
        Interaction(InteractionType.LIKE, user=bob, post_id=101),
        Interaction(InteractionType.LIKE, user=carol, post_id=101),
        Interaction(InteractionType.LIKE, user=dave, post_id=101),
        # Replies on Alice's post
        Interaction(InteractionType.REPLY, user=bob, post_id=101),   # mutual -> 30
        Interaction(InteractionType.REPLY, user=eve, post_id=101),   # non‑mutual -> 5
        # Likes on Bob's post
        Interaction(InteractionType.LIKE, user=alice, post_id=102),
        Interaction(InteractionType.LIKE, user=carol, post_id=102),
        # Replies on Bob's post
        Interaction(InteractionType.REPLY, user=alice, post_id=102), # mutual -> 30
        Interaction(InteractionType.REPLY, user=dave, post_id=102),  # non‑mutual -> 5
    ]

    return users, posts, interactions, mutual_map


def _run_unit_tests() -> None:
    _, posts, interactions, mutual_map = _sample_data()
    alice_post = posts[0]
    bob_post = posts[1]

    # Test 1: Score for Alice's post
    score_alice = ScoringEngine.compute_score(
        post=alice_post,
        interactions=interactions,
        mutual_friends=mutual_map[alice_post.author.user_id],
    )
    # Expected: 3 likes *1 =3, 1 mutual reply *30 =30, 1 non‑mutual reply *5 =5 => total 38
    assert score_alice == 38, f"Expected 38, got {score_alice}"

    # Test 2: Score for Bob's post
    score_bob = ScoringEngine.compute_score(
        post=bob_post,
        interactions=interactions,
        mutual_friends=mutual_map[bob_post.author.user_id],
    )
    # Expected: 2 likes *1 =2, 1 mutual reply *30 =30, 1 non‑mutual reply *5 =5 => total 37
    assert score_bob == 37, f"Expected 37, got {score_bob}"

    # Test 3: Empty interactions
    empty_score = ScoringEngine.compute_score(
        post=alice_post,
        interactions=[],
        mutual_friends=set(),
    )
    assert empty_score == 0, f"Empty interactions should yield 0, got {empty_score}"

    # Test 4: Interaction for unrelated post should be ignored
    unrelated_inter = Interaction(InteractionType.LIKE, user=User(99, "Ghost"), post_id=999)
    score_with_unrelated = ScoringEngine.compute_score(
        post=alice_post,
        interactions=[unrelated_inter],
        mutual_friends=set(),
    )
    assert score_with_unrelated == 0, "Interactions for other posts must be ignored."

    # Test 5: Aggregate scores across multiple posts
    agg_scores = ScoringEngine.aggregate_scores(
        posts=posts,
        interactions=interactions,
        mutual_map=mutual_map,
    )
    assert agg_scores == {101: 38, 102: 37}, f"Aggregate scores mismatch: {agg_scores}"

    # Test 6: Verify that duplicate likes are counted (no deduplication)
    dup_like = Interaction(InteractionType.LIKE, user=User(2, "Bob"), post_id=101)
    dup_score = ScoringEngine.compute_score(
        post=alice_post,
        interactions=[dup_like, dup_like],
        mutual_friends=mutual_map[alice_post.author.user_id],
    )
    # Two likes from same user => 2 points
    assert dup_score == 2, f"Duplicate likes should each count: got {dup_score}"

    print("All unit tests passed.")


if __name__ == "__main__":
    _run_unit_tests()
    # Demonstration output
    users, posts, interactions, mutual_map = _sample_data()
    for post in posts:
        score = ScoringEngine.compute_score(
            post,
            interactions,
            mutual_map.get(post.author.user_id, set()),
        )
        print(f"Post {post.post_id} by {post.author.name} has score: {score}")
