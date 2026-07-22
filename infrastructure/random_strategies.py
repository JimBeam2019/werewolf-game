import random
from typing import List, Optional

from domain.entities import ChatMessage, Player


class RandomWerewolfStrategy:
    """Simulation-friendly strategy: werewolves pick a random living villager."""

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player]
    ) -> Player:
        return self._rng.choice(candidates)


class RandomVoteStrategy:
    """Simulation-friendly strategy: every player votes for a random other
    living player (this is deliberately naive - it is a prototype baseline
    and easy to later replace with a smarter/heuristic-driven bot).
    Ignores the discussion transcript entirely.
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        return self._rng.choice(candidates)
