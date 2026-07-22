from typing import List

from domain.entities import Player
from domain.enums import GameResult, Role
from domain.exceptions import InvalidPlayerCountError

MIN_PLAYERS = 5
MAX_PLAYERS = 8
MAX_WEREWOLVES = 2

# Below this player count, a single werewolf is used so villagers still have
# a fair chance. At 7-8 players, two werewolves are used (the game's max).
WEREWOLF_SCALING_THRESHOLD = 7


def werewolf_count_for(num_players: int) -> int:
    """Determine how many werewolves should be in play for a given player count."""
    if not (MIN_PLAYERS <= num_players <= MAX_PLAYERS):
        raise InvalidPlayerCountError(
            f"Player count must be between {MIN_PLAYERS} and {MAX_PLAYERS}, got {num_players}."
        )
    return MAX_WEREWOLVES if num_players >= WEREWOLF_SCALING_THRESHOLD else 1


class WinConditionService:
    """Pure domain logic for deciding whether the game has been won by
    either faction. Contains no I/O or orchestration concerns.
    """

    @staticmethod
    def evaluate(players: List[Player]) -> GameResult:
        alive = [p for p in players if p.is_alive]
        alive_werewolves = [p for p in alive if p.role == Role.WEREWOLF]
        alive_villagers = [p for p in alive if p.role == Role.VILLAGER]

        if not alive_werewolves:
            return GameResult.VILLAGERS_WIN
        if len(alive_werewolves) >= len(alive_villagers):
            return GameResult.WEREWOLVES_WIN
        return GameResult.ONGOING
