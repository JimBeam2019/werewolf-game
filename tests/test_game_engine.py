import asyncio
import unittest
from typing import List

from application.game_engine import GameEngine
from domain.entities import Player
from domain.enums import GameResult, Role
from domain.exceptions import GameAlreadyEndedError
from infrastructure.notifiers import SilentNotifier


class AlwaysKillFirstStrategy:
    """Deterministic werewolf strategy: always kills the first candidate."""

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player], transcript=None
    ) -> Player:
        return candidates[0]


class AlwaysVoteFirstStrategy:
    """Deterministic vote strategy: every voter votes for the first candidate
    in their eligible list, guaranteeing a unanimous (non-tied) elimination.
    """

    def cast_vote(
        self, voter: Player, candidates: List[Player], transcript=None
    ) -> Player:
        return candidates[0]


def make_players():
    return [
        Player(id=0, name="Alice", role=Role.WEREWOLF),
        Player(id=1, name="Bob", role=Role.VILLAGER),
        Player(id=2, name="Carol", role=Role.VILLAGER),
        Player(id=3, name="Dave", role=Role.VILLAGER),
        Player(id=4, name="Eve", role=Role.VILLAGER),
    ]


class TestGameEngine(unittest.TestCase):
    def test_night_phase_kills_exactly_one_villager(self):
        players = make_players()
        engine = GameEngine(
            players=players,
            human_id=999,
            werewolf_strategy=AlwaysKillFirstStrategy(),
            vote_strategy=AlwaysVoteFirstStrategy(),
            notifier=SilentNotifier(),
        )
        result = engine.run_night_phase()
        dead = [p for p in players if not p.is_alive]
        self.assertEqual(len(dead), 1)
        self.assertEqual(result.victim.role, Role.VILLAGER)

    def test_day_phase_eliminates_top_voted_player(self):
        players = make_players()
        engine = GameEngine(
            players=players,
            human_id=999,
            werewolf_strategy=AlwaysKillFirstStrategy(),
            vote_strategy=AlwaysVoteFirstStrategy(),
            notifier=SilentNotifier(),
        )
        result = engine.run_day_phase()
        self.assertIsNotNone(result.eliminated)
        self.assertFalse(result.tied)

    def test_actions_after_game_end_raise(self):
        # Reduce to a state where villagers already won.
        players = [
            Player(id=0, name="Alice", role=Role.WEREWOLF, is_alive=False),
            Player(id=1, name="Bob", role=Role.VILLAGER),
        ]
        engine = GameEngine(
            players=players,
            human_id=999,
            werewolf_strategy=AlwaysKillFirstStrategy(),
            vote_strategy=AlwaysVoteFirstStrategy(),
            notifier=SilentNotifier(),
            result=GameResult.VILLAGERS_WIN,
        )
        with self.assertRaises(GameAlreadyEndedError):
            engine.run_night_phase()

    def test_full_game_terminates_with_a_winner(self):
        players = make_players()
        engine = GameEngine(
            players=players,
            human_id=999,
            werewolf_strategy=AlwaysKillFirstStrategy(),
            vote_strategy=AlwaysVoteFirstStrategy(),
            notifier=SilentNotifier(),
        )
        result = engine.play()
        self.assertIn(result, (GameResult.VILLAGERS_WIN, GameResult.WEREWOLVES_WIN))


if __name__ == "__main__":
    unittest.main()
