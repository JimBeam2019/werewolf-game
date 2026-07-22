import unittest

from domain.entities import Player
from domain.enums import GameResult, Role
from domain.exceptions import InvalidPlayerCountError
from domain.rules import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    WinConditionService,
    werewolf_count_for,
)


class TestWerewolfCountFor(unittest.TestCase):
    def test_below_minimum_raises(self):
        with self.assertRaises(InvalidPlayerCountError):
            werewolf_count_for(MIN_PLAYERS - 1)

    def test_above_maximum_raises(self):
        with self.assertRaises(InvalidPlayerCountError):
            werewolf_count_for(MAX_PLAYERS + 1)

    def test_small_games_get_one_werewolf(self):
        for n in range(MIN_PLAYERS, 7):
            self.assertEqual(werewolf_count_for(n), 1)

    def test_larger_games_get_two_werewolves(self):
        for n in (7, 8):
            self.assertEqual(werewolf_count_for(n), 2)


class TestWinConditionService(unittest.TestCase):
    def _players(self, roles_and_status):
        return [
            Player(id=i, name=f"P{i}", role=role, is_alive=alive)
            for i, (role, alive) in enumerate(roles_and_status)
        ]

    def test_villagers_win_when_no_werewolves_alive(self):
        players = self._players(
            [
                (Role.WEREWOLF, False),
                (Role.VILLAGER, True),
                (Role.VILLAGER, True),
            ]
        )
        self.assertEqual(
            WinConditionService.evaluate(players), GameResult.VILLAGERS_WIN
        )

    def test_werewolves_win_when_equal_to_villagers(self):
        players = self._players(
            [
                (Role.WEREWOLF, True),
                (Role.VILLAGER, True),
            ]
        )
        self.assertEqual(
            WinConditionService.evaluate(players), GameResult.WEREWOLVES_WIN
        )

    def test_werewolves_win_when_outnumbering_villagers(self):
        players = self._players(
            [
                (Role.WEREWOLF, True),
                (Role.WEREWOLF, True),
                (Role.VILLAGER, True),
            ]
        )
        self.assertEqual(
            WinConditionService.evaluate(players), GameResult.WEREWOLVES_WIN
        )

    def test_ongoing_when_werewolves_outnumbered(self):
        players = self._players(
            [
                (Role.WEREWOLF, True),
                (Role.VILLAGER, True),
                (Role.VILLAGER, True),
            ]
        )
        self.assertEqual(WinConditionService.evaluate(players), GameResult.ONGOING)


if __name__ == "__main__":
    unittest.main()
