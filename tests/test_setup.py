import random
import unittest

from application.setup import GameSetupService
from domain.enums import Role
from domain.exceptions import InvalidPlayerCountError


class TestGameSetupService(unittest.TestCase):
    def test_too_few_players_raises(self):
        with self.assertRaises(InvalidPlayerCountError):
            GameSetupService.create_players(["A", "B", "C"])

    def test_too_many_players_raises(self):
        with self.assertRaises(InvalidPlayerCountError):
            GameSetupService.create_players([f"P{i}" for i in range(9)])

    def test_six_players_get_one_werewolf(self):
        players = GameSetupService.create_players(
            ["A", "B", "C", "D", "E", "F"], rng=random.Random(42)
        )
        werewolves = [p for p in players if p.role == Role.WEREWOLF]
        self.assertEqual(len(werewolves), 1)
        self.assertEqual(len(players), 6)

    def test_eight_players_get_two_werewolves(self):
        players = GameSetupService.create_players(
            ["A", "B", "C", "D", "E", "F", "G", "H"], rng=random.Random(1)
        )
        werewolves = [p for p in players if p.role == Role.WEREWOLF]
        self.assertEqual(len(werewolves), 2)

    def test_names_are_preserved(self):
        names = ["A", "B", "C", "D", "E"]
        players = GameSetupService.create_players(names, rng=random.Random(7))
        self.assertEqual(sorted(p.name for p in players), sorted(names))


if __name__ == "__main__":
    unittest.main()
