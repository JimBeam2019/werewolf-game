import unittest

from infrastructure.turn_summary_store import TurnSummaryStore


class TestTurnSummaryStore(unittest.TestCase):
    def test_load_all_summaries_empty_for_unknown_game(self):
        store = TurnSummaryStore()
        self.assertEqual(store.load_all_summaries("nonexistent"), [])

    def test_load_previous_summary_none_on_round_one(self):
        store = TurnSummaryStore()
        self.assertIsNone(store.load_previous_summary("game-1", 1))

    def test_save_and_load_a_single_summary(self):
        store = TurnSummaryStore()
        store.save_summary("game-1", 1, "Round 1: Bob was voted out.")
        self.assertEqual(
            store.load_all_summaries("game-1"), ["Round 1: Bob was voted out."]
        )

    def test_load_all_summaries_returns_them_in_round_order(self):
        store = TurnSummaryStore()
        # Save out of order to confirm ordering is by round number, not
        # insertion order.
        store.save_summary("game-1", 2, "Round 2 summary")
        store.save_summary("game-1", 1, "Round 1 summary")
        store.save_summary("game-1", 3, "Round 3 summary")
        self.assertEqual(
            store.load_all_summaries("game-1"),
            ["Round 1 summary", "Round 2 summary", "Round 3 summary"],
        )

    def test_load_previous_summary_returns_the_immediately_preceding_round(self):
        store = TurnSummaryStore()
        store.save_summary("game-1", 1, "Round 1 summary")
        store.save_summary("game-1", 2, "Round 2 summary")
        self.assertEqual(store.load_previous_summary("game-1", 2), "Round 1 summary")
        self.assertEqual(store.load_previous_summary("game-1", 3), "Round 2 summary")

    def test_games_are_isolated_by_id(self):
        store = TurnSummaryStore()
        store.save_summary("game-1", 1, "Game 1's round 1")
        store.save_summary("game-2", 1, "Game 2's round 1")
        self.assertEqual(store.load_all_summaries("game-1"), ["Game 1's round 1"])
        self.assertEqual(store.load_all_summaries("game-2"), ["Game 2's round 1"])

    def test_clear_removes_only_the_specified_game(self):
        store = TurnSummaryStore()
        store.save_summary("game-1", 1, "Game 1's round 1")
        store.save_summary("game-2", 1, "Game 2's round 1")
        store.clear("game-1")
        self.assertEqual(store.load_all_summaries("game-1"), [])
        self.assertEqual(store.load_all_summaries("game-2"), ["Game 2's round 1"])

    def test_clear_all_removes_every_game(self):
        store = TurnSummaryStore()
        store.save_summary("game-1", 1, "Game 1's round 1")
        store.save_summary("game-2", 1, "Game 2's round 1")
        store.clear_all()
        self.assertEqual(store.load_all_summaries("game-1"), [])
        self.assertEqual(store.load_all_summaries("game-2"), [])


if __name__ == "__main__":
    unittest.main()
