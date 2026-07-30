import unittest

from domain.entities import ChatMessage
from infrastructure.langgraph_memory import LangGraphMemoryStore


class TestLangGraphMemoryStore(unittest.TestCase):
    def test_unseen_game_id_returns_empty_history(self):
        store = LangGraphMemoryStore()
        self.assertEqual(store.load_history("never-seen"), [])

    def test_append_and_save_accumulates_across_calls(self):
        store = LangGraphMemoryStore()
        store.append_and_save(
            "game-1", [ChatMessage(speaker_name="Alice", content="I suspect Bob")]
        )
        result = store.append_and_save(
            "game-1", [ChatMessage(speaker_name="Game", content="Bob was killed.")]
        )

        contents = [(m.speaker_name, m.content) for m in result]
        self.assertEqual(
            contents,
            [("Alice", "I suspect Bob"), ("Game", "Bob was killed.")],
        )

    def test_load_history_reflects_prior_appends(self):
        store = LangGraphMemoryStore()
        store.append_and_save(
            "game-2", [ChatMessage(speaker_name="Carol", content="Hello")]
        )
        loaded = store.load_history("game-2")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].speaker_name, "Carol")

    def test_games_are_isolated_by_id(self):
        store = LangGraphMemoryStore()
        store.append_and_save(
            "game-a", [ChatMessage(speaker_name="Alice", content="only in game A")]
        )
        self.assertEqual(store.load_history("game-b"), [])

    def test_appending_empty_list_is_a_no_op_and_returns_current_history(self):
        store = LangGraphMemoryStore()
        store.append_and_save(
            "game-3", [ChatMessage(speaker_name="Alice", content="first")]
        )
        result = store.append_and_save("game-3", [])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].content, "first")

    def test_round_trip_preserves_is_human_flag(self):
        store = LangGraphMemoryStore()
        store.append_and_save(
            "game-4", [ChatMessage(speaker_name="Jim", content="hi", is_human=True)]
        )
        loaded = store.load_history("game-4")
        self.assertTrue(loaded[0].is_human)


if __name__ == "__main__":
    unittest.main()
