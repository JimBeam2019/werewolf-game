import unittest

from domain.entities import ChatMessage
from infrastructure.turn_summary_strategy import (
    LangChainSummaryStrategy,
    StubTurnSummaryStrategy,
)


class TestStubTurnSummaryStrategy(unittest.TestCase):
    def test_mentions_eliminated_player_and_role(self):
        strategy = StubTurnSummaryStrategy()
        summary = strategy.summarize_turn(
            round_number=1,
            transcript=[],
            eliminated_name="Bob",
            eliminated_role="werewolf",
            votes={"Alice": "Bob"},
            tied=False,
        )
        self.assertIn("Bob", summary)
        self.assertIn("werewolf", summary)
        self.assertIn("Round 1", summary)

    def test_mentions_tie_when_tied(self):
        strategy = StubTurnSummaryStrategy()
        summary = strategy.summarize_turn(
            round_number=2,
            transcript=[],
            eliminated_name=None,
            eliminated_role=None,
            votes={"Alice": "Bob", "Bob": "Alice"},
            tied=True,
        )
        self.assertIn("tied", summary.lower())
        self.assertNotIn("None", summary)

    def test_mentions_no_elimination_when_not_tied_and_no_one_eliminated(self):
        strategy = StubTurnSummaryStrategy()
        summary = strategy.summarize_turn(
            round_number=3,
            transcript=[],
            eliminated_name=None,
            eliminated_role=None,
            votes={},
            tied=False,
        )
        self.assertIn("No one was eliminated", summary)


class FakeLLM:
    def __init__(self, response_text=None, raise_error=False):
        self._response_text = response_text
        self._raise_error = raise_error
        self.invoke_calls = 0

    def invoke(self, messages):
        self.invoke_calls += 1
        if self._raise_error:
            raise RuntimeError("model unavailable")

        class _Response:
            def __init__(self, content):
                self.content = content

        return _Response(self._response_text)


class TestLangChainSummaryStrategy(unittest.TestCase):
    def test_uses_the_models_reply_when_available(self):
        llm = FakeLLM(response_text="Bob was accused by Alice and voted out as a werewolf.")
        strategy = LangChainSummaryStrategy(llm)
        summary = strategy.summarize_turn(
            round_number=1,
            transcript=[ChatMessage(speaker_name="Alice", content="I suspect Bob.")],
            eliminated_name="Bob",
            eliminated_role="werewolf",
            votes={"Alice": "Bob"},
            tied=False,
        )
        self.assertEqual(
            summary, "Bob was accused by Alice and voted out as a werewolf."
        )
        self.assertEqual(llm.invoke_calls, 1)

    def test_falls_back_to_stub_on_model_exception(self):
        llm = FakeLLM(raise_error=True)
        strategy = LangChainSummaryStrategy(llm)
        summary = strategy.summarize_turn(
            round_number=1,
            transcript=[],
            eliminated_name="Bob",
            eliminated_role="werewolf",
            votes={"Alice": "Bob"},
            tied=False,
        )
        self.assertIn("Bob", summary)
        self.assertIn("werewolf", summary)

    def test_falls_back_to_stub_on_empty_reply(self):
        llm = FakeLLM(response_text="   ")
        strategy = LangChainSummaryStrategy(llm)
        summary = strategy.summarize_turn(
            round_number=1,
            transcript=[],
            eliminated_name="Carol",
            eliminated_role="villager",
            votes={},
            tied=False,
        )
        self.assertIn("Carol", summary)
        self.assertIn("villager", summary)


if __name__ == "__main__":
    unittest.main()
