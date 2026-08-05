import unittest

from langchain_core.messages import AIMessage

from domain.entities import ChatMessage, Player
from domain.enums import Role
from infrastructure.transcript_vote_strategy import (
    LangChainVoteStrategy,
    StubTranscriptVoteStrategy,
)


def make_candidates():
    return [
        Player(id=0, name="Alice", role=Role.VILLAGER),
        Player(id=1, name="Bob", role=Role.VILLAGER),
        Player(id=2, name="Carol", role=Role.WEREWOLF),
    ]


class TestStubTranscriptVoteStrategy(unittest.TestCase):
    def test_votes_for_random_candidate_with_no_transcript(self):
        strategy = StubTranscriptVoteStrategy()
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        self.assertIn(strategy.cast_vote(voter, candidates, None), candidates)

    def test_votes_for_most_mentioned_candidate(self):
        strategy = StubTranscriptVoteStrategy()
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        transcript = [
            ChatMessage(speaker_name="Dave", content="I think Bob is suspicious."),
            ChatMessage(speaker_name="Alice", content="Yeah, Bob has been quiet."),
        ]
        result = strategy.cast_vote(voter, candidates, transcript)
        self.assertEqual(result.name, "Bob")


class ScriptedFakeLLM:
    """Minimal fake chat model compatible with langchain.agents.create_agent:
    pops pre-scripted responses in order regardless of the actual input,
    and bind_tools just returns the same instance so the tool-calling
    loop keeps consuming from the same script.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_count = 0

    def invoke(self, messages):
        self.invoke_count += 1
        if not self._responses:
            return AIMessage(content="")
        return self._responses.pop(0)

    def bind_tools(self, tools, **kwargs):
        return self


class FakeSummaryStore:
    def __init__(self, previous_summary=None):
        self._previous_summary = previous_summary
        self.load_previous_summary_calls = []

    def load_previous_summary(self, game_id, round_number):
        self.load_previous_summary_calls.append((game_id, round_number))
        return self._previous_summary


class TestLangChainVoteStrategy(unittest.TestCase):
    def _make_strategy(self, llm, previous_summary=None, round_number=1):
        summary_store = FakeSummaryStore(previous_summary)
        return (
            LangChainVoteStrategy(
                llm,
                summary_store=summary_store,
                game_id_getter=lambda: "game-1",
                round_getter=lambda: round_number,
            ),
            summary_store,
        )

    def test_direct_reply_without_tool_calls(self):
        llm = ScriptedFakeLLM([AIMessage(content="Bob")])
        strategy, _ = self._make_strategy(llm)
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        result = strategy.cast_vote(voter, candidates, [])
        self.assertEqual(result.name, "Bob")

    def test_full_tool_call_then_decide_flow(self):
        llm = ScriptedFakeLLM(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_previous_turn_summary",
                            "args": {},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="Carol"),
            ]
        )
        strategy, summary_store = self._make_strategy(
            llm, previous_summary="Bob was voted out as a werewolf.", round_number=2
        )
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        result = strategy.cast_vote(voter, candidates, [])
        self.assertEqual(result.name, "Carol")
        self.assertEqual(llm.invoke_count, 2)
        # Confirm it actually asked the summary store for the *previous*
        # round relative to the round it was told it's currently in.
        self.assertEqual(summary_store.load_previous_summary_calls, [("game-1", 2)])

    def test_falls_back_to_random_on_unparseable_reply(self):
        llm = ScriptedFakeLLM([AIMessage(content="I refuse to choose.")])
        strategy, _ = self._make_strategy(llm)
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        result = strategy.cast_vote(voter, candidates, [])
        self.assertIn(result, candidates)

    def test_falls_back_to_random_on_model_exception(self):
        class ExplodingLLM:
            def invoke(self, messages):
                raise RuntimeError("model unavailable")

            def bind_tools(self, tools, **kwargs):
                return self

        strategy, _ = self._make_strategy(ExplodingLLM())
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        result = strategy.cast_vote(voter, candidates, [])
        self.assertIn(result, candidates)

    def test_previous_summary_is_none_on_first_round(self):
        llm = ScriptedFakeLLM([AIMessage(content="Bob")])
        strategy, summary_store = self._make_strategy(
            llm, previous_summary=None, round_number=1
        )
        candidates = make_candidates()
        voter = Player(id=99, name="Dave", role=Role.VILLAGER)
        strategy.cast_vote(voter, candidates, [])
        self.assertEqual(summary_store.load_previous_summary_calls, [("game-1", 1)])


if __name__ == "__main__":
    unittest.main()
