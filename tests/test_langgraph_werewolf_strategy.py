import unittest

from langchain_core.messages import AIMessage

from domain.entities import ChatMessage, Player
from domain.enums import Role
from infrastructure.langgraph_werewolf_strategy import LangGraphWerewolfStrategy
from infrastructure.werewolf_tools import build_werewolf_tools


def make_candidates():
    return [
        Player(id=0, name="Alice", role=Role.VILLAGER),
        Player(id=1, name="Bob", role=Role.VILLAGER),
        Player(id=2, name="Carol", role=Role.VILLAGER),
    ]


class TestWerewolfTools(unittest.TestCase):
    def test_list_alive_villagers(self):
        candidates = make_candidates()
        tools = build_werewolf_tools(candidates, [])
        list_alive = next(t for t in tools if t.name == "list_alive_villagers")
        result = list_alive.invoke({})
        self.assertIn("Alice", result)
        self.assertIn("Bob", result)
        self.assertIn("Carol", result)

    def test_get_voting_history_returns_recorded_votes(self):
        candidates = make_candidates()
        transcript = [
            ChatMessage(speaker_name="Game", content="Bob votes for Alice"),
            ChatMessage(speaker_name="Game", content="Carol votes for Alice"),
            ChatMessage(speaker_name="Alice", content="I don't trust Bob."),
        ]
        tools = build_werewolf_tools(candidates, transcript)
        get_history = next(t for t in tools if t.name == "get_voting_history")
        result = get_history.invoke({"player_name": "Alice"})
        self.assertIn("Bob votes for Alice", result)
        self.assertIn("Carol votes for Alice", result)
        # Plain discussion chat mentioning Alice shouldn't be conflated
        # with structured game-system voting events.
        self.assertNotIn("I don't trust Bob", result)

    def test_get_voting_history_rejects_unknown_player(self):
        candidates = make_candidates()
        tools = build_werewolf_tools(candidates, [])
        get_history = next(t for t in tools if t.name == "get_voting_history")
        result = get_history.invoke({"player_name": "Zeke"})
        self.assertIn("not a currently living villager", result)

    def test_check_suspicion_level_counts_other_players_mentions_only(self):
        candidates = make_candidates()
        transcript = [
            ChatMessage(speaker_name="Bob", content="I think Alice is the werewolf."),
            ChatMessage(speaker_name="Carol", content="Alice has been quiet."),
            ChatMessage(
                speaker_name="Alice", content="I promise I'm not Alice-suspicious."
            ),
            ChatMessage(speaker_name="Game", content="Alice votes for Bob"),
        ]
        tools = build_werewolf_tools(candidates, transcript)
        check_suspicion = next(t for t in tools if t.name == "check_suspicion_level")
        result = check_suspicion.invoke({"player_name": "Alice"})
        # Bob's and Carol's messages count; Alice's own message and the
        # Game system event should not.
        self.assertIn("2 time(s)", result)

    def test_get_most_active_speakers_ranks_by_message_count(self):
        candidates = make_candidates()
        transcript = [
            ChatMessage(speaker_name="Bob", content="msg1"),
            ChatMessage(speaker_name="Bob", content="msg2"),
            ChatMessage(speaker_name="Carol", content="msg3"),
            ChatMessage(speaker_name="Game", content="ignored system event"),
        ]
        tools = build_werewolf_tools(candidates, transcript)
        most_active = next(t for t in tools if t.name == "get_most_active_speakers")
        result = most_active.invoke({})
        self.assertIn("Bob: 2 message(s)", result)
        self.assertIn("Carol: 1 message(s)", result)
        self.assertIn("Alice: 0 message(s)", result)


class ScriptedFakeLLM:
    """Minimal fake chat model: pops pre-scripted responses in order,
    regardless of the actual input messages. `bind_tools` returns the
    same instance so the tool-calling sub-loop keeps consuming from the
    same script - this is simpler and more predictable for testing the
    graph's control flow than trying to make a general-purpose fake
    reason about which tool to call.
    """

    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_count = 0

    def invoke(self, messages):
        self.invoke_count += 1
        if not self._responses:
            return AIMessage(content="")
        return self._responses.pop(0)

    def bind_tools(self, tools):
        return self


class FakeSummaryStore:
    def __init__(self, summaries):
        self._summaries = summaries
        self.load_all_summaries_calls = []

    def load_all_summaries(self, game_id):
        self.load_all_summaries_calls.append(game_id)
        return self._summaries


class TestLangGraphWerewolfStrategy(unittest.TestCase):
    def test_single_candidate_is_returned_without_invoking_the_model(self):
        llm = ScriptedFakeLLM([])
        strategy = LangGraphWerewolfStrategy(llm)
        candidates = make_candidates()[:1]
        victim = strategy.choose_victim([], candidates, [])
        self.assertEqual(victim, candidates[0])
        self.assertEqual(llm.invoke_count, 0)

    def test_full_plan_tool_decide_flow(self):
        candidates = make_candidates()
        llm = ScriptedFakeLLM(
            [
                # plan_node
                AIMessage(content="I'll check who's been most active against us."),
                # gather_intel_node #1: decides to call a tool
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_most_active_speakers",
                            "args": {},
                            "id": "call_1",
                        }
                    ],
                ),
                # gather_intel_node #2: done gathering, no more tool calls
                AIMessage(content="I have enough information now."),
                # decide_node
                AIMessage(content="Bob"),
            ]
        )
        strategy = LangGraphWerewolfStrategy(llm)
        werewolves = [Player(id=99, name="Frank", role=Role.WEREWOLF)]
        victim = strategy.choose_victim(werewolves, candidates, [])
        self.assertEqual(victim.name, "Bob")
        # plan + 2 gather_intel calls + decide = 4 real model invocations
        self.assertEqual(llm.invoke_count, 4)

    def test_falls_back_to_random_choice_on_unparseable_decision(self):
        candidates = make_candidates()
        llm = ScriptedFakeLLM(
            [
                AIMessage(content="Some plan."),
                AIMessage(content="Not ready to decide yet, no tool calls though."),
                AIMessage(content="I can't possibly choose between these fine people."),
            ]
        )
        strategy = LangGraphWerewolfStrategy(llm)
        victim = strategy.choose_victim([], candidates, [])
        self.assertIn(victim, candidates)

    def test_falls_back_to_random_choice_on_model_exception(self):
        class ExplodingLLM:
            def invoke(self, messages):
                raise RuntimeError("model unavailable")

            def bind_tools(self, tools):
                return self

        strategy = LangGraphWerewolfStrategy(ExplodingLLM())
        candidates = make_candidates()
        victim = strategy.choose_victim([], candidates, [])
        self.assertIn(victim, candidates)

    def test_tool_call_loop_is_bounded_by_max_iterations(self):
        candidates = make_candidates()
        # The model keeps wanting to call tools forever - the graph must
        # still terminate rather than looping indefinitely.
        endless_tool_calls = AIMessage(
            content="",
            tool_calls=[{"name": "list_alive_villagers", "args": {}, "id": "call_x"}],
        )
        responses = (
            [AIMessage(content="plan")]
            + [endless_tool_calls] * 20
            + [AIMessage(content="Carol")]
        )
        llm = ScriptedFakeLLM(responses)
        strategy = LangGraphWerewolfStrategy(llm, max_tool_iterations=2)
        victim = strategy.choose_victim([], candidates, [])
        self.assertIn(victim, candidates)

    def test_without_summary_store_behaves_exactly_as_before(self):
        # Omitting summary_store/game_id_getter (the default) must not
        # change existing behavior at all - no get_all_turn_summaries
        # tool should be added, and nothing should error from their absence.
        candidates = make_candidates()
        llm = ScriptedFakeLLM([AIMessage(content="Alice")])
        strategy = LangGraphWerewolfStrategy(llm)
        victim = strategy.choose_victim([], candidates, [])
        self.assertIn(victim, candidates)

    def test_summary_tool_is_added_and_reflects_past_rounds_when_configured(self):
        candidates = make_candidates()
        summary_store = FakeSummaryStore(["Round 1: Dave was voted out as a villager."])
        llm = ScriptedFakeLLM(
            [
                AIMessage(content="Checking the game history first."),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_all_turn_summaries",
                            "args": {},
                            "id": "call_1",
                        }
                    ],
                ),
                AIMessage(content="Got it, deciding now."),
                AIMessage(content="Bob"),
            ]
        )
        strategy = LangGraphWerewolfStrategy(
            llm,
            summary_store=summary_store,
            game_id_getter=lambda: "game-42",
        )
        werewolves = [Player(id=99, name="Frank", role=Role.WEREWOLF)]
        victim = strategy.choose_victim(werewolves, candidates, [])
        self.assertEqual(victim.name, "Bob")
        self.assertEqual(summary_store.load_all_summaries_calls, ["game-42"])


if __name__ == "__main__":
    unittest.main()
