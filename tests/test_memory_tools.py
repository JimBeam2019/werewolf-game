import unittest

from domain.entities import ChatMessage
from infrastructure.memory_tools import build_night_summary_tool, build_vote_memory_tools


class TestBuildVoteMemoryTools(unittest.TestCase):
    def test_previous_turn_summary_returns_the_given_summary(self):
        tools = build_vote_memory_tools("Bob was voted out as a werewolf.", [])
        get_summary = next(t for t in tools if t.name == "get_previous_turn_summary")
        self.assertEqual(get_summary.invoke({}), "Bob was voted out as a werewolf.")

    def test_previous_turn_summary_placeholder_on_first_round(self):
        tools = build_vote_memory_tools(None, [])
        get_summary = next(t for t in tools if t.name == "get_previous_turn_summary")
        result = get_summary.invoke({})
        self.assertIn("first round", result.lower())

    def test_recent_chat_messages_returns_at_most_the_last_six(self):
        messages = [
            ChatMessage(speaker_name=f"P{i}", content=f"message {i}") for i in range(10)
        ]
        tools = build_vote_memory_tools(None, messages)
        get_recent = next(t for t in tools if t.name == "get_recent_chat_messages")
        result = get_recent.invoke({})
        # Only the last 6 (messages 4-9) should appear.
        for i in range(4, 10):
            self.assertIn(f"message {i}", result)
        for i in range(0, 4):
            self.assertNotIn(f"message {i}\n", result + "\n")

    def test_recent_chat_messages_placeholder_when_empty(self):
        tools = build_vote_memory_tools(None, [])
        get_recent = next(t for t in tools if t.name == "get_recent_chat_messages")
        result = get_recent.invoke({})
        self.assertIn("no messages", result.lower())

    def test_returns_exactly_two_tools(self):
        tools = build_vote_memory_tools(None, [])
        names = {t.name for t in tools}
        self.assertEqual(names, {"get_previous_turn_summary", "get_recent_chat_messages"})


class TestBuildNightSummaryTool(unittest.TestCase):
    def test_returns_every_summary_in_round_order(self):
        tool = build_night_summary_tool(["Round 1 recap", "Round 2 recap"])
        result = tool.invoke({})
        self.assertIn("Round 1 recap", result)
        self.assertIn("Round 2 recap", result)
        # Round 1's text should appear before round 2's.
        self.assertLess(result.index("Round 1 recap"), result.index("Round 2 recap"))

    def test_placeholder_on_first_night(self):
        tool = build_night_summary_tool([])
        result = tool.invoke({})
        self.assertIn("first night", result.lower())

    def test_tool_name(self):
        tool = build_night_summary_tool([])
        self.assertEqual(tool.name, "get_all_turn_summaries")


if __name__ == "__main__":
    unittest.main()
