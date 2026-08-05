from typing import List, Optional

from langchain_core.tools import tool

from domain.entities import ChatMessage


def build_vote_memory_tools(previous_summary: Optional[str], recent_messages: List[ChatMessage]):
    """Tools for the day-phase vote - deliberately narrow.

    A voting agent (werewolf or villager) can recall ONLY the
    immediately preceding round's summary, not the entire game's
    history, and ONLY the last 6 messages of *today's* discussion, not
    the full transcript. This is a real access restriction, not just a
    prompt suggestion: anything outside what these two tools return is
    simply not available to the agent - it was never included in its
    context at all.

    On round 1, `previous_summary` will naturally be None (there's no
    prior round to summarize yet) and `recent_messages` may be short or
    empty - both tools degrade gracefully to a plain "nothing yet"
    message rather than needing special-cased handling for the first turn.
    """
    summary_text = (
        previous_summary
        or "This is the first round - there is no previous turn to recall."
    )
    recent = recent_messages[-6:]
    recent_text = (
        "\n".join(f"{m.speaker_name}: {m.content}" for m in recent)
        if recent
        else "No messages have been sent yet today."
    )

    @tool
    def get_previous_turn_summary() -> str:
        """Recall a short summary of what happened last round - who was
        accused or defended, who was voted out, and their revealed role.
        Returns a placeholder message if this is the first round.
        """
        return summary_text

    @tool
    def get_recent_chat_messages() -> str:
        """Recall the last 6 messages from today's discussion chat."""
        return recent_text

    return [get_previous_turn_summary, get_recent_chat_messages]


def build_night_summary_tool(all_summaries: List[str]):
    """Tool for the werewolves' night victim-choosing - deliberately
    wider than the day-vote tools above: it recalls every round's
    summary so far this game, not just the previous one. Werewolves
    coordinating a kill benefit from the whole pattern of who's seemed
    suspicious across the game, not only what happened last round.
    """
    summaries_text = (
        "\n".join(f"Round {i + 1}: {s}" for i, s in enumerate(all_summaries))
        if all_summaries
        else "This is the first night - there are no past turns to recall."
    )

    @tool
    def get_all_turn_summaries() -> str:
        """Recall a summary of every completed round so far this game."""
        return summaries_text

    return get_all_turn_summaries
