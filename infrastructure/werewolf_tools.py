from typing import List

from langchain_core.tools import tool

from domain.entities import ChatMessage, Player


def build_werewolf_tools(candidates: List[Player], transcript: List[ChatMessage]):
    """Builds a fresh set of tools bound to *this* night's specific
    candidates and transcript via closures - freshly built per decision
    rather than module-level singletons, since both change every round.

    Each tool is a deterministic lookup over the transcript, not a call
    to the LLM - the point is to ground the agent's reasoning in actual
    recorded game data (who voted for whom, who's been vocal) instead of
    letting it hallucinate a plausible-sounding but made-up justification.
    """
    candidate_names = {c.name for c in candidates}

    @tool
    def list_alive_villagers() -> str:
        """List the villagers currently alive and eligible to be tonight's target."""
        return ", ".join(sorted(candidate_names)) or "No villagers are alive."

    @tool
    def get_voting_history(player_name: str) -> str:
        """Return a summary of votes cast by or against the named player
        in past day-phase rounds, and whether they were ever voted out.
        Use this to check whether someone has been actively voting
        against werewolves, or has themselves come under suspicion.
        """
        if player_name not in candidate_names:
            return f"'{player_name}' is not a currently living villager."
        relevant = [
            m.content
            for m in transcript
            if m.speaker_name == "Game"
            and player_name in m.content
            and ("votes for" in m.content or "voted out" in m.content)
        ]
        if not relevant:
            return f"No recorded voting activity involving {player_name} yet."
        return "\n".join(relevant)

    @tool
    def check_suspicion_level(player_name: str) -> str:
        """Count how many times OTHER players have mentioned the named
        player during discussion chat (not game-system events). A higher
        count usually means the village suspects them - not necessarily
        that they're dangerous to the werewolves specifically.
        """
        if player_name not in candidate_names:
            return f"'{player_name}' is not a currently living villager."
        count = sum(
            1
            for m in transcript
            if m.speaker_name != "Game"
            and m.speaker_name != player_name
            and player_name.lower() in m.content.lower()
        )
        return f"{player_name} has been mentioned by other players {count} time(s)."

    @tool
    def get_most_active_speakers() -> str:
        """Rank living villagers by how many discussion messages they've
        sent. The most active, vocal players are often the ones leading
        the investigation against the werewolves, which can make them
        the more dangerous target to leave alive.
        """
        counts = {name: 0 for name in candidate_names}
        for m in transcript:
            if m.speaker_name in counts:
                counts[m.speaker_name] += 1
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        if not ranked or all(count == 0 for _, count in ranked):
            return "No discussion activity recorded yet."
        return "\n".join(f"{name}: {count} message(s)" for name, count in ranked)

    return [
        list_alive_villagers,
        get_voting_history,
        check_suspicion_level,
        get_most_active_speakers,
    ]
