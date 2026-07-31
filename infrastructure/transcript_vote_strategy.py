import random
from typing import List, Optional

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from domain.entities import ChatMessage, Player


class StubTranscriptVoteStrategy:
    """Naive transcript-aware vote: counts how many times each candidate's
    name was mentioned during the discussion and votes for whoever was
    mentioned most, falling back to random if no one was mentioned or
    there's no transcript at all.

    This exists so the "votes are informed by discussion" wiring can be
    tested without a live LLM - LangChainVoteStrategy is the real
    replacement, reasoning over the transcript instead of just counting
    name mentions.
    """

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        if not transcript:
            return self._rng.choice(candidates)

        mention_counts = {c.id: 0 for c in candidates}
        for message in transcript:
            lowered = message.content.lower()
            for candidate in candidates:
                if candidate.name.lower() in lowered:
                    mention_counts[candidate.id] += 1

        top_id = max(mention_counts, key=mention_counts.get)  # type: ignore
        if mention_counts[top_id] == 0:
            return self._rng.choice(candidates)
        return next(c for c in candidates if c.id == top_id)


class LangChainVoteStrategy:
    """Casts a day-phase vote by asking an LLM to reason over the
    discussion transcript, rather than voting randomly or by mention-count.

    Falls back to a random eligible candidate if the model's reply doesn't
    clearly match a candidate's name - this keeps the game from breaking
    if a smaller/quantized local model produces a slightly off-format
    answer.
    """

    def __init__(self, llm, rng: Optional[random.Random] = None):
        self._llm = llm
        self._rng = rng or random.Random()

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        from langchain_core.messages import HumanMessage, SystemMessage

        candidate_names = ", ".join(c.name for c in candidates)
        conversation = (
            "\n".join(f"{m.speaker_name}: {m.content}" for m in transcript)
            if transcript
            else "(no discussion took place)"
        )
        system = SystemMessage(
            content=(
                f"You are {voter.name}, playing Werewolf as a {voter.role.value}. "
                f"Based on the discussion below, vote to eliminate the player you "
                f"most suspect is a werewolf. Eligible candidates: {candidate_names}. "
                "Reply with ONLY the candidate's name, nothing else."
            )
        )
        human = HumanMessage(content=conversation)

        agent = create_agent(
            model=self._llm,
            system_prompt=system,
            checkpointer=InMemorySaver(),
        )
        response = agent.invoke({"messages": [human]})
        reply = response["messages"][-1].content

        # response = self._llm.invoke([system, human])
        # reply = (
        #     response.content if hasattr(response, "content") else str(response)
        # ).strip()

        for candidate in candidates:
            if candidate.name.lower() in reply.lower():
                return candidate
        return self._rng.choice(candidates)
