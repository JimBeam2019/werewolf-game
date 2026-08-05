import random
from typing import Callable, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain.agents import create_agent
from domain.entities import ChatMessage, Player
from infrastructure.memory_tools import build_vote_memory_tools
from infrastructure.turn_summary_store import TurnSummaryStore


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
    """Casts a day-phase vote via a tool-calling LLM agent, rather than
    baking the whole transcript directly into the prompt.

    Access here is a deliberate, real restriction, not just a prompt
    instruction to "only consider recent context": the voting agent can
    see *only* what it gets back from calling get_previous_turn_summary()
    (last round's recap - not the full game history) and
    get_recent_chat_messages() (the last 6 messages of *today's*
    discussion only). Anything beyond that literally isn't in its
    context, because it was never handed the raw transcript at all -
    only the tools that expose these two narrow slices of it.

    `transcript`, per the shared VoteDecisionStrategy protocol, is
    expected to be *today's* discussion only (not cross-round history -
    that's what get_previous_turn_summary is for); it's sliced down to
    the last 6 messages when building the tools.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        summary_store: TurnSummaryStore,
        game_id_getter: Callable[[], str],
        round_getter: Callable[[], int],
        rng: Optional[random.Random] = None,
    ):
        self._llm = llm
        self._summary_store = summary_store
        self._game_id_getter = game_id_getter
        self._round_getter = round_getter
        self._rng = rng or random.Random()

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        from langchain_core.messages import HumanMessage, SystemMessage

        previous_summary = self._summary_store.load_previous_summary(
            self._game_id_getter(), self._round_getter()
        )
        tools = build_vote_memory_tools(previous_summary, transcript or [])

        candidate_names = ", ".join(c.name for c in candidates)
        system = SystemMessage(
            content=(
                f"You are {voter.name}, playing Werewolf as a {voter.role.value}. "
                "Use the available tools to recall what happened last round and "
                "what's been said today, then vote to eliminate the player you "
                f"most suspect is a werewolf. Eligible candidates: {candidate_names}. "
                "Reply with ONLY the candidate's name, nothing else."
            )
        )
        human = HumanMessage(content="Decide who to vote for.")

        agent = create_agent(model=self._llm, system_prompt=system, tools=tools)
        try:
            response = agent.invoke({"messages": [human]})
            reply = response["messages"][-1].content
        except Exception:
            # Any failure talking to the model (connection error, timeout,
            # malformed response, etc.) degrades to a random vote rather
            # than propagating and taking down the whole day phase.
            return self._rng.choice(candidates)

        for candidate in candidates:
            if candidate.name.lower() in reply.lower():
                return candidate
        return self._rng.choice(candidates)
