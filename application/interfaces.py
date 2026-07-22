from typing import List, Optional, Protocol

from domain.entities import ChatMessage, Player


class WerewolfDecisionStrategy(Protocol):
    """Decides which villager the werewolves kill on a given night.
    Implementations may be random (bots) or prompt a human.
    """

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player]
    ) -> Player: ...


class VoteDecisionStrategy(Protocol):
    """Decides which candidate a single voter casts their day-phase vote for.

    `transcript` is the discussion that preceded the vote (may be empty if
    no discussion phase ran). Bot implementations aren't required to read
    it - a naive strategy can ignore the parameter entirely - but an
    LLM-backed strategy should use it to ground its vote in what was
    actually said, rather than voting randomly.
    """

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player: ...


class AgentSpeakStrategy(Protocol):
    """Produces one discussion-phase message for a given speaker, given the
    transcript so far and who's still alive. Implementations may call an
    LLM (LangChain) or return canned/random lines for testing.

    Async because each bot runs its own independent loop concurrently
    (see application/discussion.py) - a blocking call here would stall
    every other bot's turn too, since they'd all share one event loop.
    """

    async def speak(
        self,
        speaker: Player,
        transcript: List[ChatMessage],
        alive_players: List[Player],
    ) -> str: ...


class Notifier(Protocol):
    """Sink for game events. Implementations may print to console, log to
    a file, push to a websocket, etc. The application layer never assumes
    a specific output medium.
    """

    def notify(self, message: str) -> None: ...
