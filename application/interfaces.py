from typing import List, Optional, Protocol

from domain.entities import ChatMessage, Player


class WerewolfDecisionStrategy(Protocol):
    """Decides which villager the werewolves kill on a given night.
    Implementations may be random (bots) or prompt a human.

    `transcript` is prior cross-round history (discussion, kills, votes) -
    may be empty on night 1, when there's nothing to reason about yet.
    Bot implementations aren't required to read it - a naive strategy can
    ignore the parameter entirely - but a planning strategy should use it
    to identify real threats (who's been vocal, who's accused the
    werewolves) rather than picking blindly.
    """

    def choose_victim(
        self,
        werewolves: List[Player],
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
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


class GameMemoryStore(Protocol):
    """Persists the full, cross-round game history (discussion messages
    and game events like kills/eliminations) under a `game_id`, so a new
    day's discussion and vote strategies can see everything that happened
    on *earlier* days - not just the current round's fresh transcript.

    Without this, every new day's DiscussionCoordinator starts from an
    empty transcript and agents have no memory of who was killed, who
    they accused yesterday, or what anyone else said in prior rounds.
    """

    def load_history(self, game_id: str) -> List[ChatMessage]: ...

    def append_and_save(
        self, game_id: str, new_messages: List[ChatMessage]
    ) -> List[ChatMessage]: ...


class BackgroundKnowledgeProvider(Protocol):
    """
    Provide agent player background knowledge
    """

    def get_background(self, player: Player) -> str: ...


class TurnSummaryStrategy(Protocol):
    """Produces a short (2-3 sentence) natural-language recap of one
    completed day/night turn - who accused/defended whom, who was voted
    out, and what role they were revealed as. Saved to a TurnSummaryStore
    so later rounds can recall what happened without needing the entire
    raw transcript.
    """

    def summarize_turn(
        self,
        round_number: int,
        transcript: List[ChatMessage],
        eliminated_name: Optional[str],
        eliminated_role: Optional[str],
        votes: dict,
        tied: bool,
    ) -> str: ...


class TurnSummaryStore(Protocol):
    """Persists one short summary per completed round, under a game_id.

    Deliberately separate from GameMemoryStore, which holds the full raw
    transcript/event history: summaries exist specifically to back the
    *restricted*-access tools that only let agents recall the previous
    round (during a day-phase vote) or every round so far (during the
    werewolves' night choice) - a compact recap, not the full unabridged
    history GameMemoryStore provides.
    """

    def save_summary(self, game_id: str, round_number: int, summary: str) -> None: ...

    def load_all_summaries(self, game_id: str) -> List[str]: ...

    def load_previous_summary(
        self, game_id: str, round_number: int
    ) -> Optional[str]: ...

    def clear(self, game_id: str) -> None: ...
