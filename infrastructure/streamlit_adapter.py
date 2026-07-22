from typing import Callable, List, Optional

import streamlit as st

from application.interfaces import VoteDecisionStrategy
from domain.entities import ChatMessage, Player
from infrastructure.random_strategies import RandomVoteStrategy, RandomWerewolfStrategy


class PendingHumanDecision(Exception):
    """Raised by a Streamlit-backed strategy when it needs a real button
    click from the human player before the engine can proceed.

    Streamlit reruns the whole script top-to-bottom on every interaction,
    so GameEngine can't just call input() and block like the console
    version does. Instead: the strategy checks st.session_state for a
    previously-recorded answer; if there isn't one yet, it raises this
    exception. The app layer catches it, renders buttons for `candidates`,
    and re-invokes the engine once a click has stored an answer under
    `key`.
    """

    def __init__(self, key: str, prompt: str, candidates: List[Player]):
        super().__init__(prompt)
        self.key = key
        self.prompt = prompt
        self.candidates = candidates


class BufferingNotifier:
    """Collects notifications in memory instead of printing them.

    A phase that gets interrupted by a PendingHumanDecision only has its
    buffer discarded (not committed) by the app layer, so retried/aborted
    attempts never leave partial or duplicate lines in the on-screen log.
    """

    def __init__(self):
        self.buffer: List[str] = []

    def notify(self, message: str) -> None:
        self.buffer.append(message)


class StreamlitWerewolfStrategy:
    """Night-kill decision. Bot werewolves decide instantly; if the human
    player is among the surviving werewolves, this pauses the engine via
    PendingHumanDecision until a button click supplies an answer.
    """

    def __init__(self, human_player_id: Optional[int], round_getter: Callable[[], int]):
        self._human_player_id = human_player_id
        self._bot = RandomWerewolfStrategy()
        self._round_getter = round_getter

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player]
    ) -> Player:
        human_is_werewolf = self._human_player_id is not None and any(
            w.id == self._human_player_id for w in werewolves
        )
        if not human_is_werewolf:
            return self._bot.choose_victim(werewolves, candidates)

        key = f"night_kill_round_{self._round_getter()}"
        answer_id = st.session_state.get(key)
        if answer_id is None:
            raise PendingHumanDecision(
                key=key,
                prompt="Choose a villager to eliminate tonight:",
                candidates=candidates,
            )
        return next(c for c in candidates if c.id == answer_id)


class StreamlitVoteStrategy:
    """Day-phase vote. Bots vote instantly (via `bot_strategy`, defaulting
    to random - swap in a transcript-aware/LLM strategy to have bots
    reason from the discussion); the human's own vote pauses the engine
    until a button click supplies it.
    """

    def __init__(
        self,
        human_player_id: Optional[int],
        round_getter: Callable[[], int],
        bot_strategy: Optional[VoteDecisionStrategy] = None,
    ):
        self._human_player_id = human_player_id
        self._bot = bot_strategy or RandomVoteStrategy()
        self._round_getter = round_getter

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        if self._human_player_id is None or voter.id != self._human_player_id:
            return self._bot.cast_vote(voter, candidates, transcript)

        key = f"day_vote_round_{self._round_getter()}"
        answer_id = st.session_state.get(key)
        if answer_id is None:
            raise PendingHumanDecision(
                key=key,
                prompt="Who do you vote to eliminate?",
                candidates=candidates,
            )
        return next(c for c in candidates if c.id == answer_id)
