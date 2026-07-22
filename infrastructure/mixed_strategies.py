from typing import List, Optional

from domain.entities import ChatMessage, Player
from infrastructure.console_strategies import (
    ConsoleVoteStrategy,
    ConsoleWerewolfStrategy,
)
from infrastructure.random_strategies import RandomVoteStrategy, RandomWerewolfStrategy


class HumanVsBotWerewolfStrategy:
    """Delegates to a human prompt if the human player is one of the
    surviving werewolves, otherwise falls back to the bot strategy.
    Lets a single human play against otherwise-simulated opponents.
    """

    def __init__(self, human_player_id: Optional[int]):
        self._human_player_id = human_player_id
        self._console = ConsoleWerewolfStrategy()
        self._bot = RandomWerewolfStrategy()

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player]
    ) -> Player:
        if self._human_player_id is not None and any(
            w.id == self._human_player_id for w in werewolves
        ):
            return self._console.choose_victim(werewolves, candidates)
        return self._bot.choose_victim(werewolves, candidates)


class HumanVsBotVoteStrategy:
    """Delegates to a human prompt only when the current voter is the
    human player; every other (bot) voter votes automatically.
    """

    def __init__(self, human_player_id: Optional[int]):
        self._human_player_id = human_player_id
        self._console = ConsoleVoteStrategy()
        self._bot = RandomVoteStrategy()

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        if self._human_player_id is not None and voter.id == self._human_player_id:
            return self._console.cast_vote(voter, candidates, transcript)
        return self._bot.cast_vote(voter, candidates, transcript)
