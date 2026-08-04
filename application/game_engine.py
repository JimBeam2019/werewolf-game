import asyncio
import inspect

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

from application.interfaces import (
    Notifier,
    VoteDecisionStrategy,
    WerewolfDecisionStrategy,
    BackgroundKnowledgeProvider,
)
from domain.entities import ChatMessage, Player
from domain.enums import GamePhase, GameResult, Role
from domain.exceptions import GameAlreadyEndedError
from domain.rules import WinConditionService


@dataclass
class NightResult:
    round_number: int
    victim: Player


@dataclass
class DayResult:
    round_number: int
    votes: Dict[str, str]  # voter name -> candidate name, for transparency/logging
    eliminated: Optional[Player]
    tied: bool


@dataclass
class GameEngine:
    """Orchestrates the night/day cycle. Depends only on domain entities
    and application-level interfaces - no concrete strategy or I/O
    implementation is referenced here (dependency inversion).
    """

    players: List[Player]
    human_id: int
    background_provider: BackgroundKnowledgeProvider
    werewolf_strategy: WerewolfDecisionStrategy
    vote_strategy: VoteDecisionStrategy
    notifier: Notifier
    phase: GamePhase = GamePhase.NIGHT
    round_number: int = 1
    result: GameResult = GameResult.ONGOING
    history: List[object] = field(default_factory=list)

    async def _run_async(
        self, func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        if inspect.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        return await asyncio.to_thread(func, *args, **kwargs)

    async def _get_background(self, player: Player) -> Any:
        return await self._run_async(self.background_provider.get_background, player)

    async def set_player_backgrounds(self) -> None:
        tasks = []
        for player in self.players:
            if player.id != self.human_id:
                tasks.append(
                    (player, asyncio.create_task(self._get_background(player)))
                )

        for player, task in tasks:
            background = await task
            player.set_background(background)

    def alive_players(self) -> List[Player]:
        return [p for p in self.players if p.is_alive]

    def alive_werewolves(self) -> List[Player]:
        return [p for p in self.alive_players() if p.role == Role.WEREWOLF]

    def alive_villagers(self) -> List[Player]:
        return [p for p in self.alive_players() if p.role == Role.VILLAGER]

    def _ensure_ongoing(self) -> None:
        if self.result != GameResult.ONGOING:
            raise GameAlreadyEndedError("The game has already ended.")

    def run_night_phase(self) -> NightResult:
        """Werewolves collectively choose exactly one villager to kill."""
        self._ensure_ongoing()
        self.notifier.notify(f"\n--- Night {self.round_number} ---")

        werewolves = self.alive_werewolves()
        candidates = self.alive_villagers()
        victim = self.werewolf_strategy.choose_victim(werewolves, candidates)
        victim.kill()
        self.notifier.notify(f"{victim.name} was found dead this morning.")

        result_entry = NightResult(round_number=self.round_number, victim=victim)
        self.history.append(result_entry)

        self.result = WinConditionService.evaluate(self.players)
        self.phase = (
            GamePhase.ENDED if self.result != GameResult.ONGOING else GamePhase.DAY
        )
        return result_entry

    def run_day_phase(
        self, transcript: Optional[List[ChatMessage]] = None
    ) -> DayResult:
        """All surviving players vote on who they believe is a werewolf.
        The candidate with the most votes is eliminated; ties result in
        no elimination that round.

        `transcript` is the discussion that preceded this vote, if any -
        it's passed straight through to the vote strategy so an
        LLM-backed strategy can ground its vote in what was actually said.
        """
        self._ensure_ongoing()
        self.notifier.notify(f"\n--- Day {self.round_number} ---")

        voters = self.alive_players()
        tally: Dict[int, int] = {p.id: 0 for p in voters}
        votes_cast: Dict[str, str] = {}

        for voter in voters:
            eligible_candidates = [c for c in voters if c.id != voter.id]
            choice = self.vote_strategy.cast_vote(
                voter, eligible_candidates, transcript
            )
            tally[choice.id] += 1
            votes_cast[voter.name] = choice.name
            self.notifier.notify(f"{voter.name} votes for {choice.name}")

        max_votes = max(tally.values())
        top_candidate_ids = [pid for pid, count in tally.items() if count == max_votes]

        eliminated: Optional[Player] = None
        tied = len(top_candidate_ids) > 1
        if not tied:
            eliminated = next(p for p in self.players if p.id == top_candidate_ids[0])
            eliminated.kill()
            self.notifier.notify(
                f"{eliminated.name} was voted out by the village. "
                f"They were a {eliminated.role.value}."
            )
        else:
            self.notifier.notify("The vote was tied. No one is eliminated today.")

        result_entry = DayResult(
            round_number=self.round_number,
            votes=votes_cast,
            eliminated=eliminated,
            tied=tied,
        )
        self.history.append(result_entry)

        self.result = WinConditionService.evaluate(self.players)
        self.round_number += 1
        self.phase = (
            GamePhase.ENDED if self.result != GameResult.ONGOING else GamePhase.NIGHT
        )
        return result_entry

    def play(self) -> GameResult:
        """Runs full night/day cycles until a faction wins."""
        while self.result == GameResult.ONGOING:
            self.run_night_phase()
            if self.result != GameResult.ONGOING:
                break
            self.run_day_phase()

        self.notifier.notify(f"\n=== Game Over: {self.result.value.upper()} ===")
        for p in self.players:
            self.notifier.notify(f"  {p}")
        return self.result
