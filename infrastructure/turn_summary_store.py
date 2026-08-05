from typing import Dict, List, Optional


class TurnSummaryStore:
    """In-memory, per-game store of short turn-by-turn summaries.

    Deliberately NOT built on LangGraph's checkpointer like
    LangGraphMemoryStore: that machinery exists there to handle several
    concurrent writers merging onto the same thread_id. Summaries are
    generated and saved one at a time, from the main thread, once per
    completed round - there's no concurrent-write problem to solve, so a
    plain dict keeps this simple and easy to verify correct instead of
    pulling in checkpointing complexity this use case doesn't need.
    """

    def __init__(self) -> None:
        self._summaries: Dict[str, Dict[int, str]] = {}

    def save_summary(self, game_id: str, round_number: int, summary: str) -> None:
        self._summaries.setdefault(game_id, {})[round_number] = summary

    def load_all_summaries(self, game_id: str) -> List[str]:
        """Every summary saved for this game, in round order."""
        rounds = self._summaries.get(game_id, {})
        return [rounds[r] for r in sorted(rounds)]

    def load_previous_summary(self, game_id: str, round_number: int) -> Optional[str]:
        """The immediately preceding round's summary, or None if there
        isn't one yet - either because `round_number` is 1 (no round
        before it exists) or because that round's summary hasn't been
        saved for some other reason. Callers should treat None as "no
        prior context available" rather than an error.
        """
        rounds = self._summaries.get(game_id, {})
        return rounds.get(round_number - 1)

    def clear(self, game_id: str) -> None:
        """Removes every summary recorded for this specific game_id,
        without touching any other game's data that might happen to
        share the same store instance.
        """
        self._summaries.pop(game_id, None)

    def clear_all(self) -> None:
        """Wipes every game's summaries. Mainly useful for tests; normal
        game-to-game isolation is handled by each new game getting its
        own fresh TurnSummaryStore instance (see streamlit_app.py), not
        by calling this.
        """
        self._summaries.clear()
