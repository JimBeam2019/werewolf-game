from typing import Dict, List, Optional

from langchain_core.language_models import BaseChatModel

from domain.entities import ChatMessage


def _describe_outcome(
    eliminated_name: Optional[str], eliminated_role: Optional[str], tied: bool
) -> str:
    if tied:
        return "The vote was tied, so no one was eliminated."
    if eliminated_name:
        return (
            f"{eliminated_name} was voted out and revealed to be a {eliminated_role}."
        )
    return "No one was eliminated this round."


class StubTurnSummaryStrategy:
    """Deterministic, template-based turn summary - no LLM involved.

    Used as LangChainSummaryStrategy's fallback when the model call fails
    for any reason, and lets the save-a-summary-after-each-day wiring be
    tested without a live model.
    """

    def summarize_turn(
        self,
        round_number: int,
        transcript: List[ChatMessage],
        eliminated_name: Optional[str],
        eliminated_role: Optional[str],
        votes: Dict[str, str],
        tied: bool,
    ) -> str:
        outcome = _describe_outcome(eliminated_name, eliminated_role, tied)
        return f"Round {round_number}: the village discussed and voted. {outcome}"


class LangChainSummaryStrategy:
    """Asks an LLM to write a 2-3 sentence recap of one completed turn -
    who accused/defended whom during discussion, who was voted out, and
    their revealed role - falling back to StubTurnSummaryStrategy's
    deterministic template if the model call fails for any reason
    (connection error, timeout, empty reply, etc.), so a summary is
    always produced and saved even if the model is briefly unavailable.
    """

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._fallback = StubTurnSummaryStrategy()

    def summarize_turn(
        self,
        round_number: int,
        transcript: List[ChatMessage],
        eliminated_name: Optional[str],
        eliminated_role: Optional[str],
        votes: Dict[str, str],
        tied: bool,
    ) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        conversation = (
            "\n".join(f"{m.speaker_name}: {m.content}" for m in transcript)
            if transcript
            else "(no discussion took place)"
        )
        vote_lines = "\n".join(
            f"{voter} voted for {target}" for voter, target in votes.items()
        )
        outcome = _describe_outcome(eliminated_name, eliminated_role, tied)

        system = SystemMessage(
            content=(
                "You write short, neutral recaps of a social-deduction game's "
                "turn, for other players to recall later. In EXACTLY 2-3 "
                "simple sentences, summarize: who accused or defended whom "
                "during the discussion, who was voted out, and what role "
                "they were revealed as. Do not speculate beyond what's given."
            )
        )
        human = HumanMessage(
            content=(
                f"Round {round_number} discussion:\n{conversation}\n\n"
                f"Votes cast:\n{vote_lines or '(no votes recorded)'}\n\n"
                f"Outcome: {outcome}"
            )
        )

        try:
            response = self._llm.invoke([system, human])
            summary = getattr(response, "content", "") or ""
            summary = summary.strip()
            if summary:
                return summary
        except Exception:
            pass

        return self._fallback.summarize_turn(
            round_number, transcript, eliminated_name, eliminated_role, votes, tied
        )
