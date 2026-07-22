from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from domain.entities import ChatMessage, Player


class LangChainSpeakStrategy:
    """Produces one discussion-phase message by prompting an LLM.

    Accepts any langchain_core BaseChatModel, so this works with a local
    ChatOllama instance (for ROCm-accelerated local inference) or any
    other LangChain-compatible chat model without code changes here -
    only the model passed in at construction time changes:

        from langchain_ollama import ChatOllama
        llm = ChatOllama(model="llama3.1:8b-instruct-q4_K_M")
        strategy = LangChainSpeakStrategy(llm)

    Async, using `ainvoke` rather than `invoke`: DiscussionCoordinator
    runs one independent loop per bot on the same event loop, so a
    blocking call here would stall every other bot's turn too while this
    one waits on the model.
    """

    def __init__(self, llm: BaseChatModel, max_history: int = 12):
        self._llm = llm
        self._max_history = max_history

    async def speak(
        self,
        speaker: Player,
        transcript: List[ChatMessage],
        alive_players: List[Player],
    ) -> str:
        others = ", ".join(p.name for p in alive_players if p.id != speaker.id)
        system = SystemMessage(
            content=(
                f"You are {speaker.name}, playing a game of Werewolf as a {speaker.role.value}. "
                f"Other living players: {others}. "
                "Discuss who you suspect is a werewolf in 1-2 short, natural sentences. "
                "Stay in character and never directly reveal your own role."
            )
        )
        recent = transcript[-self._max_history :]
        conversation = "\n".join(f"{m.speaker_name}: {m.content}" for m in recent)
        human = HumanMessage(content=conversation or "The discussion has just begun.")

        response = await self._llm.ainvoke([system, human])
        content = response.content if hasattr(response, "content") else str(response)
        return content.strip()  # type: ignore
