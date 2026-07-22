import random
from typing import List, Optional

from domain.entities import ChatMessage, Player


class StubSpeakStrategy:
    """Canned-line discussion strategy. Lets the discussion loop, timer,
    and human-injection wiring be fully exercised and tested without a
    live LLM. Swap in LangChainSpeakStrategy for real gameplay.
    """

    _VILLAGER_LINES = [
        "I don't trust {target}, they've been too quiet.",
        "Has anyone else noticed {target} acting strange?",
        "I think we should focus our vote on {target}.",
        "I'm not convinced yet, let's hear more first.",
    ]
    _WEREWOLF_LINES = [
        "I really think {target} looks suspicious.",
        "Let's not jump to conclusions about anyone yet.",
        "I agree, {target} has been avoiding questions.",
    ]

    def __init__(self, rng: Optional[random.Random] = None):
        self._rng = rng or random.Random()

    async def speak(
        self,
        speaker: Player,
        transcript: List[ChatMessage],
        alive_players: List[Player],
    ) -> str:
        others = [p for p in alive_players if p.id != speaker.id]
        target = self._rng.choice(others).name if others else "someone"
        lines = self._WEREWOLF_LINES if speaker.is_werewolf else self._VILLAGER_LINES
        return self._rng.choice(lines).format(target=target)
