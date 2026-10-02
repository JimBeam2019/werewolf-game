import re

from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from domain.entities import ChatMessage, Player

_MAX_SENTENCES = 2
_MAX_WORDS = 15  # matches the "under 15 words" rule stated in the prompt below

# Stock formal openers a small local model reaches for even when told not
# to. `_shorten()` only guards against *length*; a model can easily stay
# under the word cap while still opening with one of these, so they get
# stripped outright rather than relying on the prompt alone to suppress
# them - the same "don't just ask nicely, back it up mechanically" logic
# as the length cap below.
_FORMAL_OPENERS = [
    "i believe that",
    "i believe",
    "in my opinion,",
    "in my opinion",
    "it appears that",
    "it seems that",
    "based on my observations,",
    "based on my observations",
    "i would say that",
    "i would say",
    "i think that",
    "as someone who has been paying close attention,",
    "well, in my opinion,",
]


def _strip_formal_openers(text: str) -> str:
    # Loop rather than a single pass: models often stack these ("Based on
    # my observations, it appears that Carol...") so stripping only the
    # first opener would still leave a second one exposed at the start.
    stripped_anything = False
    for _ in range(len(_FORMAL_OPENERS)):
        lowered = text.lower()
        matched = False
        for opener in _FORMAL_OPENERS:
            if lowered.startswith(opener):
                text = text[len(opener) :].lstrip(" ,")
                matched = True
                stripped_anything = True
                break
        if not matched:
            break
    if stripped_anything and text:
        # Re-capitalize what's now the first letter, since stripping an
        # opener can leave a lowercase word at the start. Text that never
        # had an opener stripped is left exactly as the model wrote it -
        # an intentionally-casual lowercase "nah..." shouldn't get
        # capitalized into something stiffer than what it started as.
        text = text[:1].upper() + text[1:]
    return text


def _shorten(text: str) -> str:
    """Backstop against verbose, formal output. Prompting alone isn't
    reliable for this, especially with smaller/quantized local models,
    which often ignore both length and tone instructions entirely - so
    this mechanically strips common formal openers and caps the result at
    a couple of sentences and a word count regardless of what the model
    actually produced.
    """
    text = text.strip().strip('"')
    text = _strip_formal_openers(text)

    # Keep at most the first couple of sentences.
    sentences: List[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch in ".!?":
            sentences.append(current.strip())
            current = ""
            if len(sentences) >= _MAX_SENTENCES:
                break
    if current.strip() and not sentences:
        sentences.append(current.strip())
    shortened = " ".join(sentences) if sentences else text

    words = shortened.split()
    if len(words) > _MAX_WORDS:
        shortened = " ".join(words[:_MAX_WORDS]) + "..."
    return shortened


# def _strip_leading_speaker_label(content: str, speaker_name: str) -> str:
#     """Small/quantized models often imitate the 'Name: message' pattern
#     they're shown in the transcript, prefixing their own name onto their
#     reply. Since we *also* prepend the speaker's name when rendering, this
#     produces "Frank: Frank: message". Strip it deterministically rather
#     than relying on the model reliably following an instruction not to.
#     """
#     pattern = rf"^\s*{re.escape(speaker_name)}\s*:\s*"
#     return re.sub(pattern, "", content, count=1, flags=re.IGNORECASE)


_NAME_PREFIX_PATTERN = re.compile(r"^\s*[\w'\- ]{1,20}:\s*")


def _clean_agent_reply(raw_content: str) -> str:
    """Small/quantized models often don't stop after their own single
    line - they keep predicting the 'Name: message' transcript format
    they were shown, sometimes generating several more turns, including
    ones attributed to OTHER characters. Two backstops, since prompting
    alone isn't reliable:
      1. Keep only the first generated line - anything after a newline
         is the model continuing past what was actually asked for.
      2. Strip any leading "Name:" label from that line - not just the
         current speaker's own name, since the model may mislabel its
         line with a different character's name entirely.
    """
    stripped = raw_content.strip()
    first_line = stripped.splitlines()[0] if stripped else ""
    return _NAME_PREFIX_PATTERN.sub("", first_line, count=1)


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
        known_facts: str = "",
    ) -> str:
        others = ", ".join(p.name for p in alive_players if p.id != speaker.id)
        system = SystemMessage(
            content=(
                f"You're {speaker.name}, playing Werewolf as a {speaker.role.value}, "
                "chatting live with the other players. Other players still in: "
                f"{others}.\n\n"
                "Your background knowledge:\n"
                f"{speaker.background}\n\n"
                "Known facts so far - only state a player's role if it's listed "
                "here as revealed. Never claim to know an unrevealed player's "
                f"role, even if you suspect it:\n{known_facts}\n\n"
                "Talk like a real person texting in a group chat during a fast-paced "
                "game - casual, short, a little blunt. NOT like a formal writer.\n\n"
                "Hard rules:\n"
                "- Speak in first person as yourself - use 'I', never refer to yourself as "
                f"'{speaker.name}' in the third person.\n"
                "- ONE short sentence. Under 15 words.\n"
                '- No greetings, no restating the situation, no "I believe that..." '
                'or "In my opinion..." openers.\n'
                "- Contractions are good: dunno, gonna, yeah, nah.\n"
                "- Never reveal your own role.\n\n"
                f"- You may only suspect or accuse players from this list: {others}. "
                "- Never suspect, accuse, or cast doubt on yourself."
                "Good examples: \"nah I don't buy it, Bob's been dodging questions\" / "
                '"wait why\'s everyone so quiet" / "still think it\'s Carol tbh" / '
                '"same, sketchy energy from Dave"\n'
                "Bad examples (too long/formal - never write like this): "
                '"I believe that we should carefully consider the behavior of..." / '
                '"Based on my observations, it appears that..."'
            )
        )
        recent = transcript[-self._max_history :]
        conversation = "\n".join(f"{m.speaker_name}: {m.content}" for m in recent)
        human = HumanMessage(content=conversation or "The discussion has just begun.")

        response = await self._llm.ainvoke([system, human], stop=["\n"])
        content = response.content if hasattr(response, "content") else str(response)
        message = _shorten(content)  # type: ignore
        return _clean_agent_reply(message)
