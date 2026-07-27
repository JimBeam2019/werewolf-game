import operator
from typing import Annotated, List, TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from domain.entities import ChatMessage


class _MemoryState(TypedDict):
    # `operator.add` as the reducer is what turns this into real
    # checkpointed memory rather than a plain dict lookup: every
    # `.invoke()` call against a given thread_id only needs to supply the
    # *new* messages for this turn, and LangGraph's checkpointer merges
    # them onto whatever it already persisted for that thread_id and
    # saves the result - the same pattern LangGraph's own chatbot memory
    # tutorials use for message history.
    #
    # Stored as plain dicts (not ChatMessage instances) so the
    # checkpointer's msgpack serialization never has to encode an
    # arbitrary dataclass - LangGraph only guarantees msgpack support for
    # built-in/registered types, and warns that pickling unregistered
    # ones may be rejected outright in a future version.
    history: Annotated[List[dict], operator.add]


def _record_node(state: _MemoryState) -> _MemoryState:
    # No-op passthrough. The actual "remembering" happens entirely via
    # the `history` field's `operator.add` reducer above, applied by the
    # checkpointer when this node's output is merged into the persisted
    # state for this thread_id - there's nothing left to do here.
    return {}


def _build_memory_graph():
    builder = StateGraph(_MemoryState)
    builder.add_node("record", _record_node)
    builder.set_entry_point("record")
    builder.add_edge("record", END)
    return builder.compile(checkpointer=MemorySaver())


def _to_dict(message: ChatMessage) -> dict:
    return {
        "speaker_name": message.speaker_name,
        "content": message.content,
        "timestamp": message.timestamp,
        "is_human": message.is_human,
    }


def _from_dict(data: dict) -> ChatMessage:
    return ChatMessage(
        speaker_name=data["speaker_name"],
        content=data["content"],
        timestamp=data["timestamp"],
        is_human=data["is_human"],
    )


class LangGraphMemoryStore:
    """Cross-round game memory backed by a LangGraph checkpointer.

    Each game gets its own `game_id`, used as the checkpointer's
    `thread_id`. Every day's discussion transcript and every game event
    (night kills, day eliminations) gets appended here once that round
    ends, and the *next* round's DiscussionCoordinator loads it back out
    as read-only prior context - giving agents real memory of everything
    that happened earlier in the game, not just the current round.

    This is in-process memory only (matching this project's "Python
    memory only" storage choice) - MemorySaver keeps everything in a
    plain dict keyed by thread_id for the lifetime of this object; it
    doesn't persist across a process restart. Swapping in a durable
    checkpointer (e.g. SqliteSaver) later would need no changes outside
    this file, since callers only depend on the GameMemoryStore protocol.
    """

    def __init__(self):
        self._graph = _build_memory_graph()

    def load_history(self, game_id: str) -> List[ChatMessage]:
        config = {"configurable": {"thread_id": game_id}}
        snapshot = self._graph.get_state(config)
        if not snapshot or not snapshot.values:
            return []
        return [_from_dict(d) for d in snapshot.values.get("history", [])]

    def append_and_save(
        self, game_id: str, new_messages: List[ChatMessage]
    ) -> List[ChatMessage]:
        if not new_messages:
            return self.load_history(game_id)
        config = {"configurable": {"thread_id": game_id}}
        result = self._graph.invoke(
            {"history": [_to_dict(m) for m in new_messages]}, config=config
        )
        return [_from_dict(d) for d in result["history"]]
