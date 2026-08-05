import random
from typing import Any, Callable, List, Optional, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from domain.entities import ChatMessage, Player
from infrastructure.memory_tools import build_night_summary_tool
from infrastructure.werewolf_tools import build_werewolf_tools

DEFAULT_MAX_TOOL_ITERATIONS = 4


class _PlanState(TypedDict, total=False):
    werewolf_names: List[str]
    candidate_names: List[str]
    plan: str
    messages: List[Any]
    tool_iterations: int
    decision: str


class LangGraphWerewolfStrategy:
    """Werewolves' night-kill decision via a bounded, multi-step
    LangGraph: plan -> gather intel (real tool calls) -> decide, with a
    safe fallback if the model's final answer doesn't clearly name a
    valid candidate.

    Unlike DiscussionCoordinator (a long-running, streamed, multi-agent
    conversation that genuinely needed threads/asyncio to feel alive),
    this is a single bounded operation - gather some facts, make one
    decision, return - which is exactly the shape LangGraph's StateGraph
    is built for: a handful of well-defined steps, invoked once, not a
    continuous real-time loop. A fresh graph is built per decision (via
    `build_werewolf_tools`) since the tools themselves are bound to that
    night's specific candidates and transcript.

    When `summary_store` and `game_id_getter` are both supplied, the
    toolset also gains get_all_turn_summaries - a recap of every
    completed round so far this game. This is deliberately wider than
    what the day-phase vote gets (see LangChainVoteStrategy, which is
    restricted to just the previous round): werewolves coordinating a
    kill benefit from the whole pattern of who's seemed suspicious
    across the game, not only what happened last round. Both params are
    optional and independent of the existing transcript-based tools, so
    omitting them (e.g. in tests, or the CLI) leaves behavior exactly as
    it was before this tool existed.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        rng: Optional[random.Random] = None,
        max_tool_iterations: int = DEFAULT_MAX_TOOL_ITERATIONS,
        summary_store=None,
        game_id_getter: Optional[Callable[[], str]] = None,
    ):
        self._llm = llm
        self._rng = rng or random.Random()
        self._max_tool_iterations = max_tool_iterations
        self._summary_store = summary_store
        self._game_id_getter = game_id_getter

    def choose_victim(
        self,
        werewolves: List[Player],
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        if not candidates:
            raise ValueError("No candidates to choose from.")
        if len(candidates) == 1:
            return candidates[0]  # nothing to actually decide

        graph = self._build_graph(candidates, transcript or [])
        try:
            result = graph.invoke(
                {
                    "werewolf_names": [w.name for w in werewolves],
                    "candidate_names": [c.name for c in candidates],
                    "tool_iterations": 0,
                }
            )
            decision_text = result.get("decision", "")
        except Exception:
            # Any failure in the planning graph (model error, malformed
            # tool call, etc.) degrades to a random pick rather than
            # crashing the game - a night still has to end with a victim.
            decision_text = ""

        for candidate in candidates:
            if candidate.name.lower() in decision_text.lower():
                return candidate
        return self._rng.choice(candidates)

    def _build_graph(self, candidates: List[Player], transcript: List[ChatMessage]):
        tools = build_werewolf_tools(candidates, transcript)
        if self._summary_store is not None and self._game_id_getter is not None:
            all_summaries = self._summary_store.load_all_summaries(
                self._game_id_getter()
            )
            tools = tools + [build_night_summary_tool(all_summaries)]
        llm_with_tools = self._llm.bind_tools(tools)
        tool_node = ToolNode(tools)
        max_iterations = self._max_tool_iterations

        def plan_node(state: _PlanState) -> dict:
            system = SystemMessage(
                content=(
                    f"You are secretly a werewolf, coordinating in private with your "
                    f"fellow werewolves ({', '.join(state['werewolf_names'])}) to choose "
                    f"tonight's victim from: {', '.join(state['candidate_names'])}. "
                    "In 1-2 short sentences, plan what information you should gather "
                    "before deciding - e.g. who has been vocal against you, or who has "
                    "voting history worth checking. Keep it brief."
                )
            )
            response = self._llm.invoke(
                [system, HumanMessage(content="What's your plan?")]
            )
            plan_text = getattr(response, "content", str(response))
            follow_up = HumanMessage(
                content=(
                    f"Your plan: {plan_text}\n\n"
                    "Now use the available tools to gather exactly the information "
                    "your plan calls for. Once you have enough, stop calling tools "
                    "and say you're ready to decide."
                )
            )
            return {"plan": plan_text, "messages": [system, follow_up]}

        def gather_intel_node(state: _PlanState) -> dict:
            response = llm_with_tools.invoke(state["messages"])
            iterations = state.get("tool_iterations", 0)
            if getattr(response, "tool_calls", None):
                iterations += 1
            return {
                "messages": state["messages"] + [response],
                "tool_iterations": iterations,
            }

        def route_after_gathering(state: _PlanState) -> str:
            last = state["messages"][-1]
            tool_calls = getattr(last, "tool_calls", None)
            if tool_calls and state.get("tool_iterations", 0) <= max_iterations:
                return "use_tools"
            return "decide"

        def use_tools_node(state: _PlanState) -> dict:
            result = tool_node.invoke({"messages": state["messages"]})
            return {"messages": state["messages"] + result["messages"]}

        def decide_node(state: _PlanState) -> dict:
            final_prompt = HumanMessage(
                content=(
                    "Based on everything above, name exactly ONE villager to "
                    f"eliminate tonight from: {', '.join(state['candidate_names'])}. "
                    "Reply with ONLY their name, nothing else."
                )
            )
            response = self._llm.invoke(state["messages"] + [final_prompt])
            return {"decision": getattr(response, "content", str(response))}

        graph = StateGraph(_PlanState)
        graph.add_node("plan", plan_node)
        graph.add_node("gather_intel", gather_intel_node)
        graph.add_node("use_tools", use_tools_node)
        graph.add_node("decide", decide_node)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "gather_intel")
        graph.add_conditional_edges(
            "gather_intel",
            route_after_gathering,
            {"use_tools": "use_tools", "decide": "decide"},
        )
        graph.add_edge("use_tools", "gather_intel")
        graph.add_edge("decide", END)

        return graph.compile()
