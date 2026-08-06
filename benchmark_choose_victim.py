import argparse
import os
import time
from typing import List

from langchain_openai import ChatOpenAI

from domain.entities import Player
from domain.enums import Role
from infrastructure.langgraph_werewolf_strategy import LangGraphWerewolfStrategy

DEFAULT_CANDIDATE_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
    "Ivy",
    "Jack",
]


def build_agent_llm(
    model: str, base_url: str, api_key: str, temperature: float
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_retries=1,
    )


def make_candidates(count: int) -> List[Player]:
    if count < 1:
        raise ValueError("Candidate count must be at least 1.")
    players = []
    for idx in range(count):
        name = DEFAULT_CANDIDATE_NAMES[idx % len(DEFAULT_CANDIDATE_NAMES)]
        role = Role.VILLAGER
        players.append(Player(id=idx, name=name, role=role))
    return players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark LangGraphWerewolfStrategy.choose_victim performance."
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of benchmark iterations."
    )
    parser.add_argument(
        "--candidates", type=int, default=3, help="Number of victim candidates."
    )
    parser.add_argument(
        "--model",
        default=os.getenv("AGENT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M"),
        help="LLM model name to benchmark.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
        help="Base URL for the OpenAI-compatible LLM endpoint.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("OPENAI_API_KEY", "EMPTY"),
        help="API key for the LLM endpoint. Use EMPTY for local OpenAI-compatible endpoints.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature for model generation.",
    )
    parser.add_argument(
        "--max-tool-iterations",
        type=int,
        default=4,
        help="Maximum tool iterations allowed by LangGraphWerewolfStrategy.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before timed iterations.",
    )
    return parser.parse_args()


def run_benchmark(
    iterations: int,
    candidate_count: int,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tool_iterations: int,
    warmup: int,
) -> None:
    llm = build_agent_llm(
        model=model, base_url=base_url, api_key=api_key, temperature=temperature
    )
    strategy = LangGraphWerewolfStrategy(llm, max_tool_iterations=max_tool_iterations)
    candidates = make_candidates(candidate_count)
    werewolves = [Player(id=999, name="Wolf", role=Role.WEREWOLF)]
    transcript = []

    print("Benchmark configuration:")
    print(f"  model: {model}")
    print(f"  base_url: {base_url}")
    print(f"  candidates: {candidate_count}")
    print(f"  iterations: {iterations}")
    print(f"  warmup: {warmup}")
    print(f"  max_tool_iterations: {max_tool_iterations}")
    print(f"  temperature: {temperature}")
    print()

    for i in range(warmup):
        print(f"Warmup run {i + 1}/{warmup}...", end=" ")
        start = time.perf_counter()
        strategy.choose_victim(werewolves, candidates, transcript)
        print(f"{time.perf_counter() - start:.3f}s")

    durations = []
    for i in range(iterations):
        print(f"Benchmark run {i + 1}/{iterations}...", end=" ")
        start = time.perf_counter()
        victim = strategy.choose_victim(werewolves, candidates, transcript)
        duration = time.perf_counter() - start
        durations.append(duration)
        print(f"{duration:.3f}s -> {victim.name}")

    total = sum(durations)
    avg = total / len(durations) if durations else 0.0
    print()
    print("Benchmark results:")
    print(f"  total time: {total:.3f}s")
    print(f"  average time: {avg:.3f}s")
    print(
        f"  runs per second: {len(durations) / total:.2f}"
        if total > 0
        else "  runs per second: n/a"
    )


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        iterations=args.iterations,
        candidate_count=args.candidates,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        max_tool_iterations=args.max_tool_iterations,
        warmup=args.warmup,
    )
