import argparse
import random
import sys

from application.game_engine import GameEngine
from application.setup import GameSetupService
from domain.exceptions import GameError
from domain.rules import MAX_PLAYERS, MIN_PLAYERS
from infrastructure.mixed_strategies import (
    HumanVsBotVoteStrategy,
    HumanVsBotWerewolfStrategy,
)
from infrastructure.notifiers import ConsoleNotifier
from infrastructure.random_strategies import RandomVoteStrategy, RandomWerewolfStrategy

DEFAULT_BOT_NAMES = [
    "Alice",
    "Bob",
    "Carol",
    "Dave",
    "Eve",
    "Frank",
    "Grace",
    "Heidi",
]


def build_player_names(num_players: int, human_name: str | None) -> list[str]:
    names = list(DEFAULT_BOT_NAMES[:num_players])
    if human_name:
        names[0] = human_name
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Werewolf / Mafia game prototype")
    parser.add_argument(
        "--players",
        type=int,
        default=6,
        help=f"Number of players ({MIN_PLAYERS}-{MAX_PLAYERS}). Default: 6",
    )
    parser.add_argument(
        "--mode",
        choices=["simulate", "play"],
        default="simulate",
        help="'simulate' runs an all-bot game automatically; "
        "'play' lets you control one player against bots.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="You",
        help="Your player name when --mode play is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible simulations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        names = build_player_names(
            args.players, args.name if args.mode == "play" else None
        )
        players = GameSetupService.create_players(names, rng=rng)
    except GameError as exc:
        print(f"Setup error: {exc}")
        sys.exit(1)

    notifier = ConsoleNotifier()

    if args.mode == "play":
        human_player = players[0]
        notifier.notify(
            f"You are {human_player.name}. Your secret role is: {human_player.role.value}."
        )
        werewolf_strategy = HumanVsBotWerewolfStrategy(human_player_id=human_player.id)
        vote_strategy = HumanVsBotVoteStrategy(human_player_id=human_player.id)
    else:
        werewolf_strategy = RandomWerewolfStrategy(rng=rng)
        vote_strategy = RandomVoteStrategy(rng=rng)

    engine = GameEngine(
        players=players,
        human_id=players[0].id,
        werewolf_strategy=werewolf_strategy,
        vote_strategy=vote_strategy,
        notifier=notifier,
    )

    notifier.notify("Roster:")
    for p in players:
        # Don't reveal roles in play mode except to the human's own role above.
        label = p.role.value if args.mode == "simulate" else "?"
        notifier.notify(f"  {p.name}: {label}")

    engine.play()


if __name__ == "__main__":
    main()
