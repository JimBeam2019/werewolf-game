import random
from typing import List

from domain.entities import Player
from domain.enums import Role
from domain.rules import werewolf_count_for


class GameSetupService:
    """Use case: turn a list of player names into a fully-assigned roster."""

    @staticmethod
    def create_players(
        names: List[str], rng: random.Random | None = None
    ) -> List[Player]:
        rng = rng or random.Random()
        num_werewolves = werewolf_count_for(len(names))
        roles = [Role.WEREWOLF] * num_werewolves + [Role.VILLAGER] * (
            len(names) - num_werewolves
        )
        rng.shuffle(roles)
        return [
            Player(id=index, name=name, role=role)
            for index, (name, role) in enumerate(zip(names, roles))
        ]
