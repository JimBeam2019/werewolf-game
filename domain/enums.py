from enum import Enum


class Role(Enum):
    VILLAGER = "villager"
    WEREWOLF = "werewolf"


class GamePhase(Enum):
    NIGHT = "night"
    DAY = "day"
    ENDED = "ended"


class GameResult(Enum):
    ONGOING = "ongoing"
    VILLAGERS_WIN = "villagers_win"
    WEREWOLVES_WIN = "werewolves_win"
