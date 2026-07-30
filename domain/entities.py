from dataclasses import dataclass, field
from time import time

from domain.enums import Role


@dataclass
class Player:
    """Core domain entity. Has no knowledge of how decisions are made
    (bot vs human) and no knowledge of I/O - that belongs to outer layers.
    """

    id: int
    name: str
    role: Role
    background: str = ""
    is_alive: bool = True

    def kill(self) -> None:
        self.is_alive = False

    def set_background(self, background: str) -> None:
        self.background = background

    @property
    def is_werewolf(self) -> bool:
        return self.role == Role.WEREWOLF

    def __str__(self) -> str:
        status = "alive" if self.is_alive else "dead"
        return f"{self.name} ({self.role.value}, {status})"


@dataclass
class ChatMessage:
    """One line in the day-phase discussion transcript. Pure data - carries
    no opinion about who produced it (bot vs human) or how it should be
    rendered; that belongs to the application/infrastructure layers.
    """

    speaker_name: str
    content: str
    timestamp: float = field(default_factory=time)
    is_human: bool = False
