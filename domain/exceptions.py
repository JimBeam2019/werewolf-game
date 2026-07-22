class GameError(Exception):
    """Base class for all domain-level errors."""


class InvalidPlayerCountError(GameError):
    """Raised when the number of players violates game rules (5-8 players)."""


class GameAlreadyEndedError(GameError):
    """Raised when an action is attempted after the game has already concluded."""


class InvalidActionError(GameError):
    """Raised when a phase action is attempted at the wrong time or on an
    invalid target (e.g. voting for a dead player)."""
