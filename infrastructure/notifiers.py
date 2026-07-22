class ConsoleNotifier:
    """Prints game events to stdout. Swap for a LogNotifier, WebSocketNotifier,
    etc. without touching the application layer.
    """

    def notify(self, message: str) -> None:
        print(message)


class SilentNotifier:
    """Useful for tests or headless simulations where output isn't needed."""

    def notify(self, message: str) -> None:
        pass
