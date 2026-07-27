import asyncio
import random
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from application.interfaces import AgentSpeakStrategy
from domain.entities import ChatMessage, Player


@dataclass
class DiscussionCoordinator:
    """Runs a timed, freeform discussion among the alive bot players before
    a vote.

    Each bot runs its OWN independent loop in a background thread's
    asyncio event loop, speaking on its own randomized cadence
    (`speak_delay_range`), rather than a single shared turn-based
    round robin driven by the UI's own render cadence.

    That distinction matters: an earlier version of this coordinator used
    a LangGraph state graph that produced exactly one message per
    `.tick()` call, and a Streamlit fragment called `.tick()` on its own
    timer. That coupled "how often a bot speaks" to "how often the UI
    happens to rerun" - which in practice caused a frozen-looking timer
    (anything outside the fragment didn't refresh) and bursts of extra
    bot messages (any *other* rerun, like the human sending a message,
    also triggered an extra tick). Giving each bot its own independent,
    real-time-paced loop removes that coupling at the source: bots keep
    talking on their own schedule regardless of how often - or rarely -
    the UI redraws. The UI's only job becomes periodically reading
    (thread-safely) and rendering whatever has accumulated so far.

    Thread-safety: `transcript` is written to from several places at
    once - one asyncio task per bot (all on the *background thread's*
    event loop) plus Streamlit's main thread (rendering, and
    `add_human_message`). All access goes through `self._lock`, a plain
    `threading.RLock`. This is safe despite Streamlit's well-known
    restriction against background threads calling `st.*` functions or
    assigning `st.session_state` keys directly - that restriction doesn't
    apply here, since the background thread never touches Streamlit at
    all. It only mutates an ordinary Python list that a session_state
    value happens to hold a reference to.
    """

    alive_players: List[Player]
    human_id: int
    speak_strategy: AgentSpeakStrategy
    budget_seconds: float = 45.0
    speak_delay_range: Tuple[float, float] = (3.0, 7.0)
    prior_history: List[ChatMessage] = field(default_factory=list)
    """Everything that happened in *earlier* rounds (previous days'
    discussions, night kills, day eliminations) - loaded from a
    GameMemoryStore and given to agents as read-only context. Kept
    separate from `transcript` (today's chat only) rather than merged in,
    so the UI's "today's discussion" panel and existing tests/behavior
    around `transcript` don't change - `full_context()` is what combines
    the two for anything that needs the complete picture.
    """

    transcript: List[ChatMessage] = field(default_factory=list)
    start_time: Optional[float] = None

    _lock: threading.RLock = field(
        default_factory=threading.RLock, repr=False, compare=False
    )
    _thread: Optional[threading.Thread] = field(default=None, repr=False, compare=False)
    _stop_event: Optional[threading.Event] = field(
        default=None, repr=False, compare=False
    )

    def start(self) -> None:
        """Begins the discussion in a background thread. Idempotent - a
        UI can call this on every rerun without spawning a second thread
        or resetting the clock.
        """
        with self._lock:
            if self.start_time is not None:
                return
            self.start_time = time.monotonic()
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
            self._thread.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        """Requests an early stop and waits (briefly) for the background
        thread to wind down. Mainly useful for tests that don't want to
        wait out the full real-time budget; production code can just let
        the budget expire naturally.
        """
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=join_timeout)

    @property
    def time_remaining(self) -> float:
        """Wall-clock seconds left in the budget, for display purposes.
        Purely a countdown - see `is_finished` for whether it's actually
        safe to move on.
        """
        with self._lock:
            if self.start_time is None:
                return self.budget_seconds
            return max(0.0, self.budget_seconds - (time.monotonic() - self.start_time))

    @property
    def is_finished(self) -> bool:
        """True once the background thread has actually stopped running -
        not merely once the wall-clock budget has elapsed. A bot's LLM
        call in flight when the budget expires is allowed to finish
        rather than being cut off mid-message, so the thread can still be
        winding down for a moment after `time_remaining` hits zero.
        """
        with self._lock:
            return self._thread is not None and not self._thread.is_alive()

    def add_human_message(self, speaker_name: str, content: str) -> None:
        """Injects a human message into the shared transcript immediately.
        Thread-safe: callable from Streamlit's main thread at any point
        while bots are independently speaking in the background.
        """
        with self._lock:
            self.transcript.append(
                ChatMessage(speaker_name=speaker_name, content=content, is_human=True)
            )

    def snapshot_transcript(self) -> List[ChatMessage]:
        """A thread-safe copy of *today's* transcript only - use this
        rather than iterating `self.transcript` directly from outside the
        lock, since a background task could be appending to it
        concurrently. See `full_context()` for today's chat plus every
        earlier round combined.
        """
        with self._lock:
            return list(self.transcript)

    def full_context(self) -> List[ChatMessage]:
        """Everything from earlier rounds plus today's chat so far, in
        chronological order. This is what speak/vote strategies should
        reason over if they want the complete picture rather than just
        today's conversation.
        """
        with self._lock:
            return list(self.prior_history) + list(self.transcript)

    def _append_bot_message(self, speaker_name: str, content: str) -> None:
        with self._lock:
            self.transcript.append(
                ChatMessage(speaker_name=speaker_name, content=content, is_human=False)
            )

    def _snapshot_for_agent(self) -> Tuple[List[ChatMessage], List[Player]]:
        with self._lock:
            return list(self.prior_history) + list(self.transcript), list(
                self.alive_players
            )

    def _run_event_loop(self) -> None:
        """Background thread entry point: hosts a dedicated asyncio event
        loop for the lifetime of the discussion.
        """
        try:
            asyncio.run(self._async_discussion())
        except Exception:
            # A crashed background thread must never take the whole
            # Streamlit process down with it. If something goes wrong
            # here, the discussion simply stops producing new messages;
            # the game continues once the budget elapses.
            pass

    async def _async_discussion(self) -> None:
        deadline = time.monotonic() + self.budget_seconds
        queue: "asyncio.Queue[Tuple[str, str]]" = asyncio.Queue()

        bot_players = [p for p in self.alive_players if p.id != self.human_id]
        agent_tasks = [
            asyncio.create_task(self._agent_loop(player, deadline, queue))
            for player in bot_players
        ]
        consumer_task = asyncio.create_task(self._consume(queue))

        await asyncio.gather(*agent_tasks, return_exceptions=True)

        # Give the queue a moment to drain any messages still in flight
        # from the last round of agent turns before shutting the consumer down.
        await asyncio.sleep(0.3)
        consumer_task.cancel()
        await asyncio.gather(consumer_task, return_exceptions=True)

    async def _agent_loop(
        self,
        player: Player,
        deadline: float,
        queue: "asyncio.Queue[Tuple[str, str]]",
    ) -> None:
        """One bot's independent speaking loop: snapshot the transcript,
        decide what to say, publish it, wait a random interval, repeat -
        entirely independent of every other bot's loop and of whatever
        the UI happens to be doing.
        """
        if self._stop_event:
            while time.monotonic() < deadline and not self._stop_event.is_set():
                if not player.is_alive:
                    break  # this project's rules don't kill mid-discussion today,
                    # but this stays correct if that ever changes - `player` is
                    # the same object instance shared with the rest of the game,
                    # so its `is_alive` flag reflects any in-place mutation.

                transcript, alive = self._snapshot_for_agent()

                try:
                    content = await self.speak_strategy.speak(player, transcript, alive)
                    await queue.put((player.name, content))
                except Exception as exc:
                    await queue.put(
                        ("System", f"{player.name} failed to respond: {exc}")
                    )

                delay = random.uniform(*self.speak_delay_range)
                remaining = deadline - time.monotonic()
                sleep_for = max(0.0, min(delay, remaining))
                try:
                    await asyncio.wait_for(self._wait_stop(), timeout=sleep_for)
                except asyncio.TimeoutError:
                    pass

    async def _wait_stop(self) -> None:
        # threading.Event has no async wait, so poll it briefly - cheap
        # and simple, and the polling interval is invisible next to the
        # multi-second speaking cadence.
        if self._stop_event:
            while not self._stop_event.is_set():
                await asyncio.sleep(0.1)

    async def _consume(self, queue: "asyncio.Queue[Tuple[str, str]]") -> None:
        while True:
            speaker_name, content = await queue.get()
            self._append_bot_message(speaker_name, content)
            queue.task_done()
