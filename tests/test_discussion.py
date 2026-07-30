import time
import unittest

from application.discussion import DiscussionCoordinator
from domain.entities import Player
from domain.enums import Role
from infrastructure.stub_speak_strategy import StubSpeakStrategy

HUMAN_ID = 99  # not present in make_players() - a distinct human seat

# Small, fast values so these tests don't need to wait out anything close
# to a real 45-second discussion window.
FAST_BUDGET = 1.2
FAST_DELAY_RANGE = (0.05, 0.15)


def make_players():
    return [
        Player(id=0, name="Alice", role=Role.VILLAGER),
        Player(id=1, name="Bob", role=Role.VILLAGER),
        Player(id=2, name="Carol", role=Role.WEREWOLF),
    ]


def make_coordinator(**overrides):
    defaults = dict(
        alive_players=make_players(),
        human_id=HUMAN_ID,
        speak_strategy=StubSpeakStrategy(),
        budget_seconds=FAST_BUDGET,
        speak_delay_range=FAST_DELAY_RANGE,
    )
    defaults.update(overrides)
    return DiscussionCoordinator(**defaults)


class TestDiscussionCoordinator(unittest.TestCase):
    def test_start_produces_messages_from_multiple_bots(self):
        coordinator = make_coordinator()
        coordinator.start()

        # Let the discussion run its natural (short) course rather than
        # forcing an early stop, so bots actually get a chance to speak -
        # calling stop() immediately after start() would race the
        # background thread's very first iteration.
        deadline = time.monotonic() + FAST_BUDGET + 2.0
        while not coordinator.is_finished and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertTrue(coordinator.is_finished)
        transcript = coordinator.snapshot_transcript()
        self.assertGreater(len(transcript), 0)

        speakers = {m.speaker_name for m in transcript}
        # With a 1.2s budget and a 0.05-0.15s speaking cadence, all three
        # bots should get at least one turn in - this is inherently timing
        # based, so allow at least two distinct speakers rather than
        # requiring all three, to avoid flakiness on a slow CI box.
        self.assertGreaterEqual(len(speakers), 2)

    def test_human_player_never_appears_as_a_bot_speaker(self):
        # Regression test: the discussion coordinator must never generate
        # a bot line "on behalf of" the human player, even if the human
        # is included in `alive_players` (as the real Player roster is).
        players = make_players() + [Player(id=HUMAN_ID, name="Jim", role=Role.VILLAGER)]
        coordinator = make_coordinator(alive_players=players)
        coordinator.start()

        deadline = time.monotonic() + FAST_BUDGET + 2.0
        while not coordinator.is_finished and time.monotonic() < deadline:
            time.sleep(0.05)

        speakers = {m.speaker_name for m in coordinator.snapshot_transcript()}
        self.assertNotIn("Jim", speakers)

    def test_add_human_message_is_immediate_and_thread_safe(self):
        coordinator = make_coordinator(budget_seconds=100.0)  # won't finish on its own
        coordinator.start()
        try:
            coordinator.add_human_message("Jim", "I suspect Carol!")
            transcript = coordinator.snapshot_transcript()
            human_lines = [m for m in transcript if m.is_human]
            self.assertEqual(len(human_lines), 1)
            self.assertEqual(human_lines[0].content, "I suspect Carol!")
        finally:
            coordinator.stop(join_timeout=3.0)

    def test_stop_ends_the_discussion_early(self):
        coordinator = make_coordinator(budget_seconds=100.0)
        coordinator.start()
        self.assertFalse(coordinator.is_finished)

        coordinator.stop(join_timeout=3.0)
        self.assertTrue(coordinator.is_finished)

    def test_start_is_idempotent(self):
        coordinator = make_coordinator(budget_seconds=100.0)
        coordinator.start()
        first_start_time = coordinator.start_time
        thread_after_first_start = coordinator._thread  # noqa: SLF001

        time.sleep(0.05)
        coordinator.start()  # should be a no-op

        self.assertEqual(coordinator.start_time, first_start_time)
        self.assertIs(coordinator._thread, thread_after_first_start)  # noqa: SLF001
        coordinator.stop(join_timeout=3.0)

    def test_time_remaining_counts_down(self):
        coordinator = make_coordinator(budget_seconds=10.0)
        self.assertEqual(coordinator.time_remaining, 10.0)  # not started yet
        coordinator.start()
        time.sleep(0.2)
        self.assertLess(coordinator.time_remaining, 10.0)
        coordinator.stop(join_timeout=3.0)

    def test_no_bot_players_finishes_without_error(self):
        coordinator = make_coordinator(alive_players=[])
        coordinator.start()
        coordinator.stop(join_timeout=3.0)
        self.assertTrue(coordinator.is_finished)
        self.assertEqual(coordinator.snapshot_transcript(), [])

    def test_full_context_combines_prior_history_and_todays_transcript(self):
        from domain.entities import ChatMessage

        prior = [ChatMessage(speaker_name="Game", content="Bob was killed.")]
        coordinator = make_coordinator(prior_history=prior, budget_seconds=100.0)
        coordinator.start()
        try:
            coordinator.add_human_message("Jim", "hi")
            context = coordinator.full_context()
            self.assertEqual(len(context), 2)
            self.assertEqual(context[0].content, "Bob was killed.")
            self.assertEqual(context[1].content, "hi")
            # snapshot_transcript() stays scoped to today only
            self.assertEqual(len(coordinator.snapshot_transcript()), 1)
        finally:
            coordinator.stop(join_timeout=3.0)

    def test_agents_actually_see_prior_history_during_their_turn(self):
        from domain.entities import ChatMessage

        received = []

        class RecordingStrategy:
            async def speak(self, speaker, transcript, alive_players):
                received.append(list(transcript))
                return "ok"

        prior = [ChatMessage(speaker_name="Game", content="Dave was killed.")]
        coordinator = make_coordinator(
            speak_strategy=RecordingStrategy(),
            prior_history=prior,
            budget_seconds=1.0,
            speak_delay_range=(0.05, 0.1),
        )
        coordinator.start()
        deadline = time.monotonic() + 2.0
        while not coordinator.is_finished and time.monotonic() < deadline:
            time.sleep(0.05)

        self.assertTrue(received, "expected at least one agent turn to occur")
        first_turn_context = received[0]
        self.assertTrue(
            any(m.content == "Dave was killed." for m in first_turn_context),
            msg="Agents should see prior_history from the very first turn",
        )


if __name__ == "__main__":
    unittest.main()
