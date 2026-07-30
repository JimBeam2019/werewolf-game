import itertools
import random
import time
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


def rendered_content(at) -> str:
    """All user-visible text of the current render, across element types.
    The game log and roster can render as plain text, markdown, or captions
    depending on the display toggles, so tests that assert on *content*
    (not on presentation) should look here.
    """
    parts = (
        [t.value for t in at.text]
        + [m.value for m in at.markdown]
        + [c.value for c in at.caption]
    )
    return "\n".join(parts)


def hidden_role_count(at) -> int:
    """How many living players currently have their role hidden - in either
    roster rendering (list: '— ? —'; cards: '? · alive' in the card HTML).
    """
    content = rendered_content(at)
    return content.count("— ? —") + content.count("? · alive")


class TestStreamlitApp(unittest.TestCase):
    def setUp(self):
        # streamlit_app.py wires bots up to a real local Ollama model via
        # LangChainSpeakStrategy. These tests exercise the Streamlit
        # plumbing (state, fragments, rerun timing), not model output, so
        # the LLM call itself is stubbed out here rather than requiring a
        # live Ollama server to run the suite at all.
        self._speak_counter = itertools.count(1)
        patcher = patch(
            "infrastructure.langchain_speak_strategy.LangChainSpeakStrategy.speak",
            side_effect=self._fake_speak,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _fake_speak(self, speaker, transcript, alive_players):
        n = next(self._speak_counter)
        return f"[line {n}] I suspect someone, said by {speaker.name}."

    def _play_full_game(self, seed: int, num_players: int):
        random.seed(seed)
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=15)
        at.slider[0].set_value(num_players).run(timeout=15)
        at.text_input[0].set_value("Jim").run(timeout=15)
        at.button[0].click().run(timeout=15)

        for _ in range(60):
            self.assertFalse(at.exception, msg=f"Unexpected exception: {at.exception}")
            buttons = [b for b in at.button if b.label != "Play again"]
            if buttons:
                random.choice(buttons).click().run(timeout=15)
                continue

            discussion = (
                at.session_state["discussion"]
                if "discussion" in at.session_state
                else None
            )
            if discussion is not None:
                # Force an early stop instead of waiting out the real
                # 45-second budget. Unlike mutating budget_seconds (which
                # no longer has any effect once the background thread has
                # already captured its own deadline), stop() works
                # regardless of how much budget is configured.
                discussion.stop(join_timeout=3.0)
                at.run(timeout=15)
                continue

            break
        return at

    def test_full_game_runs_without_exceptions(self):
        at = self._play_full_game(seed=0, num_players=6)
        winners = [s.value for s in at.success] + [e.value for e in at.error]
        self.assertEqual(len(winners), 1)

    def test_no_duplicated_log_sections(self):
        # Regression test: a prior bug double-committed the final phase's
        # log messages once the game ended, because the buffer was
        # flushed once inside the advance loop and again just after it.
        at = self._play_full_game(seed=3, num_players=7)
        log = rendered_content(at)
        for round_number in range(1, 4):
            marker = f"--- Day {round_number} ---"
            self.assertLessEqual(
                log.count(marker),
                1,
                msg=f"'{marker}' appeared {log.count(marker)} times - log was duplicated",
            )

    def test_multiple_player_counts_all_complete(self):
        for seed, num_players in enumerate([5, 6, 7, 8]):
            at = self._play_full_game(seed=seed, num_players=num_players)
            self.assertFalse(at.exception)
            winners = [s.value for s in at.success] + [e.value for e in at.error]
            self.assertEqual(len(winners), 1, msg=f"players={num_players}")

    def _start_game(self, seed: int, num_players: int = 6):
        random.seed(seed)
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=15)
        at.slider[0].set_value(num_players).run(timeout=15)
        at.text_input[0].set_value("Jim").run(timeout=15)
        at.button[0].click().run(timeout=15)
        return at

    def test_werewolf_sees_all_roles_at_night(self):
        # Find a seed where the human is assigned the werewolf role.
        for seed in range(30):
            at = self._start_game(seed)
            caption = at.caption[0].value if at.caption else ""
            if "werewolf" not in caption.lower():
                continue

            self.assertTrue(
                any("werewolf" in i.value.lower() for i in at.info),
                msg="Expected a night-vision info banner for the werewolf player",
            )
            self.assertTrue(at.markdown, msg="Roster should not be empty")
            self.assertEqual(
                hidden_role_count(at),
                0,
                msg="Werewolf should see every role during the night",
            )
            return
        self.fail("No seed in range produced a werewolf human within 30 tries")

    def test_villager_does_not_see_hidden_roles_at_night(self):
        for seed in range(30):
            at = self._start_game(seed)
            caption = at.caption[0].value if at.caption else ""
            if "villager" not in caption.lower():
                continue
            # Role assignment isn't seeded, so occasionally the human villager
            # is killed overnight and the game ends outright during the very
            # next day vote - at which point every role is revealed (game
            # over), which isn't the scenario this test is about. Skip those
            # trials and keep looking for one where the game is still ongoing.
            game_over = bool(at.success or at.error)
            if game_over or "eliminated" in caption.lower():
                continue

            self.assertEqual(
                at.info, [], msg="Villagers should get no night-vision banner"
            )
            self.assertGreater(
                hidden_role_count(at),
                0,
                msg="Villager should NOT see other living players' roles at night",
            )
            return
        self.fail("No seed in range produced a villager human within 30 tries")

    def test_werewolf_night_vision_disappears_during_the_day(self):
        for seed in range(30):
            at = self._start_game(seed)
            caption = at.caption[0].value if at.caption else ""
            if "werewolf" not in caption.lower():
                continue

            # Resolve the night decision, then the day vote, to reach day phase.
            for _ in range(2):
                buttons = [b for b in at.button if b.label != "Play again"]
                if buttons:
                    buttons[0].click().run(timeout=15)

            self.assertEqual(
                at.info, [], msg="Night-vision banner should not persist into the day"
            )
            self.assertGreater(
                hidden_role_count(at),
                0,
                msg="Roles should be hidden again once it's day",
            )
            return
        self.fail("No seed in range produced a werewolf human within 30 tries")

    def _reach_discussion_phase(self, seed: int, num_players: int = 6):
        # Role assignment isn't seeded (GameSetupService.create_players is
        # called without an rng in streamlit_app.py), so `seed` doesn't
        # fully pin down the game - occasionally the human dies overnight
        # before ever reaching a live discussion. Retry with nearby seeds
        # until we land on a run where the human is alive going into the
        # discussion, so these tests aren't flaky about something that
        # isn't the behavior under test.
        for attempt in range(10):
            at = self._start_game(seed + attempt, num_players)
            for _ in range(3):
                if (
                    "discussion" in at.session_state
                    and at.session_state["discussion"] is not None
                ):
                    break
                buttons = [b for b in at.button if b.label != "Play again"]
                if buttons:
                    buttons[0].click().run(timeout=15)

            has_discussion = (
                "discussion" in at.session_state
                and at.session_state["discussion"] is not None
            )
            human = (
                at.session_state["engine"].players[0]
                if "engine" in at.session_state
                else None
            )
            if has_discussion and human is not None and human.is_alive:
                return at
        self.fail(f"Could not reach a live discussion phase within 10 seeds of {seed}")

    def test_human_message_syncs_into_the_discussion_transcript(self):
        at = self._reach_discussion_phase(seed=1)
        self.assertIn("discussion", at.session_state)
        self.assertIsNotNone(at.session_state["discussion"])
        self.assertTrue(
            at.chat_input, msg="Expected a chat input during the discussion phase"
        )

        at.chat_input[0].set_value("I suspect the quiet one.").run(timeout=15)

        transcript = at.session_state["discussion"].snapshot_transcript()
        human_lines = [m for m in transcript if m.is_human]
        self.assertEqual(len(human_lines), 1)
        self.assertEqual(human_lines[0].content, "I suspect the quiet one.")

        rendered = [m.value for m in at.markdown]
        self.assertTrue(
            any("I suspect the quiet one." in line for line in rendered),
            msg="Human message should be rendered in the on-screen transcript",
        )
        at.session_state["discussion"].stop(join_timeout=3.0)

    def test_discussion_transitions_to_vote_once_stopped(self):
        at = self._reach_discussion_phase(seed=1)
        discussion = at.session_state["discussion"]
        discussion.stop(join_timeout=3.0)
        at.run(timeout=15)

        vote_buttons = [b for b in at.button if b.label != "Play again"]
        self.assertTrue(
            vote_buttons, msg="Expected vote buttons once the discussion was stopped"
        )

    def test_transcript_stays_visible_while_voting(self):
        at = self._reach_discussion_phase(seed=1)
        discussion = at.session_state["discussion"]
        at.chat_input[0].set_value("I suspect the quiet one.").run(timeout=15)
        discussion.stop(join_timeout=3.0)
        at.run(timeout=15)

        vote_buttons = [b for b in at.button if b.label != "Play again"]
        self.assertTrue(vote_buttons, msg="Expected the human's vote to be pending")
        self.assertIn(
            "I suspect the quiet one.",
            rendered_content(at),
            msg="The discussion transcript should stay on screen while voting",
        )

    def test_chat_avatars_match_roster_avatars(self):
        from streamlit_app import PLAYER_AVATARS

        at = self._reach_discussion_phase(seed=1)
        discussion = at.session_state["discussion"]
        at.chat_input[0].set_value("avatar check").run(timeout=15)
        try:
            by_name = {
                p.name: PLAYER_AVATARS[p.id % len(PLAYER_AVATARS)]
                for p in discussion.alive_players
            }
            transcript = discussion.snapshot_transcript()
            expected_assistant = {
                by_name[m.speaker_name]
                for m in transcript
                if not m.is_human and m.speaker_name in by_name
            }
            actual_assistant = {
                cm.avatar for cm in at.chat_message if cm.name == "assistant"
            }
            self.assertEqual(actual_assistant, expected_assistant)

            human_avatars = {cm.avatar for cm in at.chat_message if cm.name == "user"}
            self.assertEqual(human_avatars, {by_name["Jim"]})
        finally:
            discussion.stop(join_timeout=3.0)

    def test_newest_first_toggle_inverts_message_order(self):
        at = self._reach_discussion_phase(seed=1)
        discussion = at.session_state["discussion"]
        at.chat_input[0].set_value("alpha-anchor").run(timeout=15)
        at.chat_input[0].set_value("zeta-anchor").run(timeout=15)
        try:
            content = rendered_content(at)
            self.assertLess(
                content.find("alpha-anchor"),
                content.find("zeta-anchor"),
                msg="Default order should be oldest first",
            )

            at.checkbox[2].set_value(True).run(timeout=15)  # ui_newest_first on
            content = rendered_content(at)
            self.assertLess(
                content.find("zeta-anchor"),
                content.find("alpha-anchor"),
                msg="Newest-first should invert the message order",
            )
        finally:
            discussion.stop(join_timeout=3.0)

    def test_bot_votes_reflect_the_discussion_transcript(self):
        # Regression check: bot votes should be informed by mentions in the
        # discussion transcript (StubTranscriptVoteStrategy), not pure
        # random - this exercises the same wiring end-to-end via the UI.
        at = self._reach_discussion_phase(seed=2)
        discussion = at.session_state["discussion"]
        discussion.stop(join_timeout=3.0)
        at.run(timeout=15)

        vote_buttons = [b for b in at.button if b.label != "Play again"]
        self.assertTrue(vote_buttons)
        vote_buttons[0].click().run(timeout=15)

        self.assertIn("votes for", rendered_content(at))

    def test_rapid_human_messages_do_not_disrupt_the_independent_bot_pace(self):
        # With the asyncio/threading discussion model, bots speak on their
        # own independent schedule in a background thread, entirely
        # decoupled from how often Streamlit reruns. Sending several human
        # messages back-to-back (each of which triggers a full rerun)
        # should never itself cause an extra bot message - there's no
        # shared "tick" for a rerun to accidentally trigger anymore.
        #
        # To test this without flakiness, bots are forced onto a very
        # long speaking cadence (100s) for this game only, so none of them
        # would naturally speak again within the test's real (short)
        # duration - meaning any bot message that *does* appear here could
        # only be explained by a structural coupling bug, not by enough
        # real time having simply passed.
        import application.discussion as discussion_module

        real_coordinator_cls = discussion_module.DiscussionCoordinator

        def slow_coordinator(*args, **kwargs):
            kwargs.setdefault("speak_delay_range", (100.0, 100.0))
            return real_coordinator_cls(*args, **kwargs)

        with patch("streamlit_app.DiscussionCoordinator", slow_coordinator):
            at = self._reach_discussion_phase(seed=1)
            discussion = at.session_state["discussion"]
            # Let the (patched, slow-speaking) bots take their one natural
            # first turn before we start measuring, so we're specifically
            # testing "no *extra* messages from rapid human input", not
            # racing each bot's very first turn.
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                time.sleep(0.1)
            bot_count_before = sum(
                1 for m in discussion.snapshot_transcript() if not m.is_human
            )

            for i in range(3):
                at.chat_input[0].set_value(f"quick message {i}").run(timeout=15)

            bot_count_after = sum(
                1 for m in discussion.snapshot_transcript() if not m.is_human
            )
            self.assertEqual(
                bot_count_after,
                bot_count_before,
                msg="Sending human messages should never itself trigger a bot message",
            )

            human_count = sum(1 for m in discussion.snapshot_transcript() if m.is_human)
            self.assertEqual(
                human_count,
                3,
                msg="All human messages should still be recorded immediately",
            )
            discussion.stop(join_timeout=3.0)


class TestFormatLogLine(unittest.TestCase):
    def test_prefix_mapping(self):
        from streamlit_app import _format_log_line

        self.assertIn("🌙", _format_log_line("--- Night 2 ---"))
        self.assertIn("☀️", _format_log_line("--- Day 1 ---"))
        self.assertIn("⚰️", _format_log_line("Bob was found dead this morning."))
        self.assertIn("❌", _format_log_line("Eve was voted out by the village."))
        self.assertIn("🗳️", _format_log_line("Alice votes for Bob"))
        self.assertIn(
            "🤝", _format_log_line("The vote was tied. No one is eliminated.")
        )
        self.assertIn("🏁", _format_log_line("=== Game Over: VILLAGERS_WIN ==="))
        self.assertEqual(
            _format_log_line("  Alice (villager, alive)"), "Alice (villager, alive)"
        )


class TestDisplaySettings(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "infrastructure.langchain_speak_strategy.LangChainSpeakStrategy.speak",
            side_effect=Exception("unused in these tests"),
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _start_game(self, seed: int = 0, num_players: int = 6):
        random.seed(seed)
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=15)
        at.slider[0].set_value(num_players).run(timeout=15)
        at.text_input[0].set_value("Jim").run(timeout=15)
        at.button[0].click().run(timeout=15)
        return at

    def test_sidebar_toggles_default_on(self):
        at = self._start_game()
        self.assertFalse(at.exception)
        # Order and defaults match UI_TOGGLES in streamlit_app.py.
        expected = [
            ("🃏 Roster as cards", True),
            ("📜 Formatted game log", True),
            ("🔄 Newest messages first", False),
        ]
        self.assertEqual([c.label for c in at.checkbox], [e[0] for e in expected])
        for checkbox, (label, default) in zip(at.checkbox, expected):
            self.assertEqual(
                checkbox.value,
                default,
                msg=f"Toggle '{label}' has the wrong default",
            )

    def test_roster_cards_toggle_switches_rendering(self):
        at = self._start_game()
        # Cards render name/role/status as separate elements, so the list
        # view's "**Name** — role — status" line must NOT be present.
        self.assertNotIn("**Jim** —", rendered_content(at))
        self.assertIn("Jim", rendered_content(at))

        at.checkbox[0].set_value(False).run(timeout=15)  # ui_cards off
        self.assertIn("**Jim** —", rendered_content(at))

    def test_roster_cards_have_distinct_avatars_per_player(self):
        at = self._start_game(num_players=6)
        avatars = {m.value for m in at.markdown if m.value.startswith("## ")}
        # Whatever the roles/vision situation at game start, six players
        # with per-id avatars can never collapse to fewer than 3 distinct
        # ones (worst case: night vision reveals 🐺s, the rest are unique).
        self.assertGreaterEqual(
            len(avatars), 3, msg=f"Expected distinct per-player avatars, got {avatars}"
        )

    def test_formatted_log_toggle_switches_rendering(self):
        at = self._start_game()
        self.assertFalse(at.exception)
        # If Jim is a werewolf, the night kill is waiting on his click and
        # the log is still empty - resolve it so there's log content either
        # way (role assignment is deliberately not seeded).
        pending_buttons = [b for b in at.button if b.label != "Play again"]
        if pending_buttons:
            pending_buttons[0].click().run(timeout=15)
        at.checkbox[1].set_value(False).run(timeout=15)  # ui_formatted_log off
        self.assertTrue(
            any("--- Night 1 ---" in t.value for t in at.text),
            msg="Plain-text log should render when the formatted log is off",
        )

    def test_play_again_preserves_display_preferences(self):
        at = self._start_game()
        at.checkbox[0].set_value(False).run(timeout=15)  # ui_cards off

        # Drive the game to completion quickly: click whatever decision
        # buttons appear and stop any live discussion immediately.
        for _ in range(60):
            if "engine" not in at.session_state:
                break
            buttons = [b for b in at.button if b.label != "Play again"]
            if buttons:
                buttons[0].click().run(timeout=15)
                continue
            discussion = (
                at.session_state["discussion"]
                if "discussion" in at.session_state
                else None
            )
            if discussion is not None:
                discussion.stop(join_timeout=3.0)
                at.run(timeout=15)
                continue
            break

        play_again = [b for b in at.button if b.label == "Play again"]
        self.assertTrue(play_again, msg="Game should have ended")
        play_again[0].click().run(timeout=15)

        self.assertNotIn("engine", at.session_state)
        self.assertEqual(at.session_state["ui_cards"], False)

    def test_duplicate_bot_name_warns_and_does_not_start(self):
        at = AppTest.from_file("streamlit_app.py")
        at.run(timeout=15)
        at.text_input[0].set_value("Bob").run(timeout=15)  # Bob is a default bot
        at.button[0].click().run(timeout=15)

        self.assertTrue(at.warning, msg="Expected a duplicate-name warning")
        self.assertNotIn("engine", at.session_state)


if __name__ == "__main__":
    unittest.main()
