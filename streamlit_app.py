import os
import streamlit as st

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

from application.discussion import DiscussionCoordinator
from application.game_engine import GameEngine
from application.setup import GameSetupService
from domain.enums import GamePhase, GameResult, Role
from domain.exceptions import GameError
from domain.rules import MAX_PLAYERS, MIN_PLAYERS
from infrastructure.streamlit_adapter import (
    BufferingNotifier,
    PendingHumanDecision,
    StreamlitVoteStrategy,
    StreamlitWerewolfStrategy,
)
from infrastructure.langchain_speak_strategy import LangChainSpeakStrategy

# from infrastructure.stub_speak_strategy import StubSpeakStrategy
from infrastructure.transcript_vote_strategy import StubTranscriptVoteStrategy

load_dotenv()

st.set_page_config(page_title="Werewolf", page_icon="🐺", layout="centered")

DEFAULT_BOT_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
DISCUSSION_BUDGET_SECONDS = 45.0
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL")


def start_new_game(num_players: int, human_name: str) -> None:
    names = DEFAULT_BOT_NAMES[:num_players]
    names[0] = human_name.strip() or "You"

    try:
        players = GameSetupService.create_players(names)
    except GameError as exc:
        st.error(str(exc))
        return

    human_id = players[0].id
    # `holder` lets the strategies read the engine's *current* round number
    # even though the engine object doesn't exist yet when they're built.
    holder: dict = {}
    werewolf_strategy = StreamlitWerewolfStrategy(
        human_id, lambda: holder["engine"].round_number
    )
    vote_strategy = StreamlitVoteStrategy(
        human_id,
        lambda: holder["engine"].round_number,
        bot_strategy=StubTranscriptVoteStrategy(),
    )
    notifier = BufferingNotifier()

    engine = GameEngine(
        players=players,
        human_id=human_id,
        werewolf_strategy=werewolf_strategy,
        vote_strategy=vote_strategy,
        notifier=notifier,
    )
    holder["engine"] = engine

    st.session_state.engine = engine
    st.session_state.human_id = human_id
    st.session_state.log = []
    st.session_state.pending = None
    st.session_state.discussion = None


def get_or_start_discussion(engine: GameEngine) -> DiscussionCoordinator:
    """Creates the discussion coordinator for the current day round the
    first time it's needed, then returns the same instance on every
    subsequent call this round - so repeated Streamlit reruns don't reset
    the 45-second clock or lose the transcript so far.
    """
    if st.session_state.get("discussion") is None:
        llm = ChatOllama(
            model=AGENT_LLM_MODEL if AGENT_LLM_MODEL else "llama3.1:8b-instruct-q4_K_M",
            temperature=0.6,
        )
        discussion = DiscussionCoordinator(
            alive_players=engine.alive_players(),
            human_id=engine.human_id,
            # speak_strategy=StubSpeakStrategy(),
            speak_strategy=LangChainSpeakStrategy(llm=llm),
            budget_seconds=DISCUSSION_BUDGET_SECONDS,
        )
        discussion.start()
        st.session_state.discussion = discussion
    return st.session_state.discussion


def reset_game() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]


def render_setup() -> None:
    st.title("🐺 Werewolf")
    st.write("A social deduction prototype. Play one seat yourself; the rest are bots.")

    with st.form("setup_form"):
        num_players = st.slider("Number of players", MIN_PLAYERS, MAX_PLAYERS, 6)
        human_name = st.text_input("Your name", value="You")
        submitted = st.form_submit_button("Start Game", use_container_width=True)

    if submitted:
        start_new_game(num_players, human_name)
        st.rerun()


def advance_engine() -> None:
    """Runs the engine forward through as many bot-only phases as possible.
    Stops when: the game ends, a human decision is needed, or the day
    phase's discussion is in progress (in which case control hands off to
    render_discussion, which the human and the ticking fragment drive).
    """
    engine: GameEngine = st.session_state.engine

    while engine.result == GameResult.ONGOING:
        if engine.phase == GamePhase.NIGHT:
            engine.notifier.buffer.clear()  # type: ignore
            try:
                engine.run_night_phase()
            except PendingHumanDecision as pending:
                st.session_state.pending = pending
                return
            st.session_state.log.extend(engine.notifier.buffer)  # type: ignore
            st.session_state.pending = None
            continue

        if engine.phase == GamePhase.DAY:
            discussion = get_or_start_discussion(engine)
            if not discussion.is_finished:
                return  # hand off to the discussion UI - don't vote yet

            engine.notifier.buffer.clear()  # type: ignore
            try:
                engine.run_day_phase(transcript=discussion.snapshot_transcript())
            except PendingHumanDecision as pending:
                st.session_state.pending = pending
                return
            st.session_state.log.extend(engine.notifier.buffer)  # type: ignore
            st.session_state.pending = None
            st.session_state.discussion = None  # fresh coordinator next round
            continue

        break


def render_pending_decision() -> None:
    pending: PendingHumanDecision = st.session_state.pending
    st.subheader(pending.prompt)
    cols = st.columns(len(pending.candidates)) if len(pending.candidates) <= 4 else None

    for i, candidate in enumerate(pending.candidates):
        target = cols[i] if cols else st
        if target.button(
            candidate.name,
            key=f"{pending.key}_{candidate.id}",
            use_container_width=True,
        ):
            st.session_state[pending.key] = candidate.id
            st.session_state.pending = None
            st.rerun()


@st.fragment(run_every="1s")
def render_discussion(discussion: DiscussionCoordinator, human) -> None:
    """Everything about the live discussion - the countdown, the message
    list, and the human's input - lives in this ONE fragment, and all of
    it refreshes together on the same 1-second cadence.

    Unlike the earlier tick-based version, this fragment doesn't drive
    the discussion forward at all - bots are already speaking on their
    own independent schedule in a background thread (see
    application/discussion.py). This function's only job is to
    periodically read that shared state (thread-safely, via
    `snapshot_transcript()`) and render it, plus forward the human's
    chat input into the same shared transcript.
    """
    st.subheader("Village discussion")
    st.caption(f"{int(discussion.time_remaining)}s remaining before the vote.")
    st.progress(
        min(1.0, max(0.0, discussion.time_remaining / discussion.budget_seconds))
    )

    # Capture the human's input *before* rendering the message list below.
    # st.chat_input always stays visually pinned to the bottom of the page
    # regardless of where it's called in the code, so calling it here
    # doesn't affect layout - but it does mean that if this particular
    # fragment run was triggered by the human submitting a message, that
    # message is already appended by the time we render the transcript,
    # so it shows up immediately instead of waiting for the next tick.
    if human.is_alive:
        message = st.chat_input("Say something to the village...")
        if message:
            discussion.add_human_message(human.name, message)

    for msg in discussion.snapshot_transcript():
        with st.chat_message("user" if msg.is_human else "assistant"):
            st.write(f"**{msg.speaker_name}:** {msg.content}")

    if discussion.is_finished:
        st.rerun()


def render_game() -> None:
    engine: GameEngine = st.session_state.engine
    human_id = st.session_state.human_id
    human = next(p for p in engine.players if p.id == human_id)

    st.title("🐺 Werewolf")
    role_note = "" if human.is_alive else " — you have been eliminated"
    st.caption(f"You are **{human.name}**, a **{human.role.value}**{role_note}.")

    col1, col2 = st.columns(2)
    col1.metric(
        "Round", engine.round_number if engine.result == GameResult.ONGOING else "—"
    )
    col2.metric("Phase", engine.phase.value.capitalize())

    advance_engine()

    if st.session_state.pending is not None:
        render_pending_decision()
    elif (
        engine.phase == GamePhase.DAY and st.session_state.get("discussion") is not None
    ):
        render_discussion(st.session_state.discussion, human)

    werewolf_night_vision = (
        human.is_alive
        and human.role == Role.WEREWOLF
        and engine.phase == GamePhase.NIGHT
    )
    if werewolf_night_vision:
        st.info(
            "🌙 It's night — as a werewolf, you can see everyone's true role below."
        )

    with st.expander("Roster", expanded=True):
        for p in engine.players:
            reveal = (
                (not p.is_alive)
                or (p.id == human_id)
                or engine.result != GameResult.ONGOING
                or werewolf_night_vision
            )
            role_label = p.role.value if reveal else "?"
            status = "🟢 alive" if p.is_alive else "⚰️ dead"
            st.write(f"**{p.name}** — {role_label} — {status}")

    st.subheader("Game Log")
    log_text = (
        "\n".join(st.session_state.log)
        if st.session_state.log
        else "The game is about to begin..."
    )
    st.text(log_text)

    if engine.result != GameResult.ONGOING:
        if engine.result == GameResult.VILLAGERS_WIN:
            st.success("Villagers win! 🎉")
        else:
            st.error("Werewolves win! 🐺")
        if st.button("Play again", use_container_width=True):
            reset_game()
            st.rerun()


def main() -> None:
    if "engine" not in st.session_state:
        render_setup()
    else:
        render_game()


if __name__ == "__main__":
    main()
