import os
import uuid
import asyncio
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


from application.session_states import initialize_params
from application.discussion import DiscussionCoordinator
from application.game_engine import GameEngine
from application.setup import GameSetupService
from domain.entities import ChatMessage
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
from infrastructure.langgraph_memory import LangGraphMemoryStore
from infrastructure.load_file import initialize_knowledge_base
from infrastructure.background_knowledge_provider import RAGBackgroundKnowledgeProvider

# from infrastructure.stub_speak_strategy import StubSpeakStrategy
from infrastructure.transcript_vote_strategy import StubTranscriptVoteStrategy

load_dotenv()

st.set_page_config(page_title="Werewolf", page_icon="🐺", layout="wide")

DEFAULT_BOT_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi"]
DISCUSSION_BUDGET_SECONDS = 45.0
AGENT_LLM_MODEL = os.getenv("AGENT_LLM_MODEL")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# One distinct human avatar per player id (max MAX_PLAYERS players). Dead
# players and revealed werewolves override this with ⚰️/🐺 in the roster.
PLAYER_AVATARS = ["👨", "👩", "🧔", "👴", "👵", "🧓", "👱", "🧕"]

# Display-preference toggles shown in the sidebar, as key -> (label,
# default). Keys start with `ui_` so reset_game() can preserve them across
# "Play again".
UI_TOGGLES = {
    "ui_cards": ("🃏 Roster as cards", True),
    "ui_formatted_log": ("📜 Formatted game log", True),
    "ui_newest_first": ("🔄 Newest messages first", False),
}

initialize_params()
vector_store = initialize_knowledge_base(DEFAULT_BOT_NAMES)


def ui_pref(key: str) -> bool:
    return bool(st.session_state.get(key, UI_TOGGLES[key][1]))


def render_display_settings() -> None:
    st.sidebar.subheader("⚙️ Display")
    for key, (label, default) in UI_TOGGLES.items():
        st.sidebar.checkbox(label, value=default, key=key)


def start_new_game(num_players: int, human_name: str) -> None:
    names = DEFAULT_BOT_NAMES[:num_players]
    names[0] = human_name.strip() or "John"

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

    background_provider = RAGBackgroundKnowledgeProvider(vector_store)  # type: ignore

    engine = GameEngine(
        players=players,
        human_id=human_id,
        background_provider=background_provider,
        werewolf_strategy=werewolf_strategy,
        vote_strategy=vote_strategy,
        notifier=notifier,
    )

    with st.spinner("Loading game content. Please wait around a minute..."):
        try:
            asyncio.run(engine.set_player_backgrounds())
        except Exception:
            pass

    holder["engine"] = engine

    st.session_state.engine = engine
    st.session_state.human_id = human_id
    st.session_state.log = []
    st.session_state.pending = None
    st.session_state.discussion = None
    st.session_state.game_id = str(uuid.uuid4())
    st.session_state.memory_store = LangGraphMemoryStore()


def _persist_system_messages(lines) -> None:
    """Folds game-event log lines (night kills, day eliminations) into
    the same cross-round memory as the discussion transcript, so tomorrow's
    agents know who died or was voted out yesterday - not just what was
    said in chat.
    """
    messages = [
        ChatMessage(speaker_name="Game", content=line.strip(), is_human=False)
        for line in lines
        if line.strip()
    ]
    if messages:
        st.session_state.memory_store.append_and_save(
            st.session_state.game_id, messages
        )


def get_or_start_discussion(engine: GameEngine) -> DiscussionCoordinator:
    """Creates the discussion coordinator for the current day round the
    first time it's needed, then returns the same instance on every
    subsequent call this round - so repeated Streamlit reruns don't reset
    the 45-second clock or lose the transcript so far.

    `prior_history` is loaded from the cross-round memory store, giving
    this round's agents access to everything that happened on earlier
    days (previous chat, who was killed, who was voted out) - without it,
    every new day would start as if the game had no past at all.
    """
    if st.session_state.get("discussion") is None:
        prior_history = st.session_state.memory_store.load_history(
            st.session_state.game_id
        )
        llm = ChatOpenAI(
            model=AGENT_LLM_MODEL if AGENT_LLM_MODEL else "llama3.1:8b-instruct-q4_K_M",
            api_key="EMPTY",  # type: ignore
            base_url=LLM_BASE_URL,
            temperature=0.8,
            max_retries=1,
        )
        discussion = DiscussionCoordinator(
            alive_players=engine.alive_players(),
            human_id=engine.human_id,
            # speak_strategy=StubSpeakStrategy(),
            speak_strategy=LangChainSpeakStrategy(llm=llm),
            budget_seconds=DISCUSSION_BUDGET_SECONDS,
            prior_history=prior_history,
        )
        discussion.start()
        st.session_state.discussion = discussion
    return st.session_state.discussion


def reset_game() -> None:
    # Keep ui_* display preferences across games; everything else goes.
    for key in list(st.session_state.keys()):
        if not key.startswith("ui_"):
            del st.session_state[key]


def render_setup() -> None:
    _, col_center, _ = st.columns([0.2, 0.6, 0.2])

    with col_center:
        st.subheader("🐺 Werewolf")
        st.write(
            "A social deduction prototype. Play one seat yourself; the rest are bots."
        )

        with st.expander("📖 How to play"):
            st.write(
                "- **Night:** the werewolves secretly choose one villager to "
                "eliminate. If you're a werewolf, you make that choice.\n"
                "- **Day:** everyone discusses who seems suspicious, then votes. "
                "The player with the most votes is eliminated; ties eliminate no one.\n"
                "- **Villagers win** when all werewolves are dead. **Werewolves "
                "win** when they match or outnumber the villagers.\n"
                f"- Games of {MIN_PLAYERS}-6 players have 1 werewolf; "
                f"7-{MAX_PLAYERS} players have 2."
            )

        # Outside the form so the bot preview below updates live as it changes -
        # widgets inside a st.form don't trigger a rerun until submission.
        num_players = st.slider("Number of players", MIN_PLAYERS, MAX_PLAYERS, 6)
        bot_names = DEFAULT_BOT_NAMES[1:num_players]
        st.caption(f"Bots in this game: {', '.join(bot_names)}")

        with st.form("setup_form"):
            human_name = st.text_input("Your name", value="John")
            submitted = st.form_submit_button("Start Game", use_container_width=True)

        if submitted:
            name = human_name.strip() or "John"
            if name.lower() in (bot.lower() for bot in bot_names):
                # Names double as identity in the vote record and discussion
                # transcripts, so sharing one with a bot would corrupt both.
                st.warning(f"'{name}' is already taken by a bot - pick another name.")
                return
            start_new_game(num_players, name)
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
            _persist_system_messages(engine.notifier.buffer)  # type: ignore
            st.session_state.pending = None
            continue

        if engine.phase == GamePhase.DAY:
            discussion = get_or_start_discussion(engine)
            if not discussion.is_finished:
                return  # hand off to the discussion UI - don't vote yet

            engine.notifier.buffer.clear()  # type: ignore
            try:
                engine.run_day_phase(transcript=discussion.full_context())
            except PendingHumanDecision as pending:
                st.session_state.pending = pending
                return
            st.session_state.log.extend(engine.notifier.buffer)  # type: ignore
            # Persist today's fresh chat (not full_context(), which would
            # duplicate prior_history that's already saved) plus the vote
            # outcome, so tomorrow's agents remember both the conversation
            # and what actually happened as a result of it.
            st.session_state.memory_store.append_and_save(
                st.session_state.game_id, discussion.snapshot_transcript()
            )
            _persist_system_messages(engine.notifier.buffer)  # type: ignore
            st.session_state.pending = None
            st.session_state.discussion = None  # fresh coordinator next round
            continue

        break


def _render_transcript_messages(discussion: DiscussionCoordinator) -> None:
    """Renders the discussion transcript as chat messages, in chronological
    order or newest-first depending on the ui_newest_first toggle.

    Each speaker gets the same per-player human avatar shown in the roster
    cards - but never the roster's 🐺/⚰️ overrides, which would leak roles
    and deaths mid-discussion. Coordinator-injected "System" notices get a
    neutral ⚙️.
    """
    avatars = {
        p.name: PLAYER_AVATARS[p.id % len(PLAYER_AVATARS)]
        for p in discussion.alive_players
    }
    messages = discussion.snapshot_transcript()
    if ui_pref("ui_newest_first"):
        messages = list(reversed(messages))

    # Chat area
    chat_area = st.container(height=400)

    with chat_area:
        for msg in messages:
            with st.chat_message(
                "user" if msg.is_human else "assistant",
                avatar=avatars.get(msg.speaker_name, "⚙️"),
            ):
                st.write(f"**{msg.speaker_name}:** {msg.content}")


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

    # While the human is deciding their day vote, keep the discussion they
    # just had on screen below the vote buttons - read-only: the live
    # fragment (countdown, chat input, auto-rerun) only makes sense while
    # the discussion is actually running.
    discussion = st.session_state.get("discussion")
    if pending.key.startswith("day_vote") and discussion is not None:
        st.subheader("Village discussion")
        _render_transcript_messages(discussion)


@st.fragment(run_every="1s")
def render_discussion(discussion: DiscussionCoordinator) -> None:
    """The live discussion - countdown and message list - refreshes on a
    1-second cadence. It doesn't drive the discussion forward: bots speak
    on their own schedule in a background thread (see
    application/discussion.py), so this only reads that shared state
    (thread-safely, via `snapshot_transcript()`) and renders it.

    The human's chat input deliberately does NOT live here: Streamlit only
    pins an st.chat_input to the bottom of the viewport (its sticky
    "bottom" block) when the call happens at the top level of the main
    container with no ancestor blocks - inside this fragment it renders
    inline and scrolls away with the content. It's called in render_game
    instead.
    """
    col_header, col_timer = st.columns([0.3, 0.7])
    with col_header:
        st.subheader("Village Discussion")

    with col_timer:
        st.caption(f"{int(discussion.time_remaining)}s remaining before the vote.")
        st.progress(
            min(1.0, max(0.0, discussion.time_remaining / discussion.budget_seconds))
        )

    _render_transcript_messages(discussion)

    if discussion.is_finished:
        st.rerun()


def _format_log_line(line: str) -> str:
    """Maps one raw game-log line to an icon-prefixed line for the formatted
    log view. Pure presentation - the log's content is unchanged.
    """
    stripped = line.strip()
    if not stripped:
        return ""
    if stripped.startswith("--- Night"):
        return f"🌙 **{stripped}**"
    if stripped.startswith("--- Day"):
        return f"☀️ **{stripped}**"
    if "was found dead" in stripped:
        return f"⚰️ {stripped}"
    if "was voted out" in stripped:
        return f"❌ {stripped}"
    if "votes for" in stripped:
        return f"🗳️ {stripped}"
    if "tied" in stripped:
        return f"🤝 {stripped}"
    if "Game Over" in stripped:
        return f"🏁 **{stripped}**"
    return stripped


def _render_game_log() -> None:
    """The game log lives in the sidebar, below the display settings."""
    st.sidebar.divider()
    st.sidebar.subheader("Game Log")
    log_lines = st.session_state.log
    if not ui_pref("ui_formatted_log"):
        st.sidebar.text(
            "\n".join(log_lines) if log_lines else "The game is about to begin..."
        )
        return
    if not log_lines:
        st.sidebar.caption("The game is about to begin...")
    for line in log_lines:
        formatted = _format_log_line(line)
        if formatted:
            st.sidebar.markdown(formatted)


def _render_roster_list(engine: GameEngine, human_id: int, night_vision: bool) -> None:
    """The classic roster rendering (pre-cards)."""
    for p in engine.players:
        reveal = (
            (not p.is_alive)
            or (p.id == human_id)
            or engine.result != GameResult.ONGOING
            or night_vision
        )
        role_label = p.role.value if reveal else "?"
        status = "🟢 alive" if p.is_alive else "⚰️ dead"
        st.write(f"**{p.name}** — {role_label} — {status}")


def _render_roster_cards(engine: GameEngine, human_id: int, night_vision: bool) -> None:
    """Roster as a card grid. Same reveal rules as the list view - only the
    presentation changes.
    """
    for i, p in enumerate(engine.players):
        reveal = (
            (not p.is_alive)
            or (p.id == human_id)
            or engine.result != GameResult.ONGOING
            or night_vision
        )
        if not p.is_alive:
            avatar = "⚰️"
        elif reveal and p.role == Role.WEREWOLF:
            avatar = "🐺"
        else:
            # Hidden players and revealed villagers keep their own distinct
            # human avatar - the role text below says the rest.
            avatar = PLAYER_AVATARS[p.id % len(PLAYER_AVATARS)]
        role_label = p.role.value if reveal else "?"
        status = "alive" if p.is_alive else "dead"
        card = st.container(height=100, border=True)
        with card:
            icon_col, name_col = st.columns([0.25, 0.75])

            with icon_col:
                st.markdown(f"## {avatar}", anchors=False)

            with name_col:
                st.write(f"**{p.name}**")
                st.caption(f"{role_label} · {status}")


def render_game() -> None:
    engine: GameEngine = st.session_state.engine
    human_id = st.session_state.human_id
    human = next(p for p in engine.players if p.id == human_id)

    werewolf_night_vision = (
        human.is_alive
        and human.role == Role.WEREWOLF
        and engine.phase == GamePhase.NIGHT
    )

    col_main, col_right = st.columns([0.8, 0.2])

    with col_main:
        title_left, title_right = st.columns([0.3, 0.7])
        with title_left:
            st.title("🐺 Werewolf")
            role_note = "" if human.is_alive else " — you have been eliminated"
            st.caption(
                f"You are **{human.name}**, a **{human.role.value}**{role_note}."
            )

        with title_right:
            alive = sum(1 for p in engine.players if p.is_alive)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(
                "Round",
                engine.round_number if engine.result == GameResult.ONGOING else "—",
            )
            col2.metric("Phase", engine.phase.value.capitalize())
            col3.metric("Alive", f"{alive}/{len(engine.players)}")
            col4.metric("Eliminated", len(engine.players) - alive)

        advance_engine()

        if werewolf_night_vision:
            st.info(
                "🌙 It's night — as a werewolf, you can see everyone's true role below."
            )

        # The game log renders into the sidebar, next to the display settings.
        _render_game_log()

        if st.session_state.pending is not None:
            render_pending_decision()
        elif (
            engine.phase == GamePhase.DAY
            and st.session_state.get("discussion") is not None
        ):
            discussion = st.session_state.discussion
            render_discussion(discussion)

        if engine.result != GameResult.ONGOING:
            if engine.result == GameResult.VILLAGERS_WIN:
                st.success("Villagers win! 🎉")
            else:
                st.error("Werewolves win! 🐺")
            if st.button("Play again", use_container_width=True):
                reset_game()
                st.rerun()

    # Called at the top level of the main container on purpose: that's
    # the only spot where Streamlit pins the input to its sticky
    # bottom-of-viewport block, so it never scrolls out. It's also
    # captured *before* rendering the transcript below, so a submitted
    # message is appended before this same run renders the discussion -
    # it shows up immediately instead of one rerun later.
    if human.is_alive:
        message = st.chat_input("Say something to the village...", max_chars=1000)
        if message:
            discussion.add_human_message(human.name, message)
            st.rerun()

    with col_right:
        # The roster is always visible, right under the metrics - no expander.
        # st.subheader("Roster")
        if ui_pref("ui_cards"):
            _render_roster_cards(engine, human_id, werewolf_night_vision)
        else:
            _render_roster_list(engine, human_id, werewolf_night_vision)


def main() -> None:
    # Settings render on every screen: Streamlit drops the session-state
    # entry of a widget that isn't rendered in a run, so keeping the
    # checkboxes mounted is what makes the ui_* preferences survive
    # transitions like "Play again".
    render_display_settings()
    if "engine" not in st.session_state:
        render_setup()
    else:
        render_game()


if __name__ == "__main__":
    main()
