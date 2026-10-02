from application.game_engine import DayResult, NightResult


def build_known_facts(history: list) -> str:
    """A model will happily invent a 'revealed' role if left to infer
    what's public knowledge from a loose transcript. Spelling out exactly
    what is and isn't known removes the need for it to infer anything.

    `history` is GameEngine.history - a list of NightResult and DayResult
    entries, one per completed night/day.
    """
    lines = []
    for entry in history:
        if isinstance(entry, NightResult):
            lines.append(
                f"- {entry.victim.name} was found dead. "
                "Their role was NOT revealed - nobody knows it."
            )
        elif isinstance(entry, DayResult):
            if entry.tied:
                lines.append(
                    f"- Round {entry.round_number}'s vote was tied. "
                    "No one was eliminated."
                )
            elif entry.eliminated:
                lines.append(
                    f"- {entry.eliminated.name} was voted out and revealed to be a "
                    f"{entry.eliminated.role.value}."
                )
    return "\n".join(lines) if lines else "No eliminations yet."
