from typing import List, Optional

from domain.entities import ChatMessage, Player


def _prompt_choice(prompt: str, candidates: List[Player]) -> Player:
    while True:
        print(prompt)
        for idx, candidate in enumerate(candidates, start=1):
            print(f"  {idx}. {candidate.name}")
        raw = input("Enter a number: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(candidates):
            return candidates[int(raw) - 1]
        print("Invalid choice, please try again.\n")


class ConsoleWerewolfStrategy:
    """Prompts a human (e.g. one werewolf player passing the device around,
    or a single werewolf player) to choose the night's victim.
    """

    def choose_victim(
        self, werewolves: List[Player], candidates: List[Player]
    ) -> Player:
        names = ", ".join(w.name for w in werewolves)
        return _prompt_choice(f"\n[Werewolves: {names}] Choose a victim:", candidates)


class ConsoleVoteStrategy:
    """Prompts a human to cast their day-phase vote. If a discussion
    transcript is provided, it's printed first as a reminder.
    """

    def cast_vote(
        self,
        voter: Player,
        candidates: List[Player],
        transcript: Optional[List[ChatMessage]] = None,
    ) -> Player:
        if transcript:
            print("\n--- Discussion recap ---")
            for msg in transcript:
                print(f"{msg.speaker_name}: {msg.content}")
        return _prompt_choice(
            f"\n{voter.name}, who do you vote to eliminate?", candidates
        )
