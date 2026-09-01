from __future__ import annotations

import json
from collections.abc import Sequence

from bare_agent.session import SessionTurn
from bare_agent.types import JsonObject


class ContextExhaustedError(RuntimeError):
    pass


def build_context(
    system_prompt: str,
    completed_turns: Sequence[SessionTurn],
    active_turn: SessionTurn,
    char_budget: int,
) -> tuple[JsonObject, ...]:
    """Build a valid complete-round window without mutating session history."""

    system = {"role": "system", "content": system_prompt}
    kept_turns = list(completed_turns)
    active_rounds = list(active_turn.rounds)

    def render() -> list[JsonObject]:
        messages: list[JsonObject] = [system]
        for turn in kept_turns:
            messages.extend(turn.messages())
        messages.extend(active_turn.messages(rounds=active_rounds))
        return messages

    messages = render()
    while _size(messages) > char_budget and kept_turns:
        kept_turns.pop(0)
        messages = render()
    while _size(messages) > char_budget and active_rounds:
        active_rounds.pop(0)
        messages = render()
    if _size(messages) > char_budget:
        raise ContextExhaustedError("system prompt and active user message exceed context budget")
    return tuple(messages)


def _size(messages: Sequence[JsonObject]) -> int:
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
