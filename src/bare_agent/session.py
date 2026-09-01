from __future__ import annotations

from dataclasses import dataclass, field

from bare_agent.types import JsonObject, ToolCall, ToolOutcome


@dataclass(frozen=True, slots=True)
class CompletedRound:
    text: str
    calls: tuple[ToolCall, ...]
    outcomes: tuple[ToolOutcome, ...]

    def messages(self) -> list[JsonObject]:
        assistant: JsonObject = {"role": "assistant", "content": self.text or None}
        assistant["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json},
            }
            for call in self.calls
        ]
        messages = [assistant]
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": call.id,
                "name": call.name,
                "content": outcome.content,
            }
            for call, outcome in zip(self.calls, self.outcomes, strict=True)
        )
        return messages


@dataclass(slots=True)
class SessionTurn:
    user_text: str
    rounds: list[CompletedRound] = field(default_factory=list)
    final_text: str | None = None

    def messages(self, *, rounds: list[CompletedRound] | None = None) -> list[JsonObject]:
        selected_rounds = self.rounds if rounds is None else rounds
        messages: list[JsonObject] = [{"role": "user", "content": self.user_text}]
        for completed_round in selected_rounds:
            messages.extend(completed_round.messages())
        if self.final_text is not None:
            messages.append({"role": "assistant", "content": self.final_text})
        return messages


class AgentSession:
    """In-memory conversation history; it does not execute agent steps."""

    def __init__(self, owner: object) -> None:
        self._owner = owner
        self._turns: list[SessionTurn] = []

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def reset(self) -> None:
        self._turns.clear()

    def _belongs_to(self, owner: object) -> bool:
        return self._owner is owner

    def _append(self, turn: SessionTurn) -> None:
        self._turns.append(turn)

    def _completed_turns(self) -> tuple[SessionTurn, ...]:
        return tuple(self._turns)

    def _tool_call_ids(self) -> set[str]:
        return {
            call.id
            for turn in self._turns
            for completed_round in turn.rounds
            for call in completed_round.calls
        }
