"""Dataclasses for recorded LLM interactions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RecordedEvent:
    """A single event captured from a LangGraph agent's stream."""

    event_type: str
    name: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RecordedEvent:
        return cls(
            event_type=data["event_type"],
            name=data.get("name", ""),
            data=data.get("data", {}),
        )


@dataclass
class RecordedTurn:
    """A recorded conversation turn: user message, events, and final response."""

    phase: str
    user_message: str
    events: list[RecordedEvent] = field(default_factory=list)
    response: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "user_message": self.user_message,
            "events": [e.to_dict() for e in self.events],
            "response": self.response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RecordedTurn:
        return cls(
            phase=data["phase"],
            user_message=data["user_message"],
            events=[RecordedEvent.from_dict(e) for e in data.get("events", [])],
            response=data.get("response", {}),
        )


@dataclass
class RecordedAskCall:
    """A direct LLM call (e.g. ChatModel.ainvoke) recorded outside of an agent turn."""

    prompt: str
    response: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RecordedAskCall:
        return cls(prompt=data["prompt"], response=data["response"])
