"""Replay recorded agent decisions while executing tools live."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncIterator, Sequence
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage  # noqa: F401  (kept for forward compat)

from langchain_replay._models import RecordedAskCall, RecordedEvent, RecordedTurn
from langchain_replay.exceptions import (
    FixtureNotFoundError,
    ReplayExhaustedError,
    ToolNotFoundError,
)


def load_recording(recording_dir: Path) -> tuple[list[RecordedAskCall], list[RecordedTurn]]:
    """Load a recording.jsonl file and return its ask calls and turns."""
    path = recording_dir / "recording.jsonl"
    if not path.exists():
        raise FixtureNotFoundError(f"Recording not found: {path}")

    ask_calls: list[RecordedAskCall] = []
    turns: list[RecordedTurn] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            kind = data.pop("type", None)
            if kind == "ask_call":
                ask_calls.append(RecordedAskCall.from_dict(data))
            elif kind == "turn":
                turns.append(RecordedTurn.from_dict(data))

    return ask_calls, turns


class ReplayAgent:
    """Fake agent that yields recorded decisions and dispatches tool calls live."""

    def __init__(
        self,
        turns: list[RecordedTurn],
        tools: Sequence[Any],
    ) -> None:
        self._turns = list(turns)
        self._turn_index = 0
        self._tools = {tool.name: tool for tool in tools}

    def _next_turn(self) -> RecordedTurn:
        if self._turn_index >= len(self._turns):
            raise ReplayExhaustedError(
                f"No more recorded turns (requested turn {self._turn_index + 1}, "
                f"have {len(self._turns)})"
            )
        turn = self._turns[self._turn_index]
        self._turn_index += 1
        return turn

    async def _execute_tool(self, tool_name: str, tool_input: Any) -> str:
        if tool_name not in self._tools:
            raise ToolNotFoundError(
                f"Recorded tool {tool_name!r} not found in tools provided to ReplayAgent. "
                f"Available: {sorted(self._tools)}"
            )

        tool = self._tools[tool_name]
        if hasattr(tool, "ainvoke"):
            result = await tool.ainvoke(tool_input)
        elif hasattr(tool, "invoke"):
            result = tool.invoke(tool_input)
        elif hasattr(tool, "_run"):
            result = tool._run(**tool_input)
        else:
            raise ToolNotFoundError(f"Tool {tool_name!r} has no invoke method")
        return str(result)

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        turn = self._next_turn()
        for event in turn.events:
            if event.event_type == "on_tool_start":
                await self._execute_tool(event.name, event.data.get("input", {}))
        return self._make_response(turn)

    async def astream_events(self, messages: Any, **kwargs: Any) -> AsyncIterator[dict]:
        turn = self._next_turn()
        for event in turn.events:
            yield {"event": event.event_type, "name": event.name, "data": event.data}
            if event.event_type == "on_tool_start":
                await self._execute_tool(event.name, event.data.get("input", {}))
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": self._make_response(turn)},
        }

    def _make_response(self, turn: RecordedTurn) -> dict:
        response = dict(turn.response)

        if "structured_response" in response:
            sr_data = response["structured_response"]
            sr_mock = MagicMock()
            for key, value in (sr_data or {}).items():
                setattr(sr_mock, key, value)
            response["structured_response"] = sr_mock

        if "messages" in response:
            mock_messages = []
            for msg_data in response["messages"]:
                msg_mock = MagicMock()
                msg_mock.type = msg_data.get("type", "unknown")
                msg_mock.content = msg_data.get("content", "")
                msg_mock.tool_calls = msg_data.get("tool_calls", [])
                if msg_mock.type == "ai" and isinstance(msg_mock.content, str):
                    msg_mock.text = msg_mock.content
                mock_messages.append(msg_mock)
            response["messages"] = mock_messages

        return response


class ReplayAskHandler:
    """Yield recorded direct chat-model responses in order."""

    def __init__(self, ask_calls: Sequence[RecordedAskCall]) -> None:
        self._ask_calls = list(ask_calls)
        self._index = 0

    def get_next_response(self, prompt: str) -> str:
        if self._index >= len(self._ask_calls):
            raise ReplayExhaustedError(
                f"No more recorded ask calls (requested call {self._index + 1}, "
                f"have {len(self._ask_calls)})"
            )
        call = self._ask_calls[self._index]
        self._index += 1
        return call.response
