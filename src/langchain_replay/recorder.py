"""Capture LangGraph agent stream events into recorded turns."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_replay._models import RecordedAskCall, RecordedEvent, RecordedTurn


class ConversationRecorder:
    """Records LLM interactions during a test run.

    Consumes LangGraph stream events (``on_tool_start``, ``on_tool_end``,
    ``on_chat_model_end``, ``on_chain_end``) and direct chat-model calls,
    and persists them to a JSONL fixture on disk.
    """

    def __init__(self) -> None:
        self.turns: list[RecordedTurn] = []
        self.ask_calls: list[RecordedAskCall] = []
        self._current_turn: RecordedTurn | None = None

    # ---- turn lifecycle ----

    def start_turn(self, phase: str, user_message: str) -> None:
        if self._current_turn is not None:
            self.turns.append(self._current_turn)
        self._current_turn = RecordedTurn(phase=phase, user_message=user_message)

    def end_turn(self) -> None:
        if self._current_turn is not None:
            self.turns.append(self._current_turn)
            self._current_turn = None

    @property
    def current_turn(self) -> RecordedTurn | None:
        return self._current_turn

    # ---- event capture ----

    def record_event(self, event: dict) -> None:
        if self._current_turn is None:
            return

        event_type = event.get("event", "")

        if event_type == "on_chain_end" and event.get("name") == "LangGraph":
            response = event.get("data", {}).get("output", {})
            if response:
                self._current_turn.response = self._serialize_response(response)
            return

        recorded = RecordedEvent(event_type=event_type)

        if event_type == "on_tool_start":
            recorded.name = event.get("name", "")
            recorded.data = {"input": self._safe(event.get("data", {}).get("input", {}))}

        elif event_type == "on_tool_end":
            recorded.name = event.get("name", "")
            output = event.get("data", {}).get("output", "")
            if hasattr(output, "content"):
                output = output.content
            recorded.data = {"output": str(output)}

        elif event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            if output is not None and hasattr(output, "content"):
                content = output.content
                if isinstance(content, str):
                    recorded.data = {"content": content}
                elif isinstance(content, list):
                    recorded.data = {"content": [self._serialize_block(b) for b in content]}

        self._current_turn.events.append(recorded)

    def set_response(self, response: Any) -> None:
        """Attach a final response to the in-progress turn."""
        if self._current_turn is not None:
            self._current_turn.response = self._serialize_response(response)

    def record_ask_call(self, prompt: str, response: str) -> None:
        self.ask_calls.append(RecordedAskCall(prompt=prompt, response=response))

    # ---- save ----

    def save(self, recording_dir: Path) -> None:
        if self._current_turn is not None:
            self.turns.append(self._current_turn)
            self._current_turn = None

        recording_dir.mkdir(parents=True, exist_ok=True)

        with (recording_dir / "recording.jsonl").open("w") as f:
            for ask_call in self.ask_calls:
                f.write(json.dumps({"type": "ask_call", **ask_call.to_dict()}) + "\n")
            for turn in self.turns:
                f.write(json.dumps({"type": "turn", **turn.to_dict()}) + "\n")

        with (recording_dir / "metadata.json").open("w") as f:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "num_turns": len(self.turns),
                    "num_ask_calls": len(self.ask_calls),
                },
                f,
                indent=2,
            )

    # ---- helpers ----

    def _safe(self, obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): self._safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._safe(v) for v in obj]
        return str(obj)

    def _serialize_block(self, block: Any) -> dict:
        if isinstance(block, dict):
            return block
        if not hasattr(block, "type"):
            return {"value": str(block)}
        out: dict[str, Any] = {"type": block.type}
        if block.type == "text" and hasattr(block, "text"):
            out["text"] = block.text
        elif block.type == "tool_use":
            if hasattr(block, "name"):
                out["name"] = block.name
            if hasattr(block, "input"):
                out["input"] = self._safe(block.input)
            if hasattr(block, "id"):
                out["id"] = block.id
        return out

    def _serialize_response(self, response: Any) -> dict:
        if not isinstance(response, dict):
            return {}

        result: dict[str, Any] = {}

        if "structured_response" in response:
            sr = response["structured_response"]
            result["structured_response"] = vars(sr) if hasattr(sr, "__dict__") else sr

        if "messages" in response:
            messages = []
            for msg in response["messages"]:
                msg_dict: dict[str, Any] = {"type": getattr(msg, "type", "unknown")}
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str):
                        msg_dict["content"] = content
                    elif isinstance(content, list):
                        msg_dict["content"] = [self._serialize_block(b) for b in content]
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    msg_dict["tool_calls"] = [
                        {"name": tc.get("name", ""), "args": self._safe(tc.get("args", {}))} for tc in msg.tool_calls
                    ]
                messages.append(msg_dict)
            result["messages"] = messages

        return result
