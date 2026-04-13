"""Tests for the ConversationRecorder."""

import json
from pathlib import Path

import pytest

from langchain_replay._models import RecordedAskCall, RecordedTurn
from langchain_replay.recorder import ConversationRecorder


@pytest.fixture
def recorder():
    return ConversationRecorder()


# ---------- start_turn / record_event / end_turn ----------

def test_records_tool_start_event(recorder):
    recorder.start_turn("agent", "hello")
    recorder.record_event({"event": "on_tool_start", "name": "my_tool", "data": {"input": {"x": 1}}})
    recorder.end_turn()

    assert len(recorder.turns) == 1
    turn = recorder.turns[0]
    assert turn.phase == "agent"
    assert turn.user_message == "hello"
    assert len(turn.events) == 1
    assert turn.events[0].event_type == "on_tool_start"
    assert turn.events[0].name == "my_tool"
    assert turn.events[0].data == {"input": {"x": 1}}


def test_records_tool_end_event_extracts_content_attribute(recorder):
    class FakeOutput:
        content = "tool result"

    recorder.start_turn("agent", "hello")
    recorder.record_event({"event": "on_tool_end", "name": "my_tool", "data": {"output": FakeOutput()}})
    recorder.end_turn()

    assert recorder.turns[0].events[0].data == {"output": "tool result"}


def test_records_chat_model_end_with_string_content(recorder):
    class Out:
        content = "answer"

    recorder.start_turn("agent", "q")
    recorder.record_event({"event": "on_chat_model_end", "data": {"output": Out()}})
    recorder.end_turn()

    assert recorder.turns[0].events[0].data == {"content": "answer"}


def test_records_chat_model_end_with_content_blocks(recorder):
    class TextBlock:
        type = "text"
        text = "hello"

    class ToolUseBlock:
        type = "tool_use"
        name = "my_tool"
        input = {"x": 1}
        id = "tu_1"

    class Out:
        content = [TextBlock(), ToolUseBlock()]

    recorder.start_turn("agent", "q")
    recorder.record_event({"event": "on_chat_model_end", "data": {"output": Out()}})
    recorder.end_turn()

    blocks = recorder.turns[0].events[0].data["content"]
    assert blocks[0] == {"type": "text", "text": "hello"}
    assert blocks[1] == {"type": "tool_use", "name": "my_tool", "input": {"x": 1}, "id": "tu_1"}


def test_chain_end_for_langgraph_sets_response_and_does_not_emit_event(recorder):
    class Msg:
        type = "ai"
        content = "ok"
        tool_calls: list = []

    recorder.start_turn("agent", "q")
    recorder.record_event({
        "event": "on_chain_end",
        "name": "LangGraph",
        "data": {"output": {"messages": [Msg()]}},
    })
    recorder.end_turn()

    turn = recorder.turns[0]
    assert turn.events == []
    assert turn.response["messages"][0]["content"] == "ok"


def test_record_event_outside_a_turn_is_a_noop(recorder):
    recorder.record_event({"event": "on_tool_start", "name": "x", "data": {}})
    assert recorder.turns == []


def test_starting_a_new_turn_finalizes_the_previous_one(recorder):
    recorder.start_turn("a", "1")
    recorder.start_turn("b", "2")
    recorder.end_turn()
    assert [t.phase for t in recorder.turns] == ["a", "b"]


# ---------- record_ask_call ----------

def test_record_ask_call(recorder):
    recorder.record_ask_call("why?", "because")
    assert recorder.ask_calls == [RecordedAskCall(prompt="why?", response="because")]


# ---------- save ----------

def test_save_writes_jsonl_and_metadata(tmp_path: Path, recorder):
    recorder.start_turn("agent", "hello")
    recorder.record_event({"event": "on_tool_start", "name": "t", "data": {"input": {"x": 1}}})
    recorder.end_turn()
    recorder.record_ask_call("p", "r")

    target = tmp_path / "fixture"
    recorder.save(target)

    assert (target / "recording.jsonl").exists()
    assert (target / "metadata.json").exists()

    # jsonl contains an ask_call line then a turn line
    lines = (target / "recording.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    second = json.loads(lines[1])
    assert first["type"] == "ask_call"
    assert first["prompt"] == "p"
    assert second["type"] == "turn"
    assert second["phase"] == "agent"

    # metadata
    with (target / "metadata.json").open() as f:
        meta = json.load(f)
    assert meta["num_turns"] == 1
    assert meta["num_ask_calls"] == 1
    assert "created_at" in meta


def test_save_finalizes_in_progress_turn(tmp_path: Path, recorder):
    recorder.start_turn("agent", "hello")
    recorder.record_event({"event": "on_tool_start", "name": "t", "data": {"input": {}}})
    # no end_turn() called
    recorder.save(tmp_path / "fx")
    assert len(recorder.turns) == 1


def test_save_creates_parent_directories(tmp_path: Path, recorder):
    target = tmp_path / "deeply" / "nested" / "fixture"
    recorder.save(target)
    assert target.is_dir()
