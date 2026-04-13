"""Tests for ReplayAgent, ReplayAskHandler, and load_recording."""

import json
from pathlib import Path

import pytest

from langchain_replay._models import RecordedAskCall, RecordedEvent, RecordedTurn
from langchain_replay.exceptions import ReplayExhaustedError, ToolNotFoundError
from langchain_replay.replayer import ReplayAgent, ReplayAskHandler, load_recording


class FakeTool:
    """A fake LangChain tool that records its inputs and writes a marker file."""

    def __init__(self, name: str, output_dir: Path | None = None):
        self.name = name
        self.received_inputs: list = []
        self._output_dir = output_dir

    async def ainvoke(self, tool_input):
        self.received_inputs.append(tool_input)
        if self._output_dir is not None:
            (self._output_dir / f"{self.name}.txt").write_text(json.dumps(tool_input))
        return f"{self.name}-result"


# ---------- ReplayAgent ----------

async def test_replay_agent_executes_recorded_tools():
    turn = RecordedTurn(
        phase="agent",
        user_message="hi",
        events=[
            RecordedEvent(event_type="on_tool_start", name="my_tool", data={"input": {"x": 1}}),
            RecordedEvent(event_type="on_tool_end", name="my_tool", data={"output": "ok"}),
        ],
        response={"messages": [{"type": "ai", "content": "done"}]},
    )
    tool = FakeTool("my_tool")
    agent = ReplayAgent([turn], tools=[tool])

    result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})

    assert tool.received_inputs == [{"x": 1}]
    assert "messages" in result


async def test_replay_agent_raises_when_exhausted():
    agent = ReplayAgent([], tools=[])
    with pytest.raises(ReplayExhaustedError):
        await agent.ainvoke({"messages": [{"role": "user", "content": "x"}]})


async def test_replay_agent_raises_on_unknown_tool():
    turn = RecordedTurn(
        phase="agent", user_message="hi",
        events=[RecordedEvent(event_type="on_tool_start", name="missing", data={"input": {}})],
    )
    agent = ReplayAgent([turn], tools=[FakeTool("other")])
    with pytest.raises(ToolNotFoundError):
        await agent.ainvoke({"messages": []})


async def test_replay_agent_astream_events_yields_then_executes(tmp_path):
    turn = RecordedTurn(
        phase="agent", user_message="hi",
        events=[
            RecordedEvent(event_type="on_tool_start", name="t", data={"input": {"k": "v"}}),
            RecordedEvent(event_type="on_tool_end", name="t", data={"output": "ok"}),
        ],
        response={"messages": [{"type": "ai", "content": "done"}]},
    )
    tool = FakeTool("t", output_dir=tmp_path)
    agent = ReplayAgent([turn], tools=[tool])

    yielded = []
    async for ev in agent.astream_events({"messages": []}):
        yielded.append(ev)

    # All recorded events plus a final on_chain_end with the response
    types = [e["event"] for e in yielded]
    assert "on_tool_start" in types
    assert "on_tool_end" in types
    assert types[-1] == "on_chain_end"
    assert (tmp_path / "t.txt").exists()  # tool actually ran


# ---------- ReplayAskHandler ----------

def test_ask_handler_returns_recorded_responses_in_order():
    handler = ReplayAskHandler([
        RecordedAskCall(prompt="p1", response="r1"),
        RecordedAskCall(prompt="p2", response="r2"),
    ])
    assert handler.get_next_response("any") == "r1"
    assert handler.get_next_response("any") == "r2"


def test_ask_handler_raises_when_exhausted():
    handler = ReplayAskHandler([])
    with pytest.raises(ReplayExhaustedError):
        handler.get_next_response("p")


# ---------- load_recording ----------

def test_load_recording_round_trip(tmp_path):
    target = tmp_path / "fx"
    target.mkdir()
    lines = [
        json.dumps({"type": "ask_call", "prompt": "p", "response": "r"}),
        json.dumps({
            "type": "turn",
            "phase": "agent",
            "user_message": "hi",
            "events": [{"event_type": "on_tool_start", "name": "t", "data": {"input": {}}}],
            "response": {},
        }),
    ]
    (target / "recording.jsonl").write_text("\n".join(lines))

    ask_calls, turns = load_recording(target)
    assert ask_calls == [RecordedAskCall(prompt="p", response="r")]
    assert len(turns) == 1
    assert turns[0].phase == "agent"
    assert turns[0].events[0].name == "t"


def test_load_recording_raises_when_missing(tmp_path):
    from langchain_replay.exceptions import FixtureNotFoundError
    with pytest.raises(FixtureNotFoundError):
        load_recording(tmp_path / "nope")
