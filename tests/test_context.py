"""End-to-end tests for RecordingContext / ReplayContext / AutoRecordReplayContext."""

import sys
import types
from pathlib import Path

import pytest

from langchain_replay.context import (
    AutoRecordReplayContext,
    RecordingContext,
    ReplayContext,
)
from langchain_replay.registry import AgentFactoryRegistry
from langchain_replay.replayer import ReplayAgent
from langchain_replay.wrapper import RecordingAgentWrapper


# ---------- Fake agent module ----------

class FakeRealAgent:
    """A "real" agent that we record/replay against."""

    def __init__(self, tools, scripted_events, scripted_response):
        self.tools = tools
        self._events = scripted_events
        self._response = scripted_response

    async def ainvoke(self, messages, **kwargs):
        for ev in self._events:
            if ev["event"] == "on_tool_start":
                tool = next(t for t in self.tools if t.name == ev["name"])
                await tool.ainvoke(ev["data"]["input"])
        return self._response

    async def astream_events(self, messages, **kwargs):
        for ev in self._events:
            yield ev
            if ev["event"] == "on_tool_start":
                tool = next(t for t in self.tools if t.name == ev["name"])
                await tool.ainvoke(ev["data"]["input"])
        yield {"event": "on_chain_end", "name": "LangGraph", "data": {"output": self._response}}


class FakeTool:
    def __init__(self, name, output_dir: Path):
        self.name = name
        self._output_dir = output_dir
        self.invocations = []

    async def ainvoke(self, tool_input):
        self.invocations.append(tool_input)
        (self._output_dir / f"{self.name}.txt").write_text(str(tool_input))
        return f"{self.name}-ok"


@pytest.fixture
def fake_factory_module():
    """A fake module with a create_agent function we can register and patch."""
    mod = types.ModuleType("_replay_ctx_factory")

    def create_agent(tools, scripted_events, scripted_response):
        return FakeRealAgent(tools, scripted_events, scripted_response)

    mod.create_agent = create_agent
    sys.modules["_replay_ctx_factory"] = mod
    yield mod
    del sys.modules["_replay_ctx_factory"]


@pytest.fixture
def registry(fake_factory_module):
    reg = AgentFactoryRegistry()
    reg.register("_replay_ctx_factory.create_agent")
    return reg


# ---------- RecordingContext ----------

async def _drive(agent, messages):
    """Drive an agent via astream_events (so the recorder captures tool events)."""
    async for _ in agent.astream_events(messages):
        pass


async def test_ainvoke_only_workflow_records_tool_events_and_replays(
    tmp_path, registry, fake_factory_module
):
    """Mirrors the README example: user calls ``await agent.ainvoke(...)``
    once to record, then again to replay. The wrapper must drive
    astream_events under the hood so that tool events end up in the
    fixture and replay re-executes the tool live.
    """
    recordings = tmp_path / "recordings"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    scripted_events = [
        {"event": "on_tool_start", "name": "writer", "data": {"input": {"a": 1}}},
        {"event": "on_tool_end", "name": "writer", "data": {"output": "ok"}},
    ]
    scripted_response = {"messages": [{"type": "ai", "content": "done"}]}

    # ---- Record ----
    record_tool = FakeTool("writer", out_dir)
    rec_ctx = RecordingContext(recordings, agent_registry=registry)
    with rec_ctx.for_fixture("e2e", "case"):
        agent = fake_factory_module.create_agent(
            tools=[record_tool],
            scripted_events=scripted_events,
            scripted_response=scripted_response,
        )
        # Note: plain ainvoke, no stream driving
        await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    # The recording should contain the tool events
    import json
    lines = (recordings / "e2e" / "case" / "recording.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    turn = next(p for p in parsed if p["type"] == "turn")
    event_names = [e["event_type"] for e in turn["events"]]
    assert "on_tool_start" in event_names
    assert "on_tool_end" in event_names

    # ---- Replay ----
    (out_dir / "writer.txt").unlink()
    replay_tool = FakeTool("writer", out_dir)
    rep_ctx = ReplayContext(recordings, agent_registry=registry)
    with rep_ctx.for_fixture("e2e", "case"):
        agent = fake_factory_module.create_agent(
            tools=[replay_tool],
            scripted_events=[],
            scripted_response={},
        )
        await agent.ainvoke({"messages": []})

    # The replay tool was actually invoked with the recorded input
    assert (out_dir / "writer.txt").exists()
    assert replay_tool.invocations == [{"a": 1}]


async def test_recording_context_writes_fixture_files(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    tool = FakeTool("writer", out_dir)

    ctx = RecordingContext(recordings, agent_registry=registry)
    with ctx.for_fixture("group", "case1"):
        agent = fake_factory_module.create_agent(
            tools=[tool],
            scripted_events=[
                {"event": "on_tool_start", "name": "writer", "data": {"input": {"a": 1}}},
                {"event": "on_tool_end", "name": "writer", "data": {"output": "ok"}},
            ],
            scripted_response={"messages": []},
        )
        # The agent is wrapped — must be a RecordingAgentWrapper
        assert isinstance(agent, RecordingAgentWrapper)
        await _drive(agent, {"messages": [{"role": "user", "content": "go"}]})

    fixture_dir = recordings / "group" / "case1"
    assert (fixture_dir / "recording.jsonl").exists()
    assert (fixture_dir / "metadata.json").exists()
    # Tool actually executed during recording
    assert (out_dir / "writer.txt").exists()
    assert tool.invocations == [{"a": 1}]


# ---------- ReplayContext ----------

async def test_replay_context_replays_recorded_fixture(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    record_tool = FakeTool("writer", out_dir)

    # First, record
    rec_ctx = RecordingContext(recordings, agent_registry=registry)
    with rec_ctx.for_fixture("group", "case1"):
        agent = fake_factory_module.create_agent(
            tools=[record_tool],
            scripted_events=[
                {"event": "on_tool_start", "name": "writer", "data": {"input": {"a": 1}}},
                {"event": "on_tool_end", "name": "writer", "data": {"output": "ok"}},
            ],
            scripted_response={"messages": [{"type": "ai", "content": "done"}]},
        )
        await _drive(agent, {"messages": [{"role": "user", "content": "go"}]})

    # Now replay — clear the output file to prove the replay re-executes the tool
    (out_dir / "writer.txt").unlink()
    replay_tool = FakeTool("writer", out_dir)

    rep_ctx = ReplayContext(recordings, agent_registry=registry)
    with rep_ctx.for_fixture("group", "case1"):
        agent = fake_factory_module.create_agent(
            tools=[replay_tool],
            scripted_events=[],  # ignored — replay returns ReplayAgent
            scripted_response={},
        )
        assert isinstance(agent, ReplayAgent)
        await agent.ainvoke({"messages": []})

    # Replay tool actually executed
    assert (out_dir / "writer.txt").exists()
    assert replay_tool.invocations == [{"a": 1}]


async def test_replay_context_raises_on_missing_fixture(tmp_path, registry):
    from langchain_replay.exceptions import FixtureNotFoundError
    rep_ctx = ReplayContext(tmp_path, agent_registry=registry)
    with pytest.raises(FixtureNotFoundError):
        with rep_ctx.for_fixture("nope", "missing"):
            pass


# ---------- AutoRecordReplayContext ----------

async def test_auto_records_first_time_then_replays(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    tool1 = FakeTool("writer", out_dir)

    auto = AutoRecordReplayContext(recordings, agent_registry=registry)

    # First call: records (because the fixture doesn't exist yet)
    with auto.for_fixture("g", "c"):
        agent = fake_factory_module.create_agent(
            tools=[tool1],
            scripted_events=[
                {"event": "on_tool_start", "name": "writer", "data": {"input": {"a": 1}}},
            ],
            scripted_response={"messages": []},
        )
        await _drive(agent, {"messages": [{"role": "user", "content": "go"}]})

    assert (recordings / "g" / "c" / "recording.jsonl").exists()

    # Second call: replays — clear evidence and use a new tool
    (out_dir / "writer.txt").unlink()
    tool2 = FakeTool("writer", out_dir)
    with auto.for_fixture("g", "c"):
        agent = fake_factory_module.create_agent(
            tools=[tool2],
            scripted_events=[],
            scripted_response={},
        )
        assert isinstance(agent, ReplayAgent)
        await agent.ainvoke({"messages": []})

    assert tool2.invocations == [{"a": 1}]


async def test_auto_overwrite_re_records(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Pre-create a fixture file
    fixture_dir = recordings / "g" / "c"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "recording.jsonl").write_text("")

    auto = AutoRecordReplayContext(recordings, agent_registry=registry, overwrite=True)
    with auto.for_fixture("g", "c"):
        agent = fake_factory_module.create_agent(
            tools=[FakeTool("writer", out_dir)],
            scripted_events=[],
            scripted_response={"messages": []},
        )
        # When overwriting, we get a recording wrapper, not a replay agent
        assert isinstance(agent, RecordingAgentWrapper)
        await agent.ainvoke({"messages": [{"role": "user", "content": "go"}]})


# ---------- chat-model patching for direct ask() calls ----------

class FakeChatModel:
    """A "real" chat model class with an ainvoke method."""

    async def ainvoke(self, prompt, **kwargs):
        return _AIResponse("REAL-LLM-OUTPUT")


class _AIResponse:
    def __init__(self, content):
        self.content = content


async def test_recording_context_records_chat_model_calls(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"
    ctx = RecordingContext(
        recordings,
        agent_registry=registry,
        chat_models_to_patch=[(FakeChatModel, "ainvoke")],
    )
    with ctx.for_fixture("g", "c"):
        model = FakeChatModel()
        result = await model.ainvoke("question?")
        # The original is still called and returned
        assert result.content == "REAL-LLM-OUTPUT"

    # Recording contains an ask_call line
    import json
    lines = (recordings / "g" / "c" / "recording.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    ask_calls = [p for p in parsed if p["type"] == "ask_call"]
    assert len(ask_calls) == 1
    assert ask_calls[0]["prompt"] == "question?"
    assert ask_calls[0]["response"] == "REAL-LLM-OUTPUT"


async def test_replay_context_replays_chat_model_calls(tmp_path, registry, fake_factory_module):
    recordings = tmp_path / "recordings"

    # Record
    rec = RecordingContext(
        recordings, agent_registry=registry,
        chat_models_to_patch=[(FakeChatModel, "ainvoke")],
    )
    with rec.for_fixture("g", "c"):
        await FakeChatModel().ainvoke("q")

    # Replay
    rep = ReplayContext(
        recordings, agent_registry=registry,
        chat_models_to_patch=[(FakeChatModel, "ainvoke")],
    )
    with rep.for_fixture("g", "c"):
        # Even though FakeChatModel returns "REAL-LLM-OUTPUT", replay should yield the recorded value
        result = await FakeChatModel().ainvoke("q")
        from langchain_core.messages import AIMessage
        assert isinstance(result, AIMessage)
        assert result.content == "REAL-LLM-OUTPUT"
