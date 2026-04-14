"""Tests for RecordingAgentWrapper."""

import pytest

from langchain_replay.recorder import ConversationRecorder
from langchain_replay.wrapper import RecordingAgentWrapper


class FakeAgent:
    """Minimal stand-in for a LangChain agent.

    ``astream_events`` yields the configured events plus a final
    LangGraph ``on_chain_end`` carrying the response — matching how a
    real LangGraph runnable advertises its final state.
    """

    def __init__(self, response=None, events=None):
        self._response = response or {"messages": []}
        self._events = events or []
        self.invoke_calls = []
        self.stream_calls = []

    async def ainvoke(self, messages, **kwargs):
        self.invoke_calls.append((messages, kwargs))
        return self._response

    async def astream_events(self, messages, **kwargs):
        self.stream_calls.append((messages, kwargs))
        for ev in self._events:
            yield ev
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": self._response},
        }

    @property
    def custom_attr(self):
        return "forwarded"


async def test_ainvoke_records_a_turn_and_returns_response():
    response = {"messages": [], "structured_response": None}
    agent = FakeAgent(response=response)
    recorder = ConversationRecorder()
    wrapped = RecordingAgentWrapper(agent, recorder, phase="agent")

    result = await wrapped.ainvoke({"messages": [{"role": "user", "content": "hello"}]})

    assert result == response
    assert len(recorder.turns) == 1
    assert recorder.turns[0].phase == "agent"
    assert recorder.turns[0].user_message == "hello"


async def test_ainvoke_drives_astream_events_to_capture_tool_events():
    """Regression: a bare ainvoke on the wrapped agent yields no events,
    so the wrapper must drive astream_events internally to record tools."""
    events = [
        {"event": "on_tool_start", "name": "writer", "data": {"input": {"a": 1}}},
        {"event": "on_tool_end", "name": "writer", "data": {"output": "ok"}},
    ]
    agent = FakeAgent(events=events, response={"messages": []})
    recorder = ConversationRecorder()
    wrapped = RecordingAgentWrapper(agent, recorder)

    await wrapped.ainvoke({"messages": [{"role": "user", "content": "go"}]})

    assert agent.stream_calls, "wrapper should have called astream_events"
    assert agent.invoke_calls == [], "wrapper should NOT call the wrapped agent's ainvoke"
    assert len(recorder.turns) == 1
    event_types = [e.event_type for e in recorder.turns[0].events]
    assert "on_tool_start" in event_types
    assert "on_tool_end" in event_types


async def test_ainvoke_passes_version_v2_by_default():
    agent = FakeAgent()
    wrapped = RecordingAgentWrapper(agent, ConversationRecorder())
    await wrapped.ainvoke({"messages": [{"role": "user", "content": "x"}]})
    _, kwargs = agent.stream_calls[-1]
    assert kwargs.get("version") == "v2"


async def test_ainvoke_respects_explicit_version_kwarg():
    agent = FakeAgent()
    wrapped = RecordingAgentWrapper(agent, ConversationRecorder())
    await wrapped.ainvoke({"messages": [{"role": "user", "content": "x"}]}, version="v1")
    _, kwargs = agent.stream_calls[-1]
    assert kwargs.get("version") == "v1"


async def test_ainvoke_extracts_user_message_from_messages_dict():
    agent = FakeAgent()
    recorder = ConversationRecorder()
    wrapped = RecordingAgentWrapper(agent, recorder)
    await wrapped.ainvoke(
        {
            "messages": [
                {"role": "system", "content": "ignore me"},
                {"role": "user", "content": "the question"},
            ]
        }
    )
    assert recorder.turns[0].user_message == "the question"


async def test_astream_events_yields_each_event_and_records_them():
    events = [
        {"event": "on_tool_start", "name": "t", "data": {"input": {"x": 1}}},
        {"event": "on_tool_end", "name": "t", "data": {"output": "result"}},
    ]
    agent = FakeAgent(events=events)
    recorder = ConversationRecorder()
    wrapped = RecordingAgentWrapper(agent, recorder, phase="agent")

    yielded = []
    async for ev in wrapped.astream_events({"messages": [{"role": "user", "content": "go"}]}):
        yielded.append(ev)

    # FakeAgent always appends a synthetic LangGraph on_chain_end after the scripted events.
    assert yielded[:2] == events
    assert yielded[-1]["event"] == "on_chain_end"
    assert len(recorder.turns) == 1
    # The on_chain_end event is consumed by record_event but not stored as a regular event.
    assert len(recorder.turns[0].events) == 2


async def test_astream_events_finalizes_turn_on_exception():
    class BoomAgent:
        async def astream_events(self, *a, **kw):
            yield {"event": "on_tool_start", "name": "t", "data": {"input": {}}}
            raise RuntimeError("boom")

    recorder = ConversationRecorder()
    wrapped = RecordingAgentWrapper(BoomAgent(), recorder)

    with pytest.raises(RuntimeError):
        async for _ in wrapped.astream_events({"messages": [{"role": "user", "content": "x"}]}):
            pass

    # Even though the stream errored, the turn should be saved (with the partial events)
    assert len(recorder.turns) == 1


def test_forwards_other_attributes_to_wrapped_agent():
    agent = FakeAgent()
    wrapped = RecordingAgentWrapper(agent, ConversationRecorder())
    assert wrapped.custom_attr == "forwarded"
