"""Tests for the recorded event/turn/ask-call data classes."""

from langchain_replay._models import RecordedAskCall, RecordedEvent, RecordedTurn


def test_recorded_event_round_trip():
    event = RecordedEvent(event_type="on_tool_start", name="my_tool", data={"input": {"x": 1}})
    assert RecordedEvent.from_dict(event.to_dict()) == event


def test_recorded_event_defaults():
    event = RecordedEvent(event_type="on_chain_end")
    assert event.name == ""
    assert event.data == {}


def test_recorded_turn_round_trip():
    turn = RecordedTurn(
        phase="agent",
        user_message="hello",
        events=[RecordedEvent(event_type="on_tool_start", name="t", data={"input": {}})],
        response={"messages": [{"type": "ai", "content": "ok"}]},
    )
    restored = RecordedTurn.from_dict(turn.to_dict())
    assert restored.phase == "agent"
    assert restored.user_message == "hello"
    assert len(restored.events) == 1
    assert restored.events[0].name == "t"
    assert restored.response == {"messages": [{"type": "ai", "content": "ok"}]}


def test_recorded_turn_from_dict_tolerates_missing_optional_fields():
    turn = RecordedTurn.from_dict({"phase": "p", "user_message": "m"})
    assert turn.events == []
    assert turn.response == {}


def test_recorded_ask_call_round_trip():
    call = RecordedAskCall(prompt="why?", response="because")
    assert RecordedAskCall.from_dict(call.to_dict()) == call
