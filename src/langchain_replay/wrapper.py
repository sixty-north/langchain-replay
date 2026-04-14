"""Transparent wrapper that records LangChain agent calls."""

from __future__ import annotations

from typing import Any

from langchain_replay.recorder import ConversationRecorder


class RecordingAgentWrapper:
    """Wrap a LangChain agent so that ``ainvoke`` and ``astream_events`` calls are recorded.

    Other attributes are forwarded transparently to the wrapped agent.
    """

    def __init__(self, agent: Any, recorder: ConversationRecorder, phase: str = "agent") -> None:
        self._agent = agent
        self._recorder = recorder
        self._phase = phase

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """Record an ainvoke call.

        Internally drives ``astream_events`` so that intermediate tool
        events are captured (a plain ``ainvoke`` on the wrapped agent
        emits no event stream, leaving the recording empty of tool calls).
        The final response is reconstructed from the LangGraph
        ``on_chain_end`` event.
        """
        self._recorder.start_turn(self._phase, self._extract_user_message(messages))
        final_response: Any = None
        stream_kwargs = dict(kwargs)
        stream_kwargs.setdefault("version", "v2")
        try:
            async for event in self._agent.astream_events(messages, **stream_kwargs):
                self._recorder.record_event(event)
                if event.get("event") == "on_chain_end" and event.get("name") == "LangGraph":
                    final_response = event.get("data", {}).get("output")
        finally:
            self._recorder.end_turn()
        return final_response

    async def astream_events(self, messages: Any, **kwargs: Any):
        self._recorder.start_turn(self._phase, self._extract_user_message(messages))
        try:
            async for event in self._agent.astream_events(messages, **kwargs):
                self._recorder.record_event(event)
                yield event
        finally:
            self._recorder.end_turn()

    @staticmethod
    def _extract_user_message(messages: Any) -> str:
        if isinstance(messages, dict) and "messages" in messages:
            for msg in messages["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    return msg.get("content", "")
        return ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self._agent, name)
