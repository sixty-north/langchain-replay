"""langchain-replay: record and replay LangChain agent decisions while tools execute live."""

from langchain_replay._models import RecordedAskCall, RecordedEvent, RecordedTurn
from langchain_replay.context import (
    AutoRecordReplayContext,
    RecordingContext,
    ReplayContext,
)
from langchain_replay.exceptions import (
    FixtureNotFoundError,
    RecordReplayError,
    ReplayExhaustedError,
    ToolNotFoundError,
)
from langchain_replay.recorder import ConversationRecorder
from langchain_replay.registry import AgentFactoryRegistry
from langchain_replay.replayer import ReplayAgent, ReplayAskHandler, load_recording
from langchain_replay.wrapper import RecordingAgentWrapper

__all__ = [
    # Core
    "ConversationRecorder",
    "ReplayAgent",
    "ReplayAskHandler",
    "RecordingAgentWrapper",
    "load_recording",
    # Contexts
    "RecordingContext",
    "ReplayContext",
    "AutoRecordReplayContext",
    # Extension points
    "AgentFactoryRegistry",
    # Data
    "RecordedTurn",
    "RecordedEvent",
    "RecordedAskCall",
    # Errors
    "RecordReplayError",
    "ReplayExhaustedError",
    "FixtureNotFoundError",
    "ToolNotFoundError",
]
