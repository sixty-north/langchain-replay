"""Exception hierarchy for langchain-replay."""


class RecordReplayError(Exception):
    """Base class for all langchain-replay errors."""


class ReplayExhaustedError(RecordReplayError):
    """Raised when a replay is asked for more turns or ask-calls than were recorded."""


class FixtureNotFoundError(RecordReplayError, FileNotFoundError):
    """Raised when a fixture directory does not exist during replay."""


class ToolNotFoundError(RecordReplayError):
    """Raised when a recorded tool name does not match any tool provided to the replay agent."""
