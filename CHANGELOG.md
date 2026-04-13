# Changelog

## 0.1.0 (unreleased)

Initial release.

- `ConversationRecorder`, `RecordingAgentWrapper`, `ReplayAgent` — core record/replay primitives
- `RecordingContext`, `ReplayContext`, `AutoRecordReplayContext` — context managers
- `AgentFactoryRegistry` — register your own agent factory dotted paths to be patched
- `DefaultInputSerializer` — JSON serializer for fixture inputs (dict, dataclass, pydantic)
- Optional pytest plugin via `[pytest]` extra

Recorded tool inputs are replayed verbatim. Tests are expected to be deterministic at the prompt level — see "Tests must be deterministic" in the README.
