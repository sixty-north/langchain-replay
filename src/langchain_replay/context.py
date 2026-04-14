"""Context managers tying recording, replay, and patching together."""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack, contextmanager
from pathlib import Path
from typing import Iterator, Sequence
from unittest.mock import patch

from langchain_core.messages import AIMessage

from langchain_replay.exceptions import FixtureNotFoundError
from langchain_replay.recorder import ConversationRecorder
from langchain_replay.registry import AgentFactoryRegistry
from langchain_replay.replayer import ReplayAgent, ReplayAskHandler, load_recording
from langchain_replay.wrapper import RecordingAgentWrapper

ChatModelTarget = tuple[type, str]


class RecordingContext:
    """Patch registered agent factories and chat-model methods to capture a fixture."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        agent_registry: AgentFactoryRegistry,
        chat_models_to_patch: Sequence[ChatModelTarget] = (),
    ) -> None:
        self._recordings_dir = Path(recordings_dir)
        self._agent_registry = agent_registry
        self._chat_models = list(chat_models_to_patch)

    @contextmanager
    def for_fixture(self, master_type: str, fixture_name: str) -> Iterator[None]:
        recorder = ConversationRecorder()
        fixture_dir = self._recordings_dir / master_type / fixture_name

        def transform(original, *args, **kwargs):
            agent = original(*args, **kwargs)
            return RecordingAgentWrapper(agent, recorder)

        with ExitStack() as stack:
            stack.enter_context(self._agent_registry.patch_all(transform))
            for cls, method_name in self._chat_models:
                stack.enter_context(_patch_chat_model_for_recording(cls, method_name, recorder))
            try:
                yield
            finally:
                recorder.save(fixture_dir)


class ReplayContext:
    """Patch registered agent factories and chat-model methods to replay a fixture."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        agent_registry: AgentFactoryRegistry,
        chat_models_to_patch: Sequence[ChatModelTarget] = (),
    ) -> None:
        self._recordings_dir = Path(recordings_dir)
        self._agent_registry = agent_registry
        self._chat_models = list(chat_models_to_patch)

    @contextmanager
    def for_fixture(self, master_type: str, fixture_name: str) -> Iterator[None]:
        fixture_dir = self._recordings_dir / master_type / fixture_name
        if not fixture_dir.exists():
            raise FixtureNotFoundError(f"Fixture not found: {fixture_dir}. Record it first.")

        ask_calls, turns = load_recording(fixture_dir)
        ask_handler = ReplayAskHandler(ask_calls)

        captured_tools: list = []

        def transform(original, *args, **kwargs):
            tools = kwargs.get("tools")
            if tools is None:
                # Some factories take tools positionally — best-effort fallback.
                for arg in args:
                    if isinstance(arg, (list, tuple)) and arg and hasattr(arg[0], "name"):
                        tools = arg
                        break
            if tools:
                captured_tools[:] = list(tools)
            return ReplayAgent(turns, captured_tools)

        with ExitStack() as stack:
            stack.enter_context(self._agent_registry.patch_all(transform))
            for cls, method_name in self._chat_models:
                stack.enter_context(_patch_chat_model_for_replay(cls, method_name, ask_handler))
            yield


class AutoRecordReplayContext:
    """Replay if the fixture exists; record otherwise (or always when ``overwrite``)."""

    def __init__(
        self,
        recordings_dir: Path,
        *,
        agent_registry: AgentFactoryRegistry,
        chat_models_to_patch: Sequence[ChatModelTarget] = (),
        overwrite: bool = False,
    ) -> None:
        self._recordings_dir = Path(recordings_dir)
        self._agent_registry = agent_registry
        self._chat_models = list(chat_models_to_patch)
        self._overwrite = overwrite

    @contextmanager
    def for_fixture(self, master_type: str, fixture_name: str) -> Iterator[None]:
        fixture_dir = self._recordings_dir / master_type / fixture_name
        if fixture_dir.exists() and not self._overwrite:
            ctx: AbstractContextManager[None] = ReplayContext(
                self._recordings_dir,
                agent_registry=self._agent_registry,
                chat_models_to_patch=self._chat_models,
            ).for_fixture(master_type, fixture_name)
        else:
            ctx = RecordingContext(
                self._recordings_dir,
                agent_registry=self._agent_registry,
                chat_models_to_patch=self._chat_models,
            ).for_fixture(master_type, fixture_name)
        with ctx:
            yield


# ---- chat-model patching helpers ----


@contextmanager
def _patch_chat_model_for_recording(cls: type, method_name: str, recorder: ConversationRecorder) -> Iterator[None]:
    original = getattr(cls, method_name)

    async def recording_method(self, prompt, *args, **kwargs):
        response = await original(self, prompt, *args, **kwargs)
        prompt_str = prompt if isinstance(prompt, str) else str(prompt)
        response_str = response.content if hasattr(response, "content") else str(response)
        recorder.record_ask_call(prompt_str, response_str)
        return response

    with patch.object(cls, method_name, new=recording_method):
        yield


@contextmanager
def _patch_chat_model_for_replay(cls: type, method_name: str, ask_handler: ReplayAskHandler) -> Iterator[None]:
    async def replay_method(self, prompt, *args, **kwargs):
        return AIMessage(content=ask_handler.get_next_response(str(prompt)))

    with patch.object(cls, method_name, new=replay_method):
        yield
