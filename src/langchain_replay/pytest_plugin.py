"""Pytest plugin for langchain-replay.

Loaded eagerly via the ``pytest11`` entry point. Provides CLI options and a
``record_replay`` fixture that builds an :class:`AutoRecordReplayContext`
using user-defined configuration fixtures.

Users override these fixtures in their own ``conftest.py``:

- ``langchain_replay_registry`` — an :class:`AgentFactoryRegistry`
- ``langchain_replay_recordings_dir`` — a ``pathlib.Path``
- ``langchain_replay_chat_models`` (optional) — list of ``(class, method)`` tuples
"""

from __future__ import annotations

from pathlib import Path

import pytest

from langchain_replay.context import AutoRecordReplayContext
from langchain_replay.registry import AgentFactoryRegistry


def pytest_addoption(parser):
    group = parser.getgroup("langchain-replay")
    group.addoption(
        "--record-fixtures",
        action="store_true",
        default=False,
        help="Enable langchain-replay recording/replay (without it, the record_replay fixture is None)",
    )
    group.addoption(
        "--overwrite-fixtures",
        action="store_true",
        default=False,
        help="Force re-recording even if a fixture exists",
    )


# ---- user-overridable configuration fixtures ----


@pytest.fixture
def langchain_replay_registry() -> AgentFactoryRegistry:
    """Override in your conftest.py to return your AgentFactoryRegistry."""
    return AgentFactoryRegistry()


@pytest.fixture
def langchain_replay_recordings_dir(tmp_path_factory) -> Path:
    """Override in your conftest.py to point at your fixtures directory."""
    return tmp_path_factory.mktemp("langchain_replay_fixtures")


@pytest.fixture
def langchain_replay_chat_models() -> list:
    """Override to return a list of ``(ChatModelClass, "method_name")`` tuples to patch."""
    return []


# ---- main fixture ----


@pytest.fixture
def record_replay(
    request,
    langchain_replay_registry,
    langchain_replay_recordings_dir,
    langchain_replay_chat_models,
) -> AutoRecordReplayContext | None:
    """Returns an :class:`AutoRecordReplayContext`, or ``None`` if --record-fixtures is unset."""
    if not request.config.getoption("--record-fixtures"):
        return None
    return AutoRecordReplayContext(
        langchain_replay_recordings_dir,
        agent_registry=langchain_replay_registry,
        chat_models_to_patch=langchain_replay_chat_models,
        overwrite=request.config.getoption("--overwrite-fixtures"),
    )
