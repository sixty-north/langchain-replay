# langchain-replay

Record and replay LangChain/LangGraph agent decisions while tools execute against the real filesystem.

Testing LLM agents is expensive and non-deterministic. HTTP-level cassette libraries (VCR.py, `responses`) record raw requests, but they never let your tool code actually run, so tests stop reflecting reality.

`langchain-replay` takes a different approach: it records the LLM's *decisions* (which tool to call, with what arguments, what text to return), then on replay it yields those recorded decisions while *actually executing the tools*. Your tests exercise real tool code paths without paying for LLM calls.

## Status

Pre-1.0 (0.x). The API will change as real usage exposes rough edges. Pin tightly.

## Installation

```bash
pip install langchain-replay
# Optional pytest plugin:
pip install "langchain-replay[pytest]"
```

## 30-second example

```python
import asyncio
from datetime import datetime
from pathlib import Path

import langchain.agents  # see "Where to patch" below — do NOT use `from langchain.agents import create_agent`
from langchain_community.tools import WriteFileTool

from langchain_replay import AgentFactoryRegistry, AutoRecordReplayContext

registry = AgentFactoryRegistry()
registry.register("langchain.agents.create_agent")

ctx = AutoRecordReplayContext(Path("tests/replay-recordings"), agent_registry=registry)

tools = [WriteFileTool(root_dir=str(Path(__file__).parent.absolute()))]

topic = "Vapnik-Chervonenkis dimension"
messages = {
    "messages": [
        {
            "role": "user",
            "content": f"Briefly explain {topic} and write it to a file called vcd.md.",
        }
    ]
}


async def main():
    with ctx.for_fixture("my_test", "vcd_explanation"):
        agent = langchain.agents.create_agent(model="claude-haiku-4-5-20251001", tools=tools)
        result = await agent.ainvoke(messages)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The first run records `tests/replay-recordings/my_test/vcd_explanation/recording.jsonl` and writes the markdown file for real. Subsequent runs replay the LLM's decisions without any API calls — but `WriteFileTool` still executes, so the file is recreated on every run.

## Where to patch — important

`langchain-replay` swaps factory functions in their defining module. If you write:

```python
from langchain.agents import create_agent  # binds to your module's namespace

registry.register("langchain.agents.create_agent")
agent = create_agent(model=..., tools=...)  # ⚠️ unpatched — your local binding is still the original
```

…then the patch on `langchain.agents.create_agent` does **not** affect your module's local `create_agent` symbol, and the recording will be empty. This is the standard `unittest.mock.patch` "where to patch" gotcha.

**Do this instead** (one of):

```python
# Option A: import the module, call through the namespace
import langchain.agents
agent = langchain.agents.create_agent(model=..., tools=...)
```

```python
# Option B: register where the symbol was imported to
from langchain.agents import create_agent
registry.register("my_test_module.create_agent")
agent = create_agent(model=..., tools=...)
```

Option A is the recommended pattern.

## Pytest configuration

`langchain-replay` ships an optional pytest plugin. Install with the `[pytest]` extra:

```bash
pip install "langchain-replay[pytest]"
```

The plugin adds two CLI flags: `--record-fixtures` (enable recording/replay) and `--overwrite-fixtures` (force re-record). It exposes a `record_replay` fixture that returns an `AutoRecordReplayContext` when `--record-fixtures` is set, or `None` otherwise.

### Minimal setup

Override the `langchain_replay_registry` and `langchain_replay_recordings_dir` fixtures in your `conftest.py`:

```python
# conftest.py
import pytest
from pathlib import Path
from langchain_replay import AgentFactoryRegistry

@pytest.fixture
def langchain_replay_registry():
    registry = AgentFactoryRegistry()
    registry.register("langchain.agents.create_agent")
    return registry

@pytest.fixture
def langchain_replay_recordings_dir():
    return Path(__file__).parent / "fixtures" / "recordings"
```

Then use the `record_replay` fixture in your tests:

```python
@pytest.mark.costly
async def test_my_agent(record_replay):
    if record_replay:
        with record_replay.for_fixture("agents", "my_scenario"):
            result = await run_my_agent()
    else:
        result = await run_my_agent()

    assert result.status == "ok"
```

Run `pytest --record-fixtures` once to record; subsequent runs replay without API calls.

### Gating expensive tests with a marker

The example above uses `@pytest.mark.costly` to mark tests that make real API calls. This is a useful pattern but not built into the library — you wire it up yourself in `conftest.py`:

```python
# conftest.py
def pytest_addoption(parser):
    parser.addoption(
        "--run-costly", action="store_true", default=False,
        help="Run tests that incur external API costs",
    )

def pytest_configure(config):
    config.addinivalue_line("markers", "costly: mark test as incurring external API costs")

def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-costly"):
        skip = pytest.mark.skip(reason="needs --run-costly option")
        for item in items:
            if "costly" in item.keywords:
                item.add_marker(skip)
```

Now `pytest` skips `@pytest.mark.costly` tests by default, and `pytest --run-costly --record-fixtures` enables both the marker and recording. This keeps your CI fast while letting you record new fixtures on demand.

### Full setup with chat-model patching

If your code also calls a chat model directly (outside of an agent), override `langchain_replay_chat_models` to intercept those calls:

```python
# conftest.py
import pytest
from pathlib import Path
from langchain_anthropic import ChatAnthropic
from langchain_replay import AgentFactoryRegistry

@pytest.fixture
def langchain_replay_registry():
    registry = AgentFactoryRegistry()
    registry.register("langchain.agents.create_agent")
    registry.register("myproject.agents.build_agent")  # your own factories
    return registry

@pytest.fixture
def langchain_replay_recordings_dir():
    return Path(__file__).parent / "fixtures" / "recordings"

@pytest.fixture
def langchain_replay_chat_models():
    return [(ChatAnthropic, "ainvoke")]
```

### Disabling the plugin

If you use the library's classes directly (e.g. building `AutoRecordReplayContext` yourself in a custom fixture) and don't need the plugin's CLI flags or fixtures, disable it to avoid conflicts:

```toml
# pyproject.toml
[tool.pytest.ini_options]
addopts = "-p no:langchain_replay"
```

This is useful when your project already defines its own `--record-fixtures` or `--overwrite-fixtures` options.

## Using with unittest

The core classes are plain context managers with no pytest dependency. To use with `unittest`, build the registry and context directly and control recording via an environment variable:

```python
import asyncio
import os
import unittest
from pathlib import Path

import langchain.agents
from langchain_replay import AgentFactoryRegistry, AutoRecordReplayContext

RECORDINGS = Path(__file__).parent / "recordings"
RECORD = os.environ.get("RECORD_FIXTURES") == "1"
OVERWRITE = os.environ.get("OVERWRITE_FIXTURES") == "1"

registry = AgentFactoryRegistry()
registry.register("langchain.agents.create_agent")

ctx = AutoRecordReplayContext(RECORDINGS, agent_registry=registry, overwrite=OVERWRITE)


class TestMyAgent(unittest.IsolatedAsyncioTestCase):

    async def test_explains_topic(self):
        if RECORD:
            with ctx.for_fixture("agents", "explain_topic"):
                result = await self.run_agent()
        else:
            result = await self.run_agent()
        self.assertIn("messages", result)

    async def run_agent(self):
        agent = langchain.agents.create_agent(
            model="claude-haiku-4-5-20251001", tools=[...]
        )
        return await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Explain monads."}]}
        )
```

```bash
# First run — records fixtures (makes real API calls)
RECORD_FIXTURES=1 python -m unittest

# Subsequent runs — replays without API calls
python -m unittest

# Re-record existing fixtures
RECORD_FIXTURES=1 OVERWRITE_FIXTURES=1 python -m unittest
```

## Tests must be deterministic

`langchain-replay` records the LLM's *decisions*, not the universe those decisions were made in. On replay, recorded tool inputs are dispatched verbatim. If your test feeds non-deterministic values into the LLM prompt — a fresh timestamp, a `uuid.uuid4()`, a `tmp_path` from pytest — those values get baked into the recorded tool calls and replayed exactly as they were captured. The replay run will not see today's timestamp; it will see the recording day's timestamp.

For most workflows this is invisible: the LLM is asked to do the same thing, calls the same tool with the same arguments, and the test passes. The trouble starts when the recorded value points at something that no longer exists on the next run (a `tmp_path` directory) or when the test asserts on a value that *should* be fresh.

The fix is at the test level, not the library level:

- **Don't put per-run values into the prompt.** If the test needs a unique filename, let the LLM choose one, or pass the path to the tool out-of-band rather than via the user message.
- **Use stable inputs in fixtures.** A test that records `{"date": "2026-01-01"}` is reproducible; one that records `{"date": datetime.now().isoformat()}` is not.
- **Keep ephemeral filesystem state out of LLM-visible inputs.** Configure tools with stable roots and let the LLM produce relative names.

If you find a class of non-determinism that genuinely cannot be designed away, open an issue describing the workflow — that's the kind of feedback that will shape the post-0.1 API.

## Releasing

Versioning is managed with [bump-my-version](https://github.com/callowayproject/bump-my-version). It updates the version in `pyproject.toml`, commits the change, and creates a Git tag in one step.

```bash
# Install dev dependencies (includes bump-my-version)
uv sync --group dev

# Bump the patch version: 0.1.0 -> 0.1.1
uv run bump-my-version bump patch

# Bump the minor version: 0.1.1 -> 0.2.0
uv run bump-my-version bump minor

# Bump the major version: 0.2.0 -> 1.0.0
uv run bump-my-version bump major
```

To preview what a bump would do without changing anything:

```bash
uv run bump-my-version bump patch --dry-run --verbose
```

After bumping, push the commit and tag together:

```bash
git push && git push --tags
```

To build and publish to PyPI:

```bash
uv build
uv publish
```

## License

Apache 2.0.
