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
filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".md"
messages = {
    "messages": [
        {
            "role": "user",
            "content": f"Briefly explain {topic} and write it to a file called {filename}.",
        }
    ]
}


async def main():
    with ctx.for_fixture("my_test", "first_run"):
        agent = langchain.agents.create_agent(model="claude-haiku-4-5-20251001", tools=tools)
        result = await agent.ainvoke(messages)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
```

The first run records `tests/replay-recordings/my_test/first_run/recording.jsonl` and writes the markdown file for real. Subsequent runs replay the LLM's decisions without any API calls — but `WriteFileTool` still executes, so the file is recreated on every run.

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

## Tests must be deterministic

`langchain-replay` records the LLM's *decisions*, not the universe those decisions were made in. On replay, recorded tool inputs are dispatched verbatim. If your test feeds non-deterministic values into the LLM prompt — a fresh timestamp, a `uuid.uuid4()`, a `tmp_path` from pytest — those values get baked into the recorded tool calls and replayed exactly as they were captured. The replay run will not see today's timestamp; it will see the recording day's timestamp.

For most workflows this is invisible: the LLM is asked to do the same thing, calls the same tool with the same arguments, and the test passes. The trouble starts when the recorded value points at something that no longer exists on the next run (a `tmp_path` directory) or when the test asserts on a value that *should* be fresh.

The fix is at the test level, not the library level:

- **Don't put per-run values into the prompt.** If the test needs a unique filename, let the LLM choose one, or pass the path to the tool out-of-band rather than via the user message.
- **Use stable inputs in fixtures.** A test that records `{"date": "2026-01-01"}` is reproducible; one that records `{"date": datetime.now().isoformat()}` is not.
- **Keep ephemeral filesystem state out of LLM-visible inputs.** Configure tools with stable roots and let the LLM produce relative names.

If you find a class of non-determinism that genuinely cannot be designed away, open an issue describing the workflow — that's the kind of feedback that will shape the post-0.1 API.

## License

Apache 2.0.
