"""Tests for the agent factory registry."""

import sys
import types

import pytest

from langchain_replay.registry import AgentFactoryRegistry


@pytest.fixture
def fake_module():
    """Create a temporary module with a stub agent factory."""
    mod = types.ModuleType("_replay_fake_factory")

    def create_agent(model=None, tools=None):
        return {"kind": "real_agent", "model": model, "tools": tools}

    mod.create_agent = create_agent
    sys.modules["_replay_fake_factory"] = mod
    yield mod
    del sys.modules["_replay_fake_factory"]


def test_register_validates_path_exists(fake_module):
    reg = AgentFactoryRegistry()
    reg.register("_replay_fake_factory.create_agent")  # ok
    assert "_replay_fake_factory.create_agent" in reg.targets


def test_register_raises_on_unknown_module():
    reg = AgentFactoryRegistry()
    with pytest.raises(ImportError):
        reg.register("definitely.not.a.real.module.thing")


def test_register_raises_on_unknown_attribute(fake_module):
    reg = AgentFactoryRegistry()
    with pytest.raises(AttributeError):
        reg.register("_replay_fake_factory.no_such_function")


def test_register_rejects_non_dotted_path():
    reg = AgentFactoryRegistry()
    with pytest.raises(ValueError):
        reg.register("nodot")


def test_patch_all_intercepts_factory_calls(fake_module):
    reg = AgentFactoryRegistry()
    reg.register("_replay_fake_factory.create_agent")

    captured = []

    def transform(original, *args, **kwargs):
        captured.append((args, kwargs))
        agent = original(*args, **kwargs)
        agent["wrapped"] = True
        return agent

    with reg.patch_all(transform):
        result = fake_module.create_agent(model="m", tools=["t1"])

    assert result == {"kind": "real_agent", "model": "m", "tools": ["t1"], "wrapped": True}
    assert captured == [((), {"model": "m", "tools": ["t1"]})]


def test_patch_all_restores_original_on_exit(fake_module):
    reg = AgentFactoryRegistry()
    reg.register("_replay_fake_factory.create_agent")
    original = fake_module.create_agent

    with reg.patch_all(lambda orig, *a, **kw: orig(*a, **kw)):
        assert fake_module.create_agent is not original

    assert fake_module.create_agent is original


def test_patch_all_restores_original_on_exception(fake_module):
    reg = AgentFactoryRegistry()
    reg.register("_replay_fake_factory.create_agent")
    original = fake_module.create_agent

    with pytest.raises(RuntimeError):
        with reg.patch_all(lambda orig, *a, **kw: orig(*a, **kw)):
            raise RuntimeError("boom")

    assert fake_module.create_agent is original


def test_patch_all_handles_multiple_targets(fake_module):
    # Register a second target on the same module
    def other_factory():
        return "other"

    fake_module.other_factory = other_factory

    reg = AgentFactoryRegistry()
    reg.register("_replay_fake_factory.create_agent")
    reg.register("_replay_fake_factory.other_factory")

    def transform(original, *args, **kwargs):
        return f"wrapped({original(*args, **kwargs)})"

    with reg.patch_all(transform):
        assert fake_module.create_agent() == "wrapped({'kind': 'real_agent', 'model': None, 'tools': None})"
        assert fake_module.other_factory() == "wrapped(other)"


def test_empty_registry_patch_all_is_a_noop():
    reg = AgentFactoryRegistry()
    with reg.patch_all(lambda orig, *a, **kw: orig(*a, **kw)):
        pass  # should not raise
