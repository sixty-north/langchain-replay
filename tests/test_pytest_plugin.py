"""Tests for the pytest plugin, exercised via pytester."""

import textwrap

import pytest

pytest_plugins = ["pytester"]


def test_record_replay_fixture_is_none_without_flag(pytester: pytest.Pytester):
    pytester.makepyfile(
        textwrap.dedent(
            """
            def test_no_flag(record_replay):
                assert record_replay is None
            """
        )
    )
    result = pytester.runpytest()
    result.assert_outcomes(passed=1)


def test_record_replay_fixture_returns_context_with_flag(pytester: pytest.Pytester):
    pytester.makepyfile(
        textwrap.dedent(
            """
            from langchain_replay import AutoRecordReplayContext

            def test_with_flag(record_replay):
                assert isinstance(record_replay, AutoRecordReplayContext)
            """
        )
    )
    result = pytester.runpytest("--record-fixtures")
    result.assert_outcomes(passed=1)


def test_overwrite_flag_propagates(pytester: pytest.Pytester):
    pytester.makepyfile(
        textwrap.dedent(
            """
            def test_overwrite_set(record_replay):
                assert record_replay._overwrite is True
            """
        )
    )
    result = pytester.runpytest("--record-fixtures", "--overwrite-fixtures")
    result.assert_outcomes(passed=1)


def test_user_can_override_registry_fixture(pytester: pytest.Pytester):
    pytester.makeconftest(
        textwrap.dedent(
            """
            import pytest
            from langchain_replay import AgentFactoryRegistry

            @pytest.fixture
            def langchain_replay_registry():
                reg = AgentFactoryRegistry()
                # we don't register any real factories — just verify identity
                reg._marker = "user_provided"
                return reg
            """
        )
    )
    pytester.makepyfile(
        textwrap.dedent(
            """
            def test_uses_overridden_registry(record_replay):
                assert record_replay._agent_registry._marker == "user_provided"
            """
        )
    )
    result = pytester.runpytest("--record-fixtures")
    result.assert_outcomes(passed=1)
