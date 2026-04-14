"""Registry of agent factory dotted paths to be patched during record/replay.

The original record/replay implementation hardcoded ``unittest.mock.patch``
calls for five specific factory entry points. The registry replaces that
fork-required coupling: users register their own factory dotted paths,
and the context managers patch each one with a user-supplied transform
when entered.
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from typing import Callable, Iterator


class AgentFactoryRegistry:
    """Registry of dotted paths to agent factory functions to be patched."""

    def __init__(self) -> None:
        self._targets: list[str] = []

    @property
    def targets(self) -> list[str]:
        """Return a copy of the registered dotted paths."""
        return list(self._targets)

    def register(self, dotted_path: str) -> None:
        """Register a factory dotted path. Raises immediately on typos.

        Validating at registration time means tests fail fast on a bad path
        rather than silently skipping the patch (and silently never recording).
        """
        module_path, _, attr = dotted_path.rpartition(".")
        if not module_path:
            raise ValueError(f"dotted_path must contain a module and attribute, got: {dotted_path!r}")
        module = importlib.import_module(module_path)
        if not hasattr(module, attr):
            raise AttributeError(f"Module {module_path!r} has no attribute {attr!r}")
        if dotted_path not in self._targets:
            self._targets.append(dotted_path)

    @contextmanager
    def patch_all(
        self,
        transform: Callable[..., object],
    ) -> Iterator[None]:
        """Patch every registered factory with ``transform``.

        ``transform`` is called as ``transform(original, *args, **kwargs)``
        for each factory invocation. It is responsible for either calling
        ``original`` or ignoring it (e.g., the replay path returns a fake
        agent without invoking the real factory).
        """
        originals: list[tuple[object, str, object]] = []
        try:
            for dotted_path in self._targets:
                module_path, _, attr = dotted_path.rpartition(".")
                module = importlib.import_module(module_path)
                original = getattr(module, attr)
                originals.append((module, attr, original))

                def make_wrapper(orig):
                    def wrapper(*args, **kwargs):
                        return transform(orig, *args, **kwargs)

                    return wrapper

                setattr(module, attr, make_wrapper(original))
            yield
        finally:
            for module, attr, original in originals:
                setattr(module, attr, original)
