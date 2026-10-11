"""Neatlogs lifecycle kept outside model and emulator code.

Only agent entry points create this object. Input tests and human-controlled emulation do
not need a tracing account, and importing the core package stays side-effect free.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Protocol

import neatlogs  # type: ignore[import-untyped]


class RunTelemetry(Protocol):
    """The tiny tracing surface used by the runner."""

    def span(
        self,
        name: str,
        kind: str,
        *,
        capture_input: bool = False,
        capture_output: bool = False,
        attributes: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[Any]: ...

    def annotate(self, span: Any, attributes: Mapping[str, object]) -> None: ...

    def log(self, template: str, **values: object) -> None: ...


class NullTelemetry:
    """Test/offline telemetry with exactly the same runner contract."""

    def span(
        self,
        name: str,
        kind: str,
        *,
        capture_input: bool = False,
        capture_output: bool = False,
        attributes: Mapping[str, object] | None = None,
    ) -> AbstractContextManager[None]:
        del name, kind, capture_input, capture_output, attributes
        return nullcontext()

    def annotate(self, span: Any, attributes: Mapping[str, object]) -> None:
        del span, attributes

    def log(self, template: str, **values: object) -> None:
        del template, values


@dataclass(slots=True)
class NeatlogsSession:
    """Initialize once, wrap provider clients, then flush a one-shot agent process."""

    api_key: str
    workflow_name: str = "visual-game-agent"
    endpoint: str = "https://ingest.neatlogs.com"
    _closed: bool = False
    _trace_ids: list[str] = field(default_factory=list, init=False, repr=False)

    @property
    def trace_id(self) -> str | None:
        """Return the most recently opened trace for shareable run evidence."""
        return self._trace_ids[-1] if self._trace_ids else None

    @property
    def trace_ids(self) -> tuple[str, ...]:
        """Return every distinct trace observed during this one-shot session."""
        return tuple(self._trace_ids)

    def start(self) -> NeatlogsSession:
        neatlogs.init(
            api_key=self.api_key,
            endpoint=self.endpoint,
            workflow_name=self.workflow_name,
            capture_logs=True,
        )
        return self

    def wrap(self, client: Any, **attributes: object) -> Any:
        """Wrap one supported model client without leaking tracing into its adapter."""
        return neatlogs.wrap(client, **attributes)

    @contextmanager
    def span(
        self,
        name: str,
        kind: str,
        *,
        capture_input: bool = False,
        capture_output: bool = False,
        attributes: Mapping[str, object] | None = None,
    ) -> Iterator[Any]:
        del capture_input, capture_output
        with neatlogs.trace(name, kind=kind, **dict(attributes or {})) as span:
            context = span.get_span_context()
            if context.is_valid:
                trace_id = f"{context.trace_id:032x}"
                if trace_id not in self._trace_ids:
                    self._trace_ids.append(trace_id)
            yield span

    def annotate(self, span: Any, attributes: Mapping[str, object]) -> None:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    def log(self, template: str, **values: object) -> None:
        neatlogs.log(template, **values)

    def close(self) -> None:
        if self._closed:
            return
        try:
            neatlogs.flush()
        finally:
            neatlogs.shutdown()
            self._closed = True

    def __enter__(self) -> NeatlogsSession:
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.close()
