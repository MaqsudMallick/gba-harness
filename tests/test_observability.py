from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from gbharness.observability import NeatlogsSession


def test_session_exposes_the_latest_trace_id(monkeypatch) -> None:
    first = SimpleNamespace(
        get_span_context=lambda: SimpleNamespace(is_valid=True, trace_id=0x123)
    )
    second = SimpleNamespace(
        get_span_context=lambda: SimpleNamespace(is_valid=True, trace_id=0x456)
    )
    spans = iter((first, first, second))
    monkeypatch.setattr(
        "gbharness.observability.neatlogs.trace",
        lambda *args, **kwargs: nullcontext(next(spans)),
    )

    session = NeatlogsSession("test-key")
    assert session.trace_id is None

    with session.span("root", "WORKFLOW"):
        pass
    with session.span("child", "TOOL"):
        pass
    assert session.trace_id == f"{0x123:032x}"

    with session.span("next-run", "WORKFLOW"):
        pass
    assert session.trace_id == f"{0x456:032x}"
