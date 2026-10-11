from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from gbharness.observability import NeatlogsSession


def test_session_exposes_the_latest_trace_id(monkeypatch) -> None:
    first = SimpleNamespace(get_span_context=lambda: SimpleNamespace(is_valid=True, trace_id=0x123))
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
    assert session.trace_ids == (f"{0x123:032x}", f"{0x456:032x}")


def test_session_forwards_and_updates_custom_span_attributes(monkeypatch) -> None:
    attributes: dict[str, object] = {}
    span = SimpleNamespace(
        get_span_context=lambda: SimpleNamespace(is_valid=True, trace_id=0xABC),
        set_attribute=lambda key, value: attributes.__setitem__(key, value),
    )

    def trace(_name: str, **values: object):
        attributes.update(values)
        return nullcontext(span)

    monkeypatch.setattr("gbharness.observability.neatlogs.trace", trace)
    session = NeatlogsSession("test-key")

    with session.span("root", "WORKFLOW", attributes={"game": "zelda"}) as active:
        session.annotate(active, {"run.status": "completed"})

    assert attributes == {
        "kind": "WORKFLOW",
        "game": "zelda",
        "run.status": "completed",
    }
