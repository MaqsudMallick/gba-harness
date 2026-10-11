from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

from gbharness.observability import NeatlogsSession, io_attributes, serialize_io


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


def test_span_io_is_named_after_the_span_kind(monkeypatch) -> None:
    """The reader resolves input/output from `neatlogs.<kind>.input|output`.

    A payload filed under any other key is dropped silently and the dashboard row
    renders blank, so the kind-derived name is the contract worth pinning.
    """
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

    with session.span("captain", "AGENT", input_value={"goal": "reach the warp"}) as active:
        session.record_output(active, "AGENT", {"action": "move_up"})

    assert attributes["neatlogs.agent.input"] == '{"goal": "reach the warp"}'
    assert attributes["neatlogs.agent.output"] == '{"action": "move_up"}'


def test_io_attributes_passes_strings_through_and_skips_empty_sides() -> None:
    assert io_attributes("WORKFLOW", input_value="four agents, one clock") == {
        "neatlogs.workflow.input": "four agents, one clock"
    }
    assert io_attributes("TOOL", output_value=3) == {"neatlogs.tool.output": "3"}
    assert io_attributes("CHAIN") == {}


def test_serialize_io_survives_a_payload_json_cannot_encode() -> None:
    """Telemetry must never be the thing that kills a long agent run."""
    # An unserializable leaf is stringified rather than raising.
    assert "<object object" in serialize_io(object())

    # A circular structure defeats `default=` too, and still degrades to a string.
    circular: list[object] = []
    circular.append(circular)
    assert serialize_io(circular).startswith("[[")
