"""Milestone 4: invariants that must hold for *any* valid action tree.

The value of this file is the choice of invariants, not the code. Each one is a property
the rest of the system is allowed to assume, and each would be easy to break with a
plausible-looking refactor of the scheduler.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from gbharness.actions import Action, Hold, Press, Release, Repeat, Sequence, Tap, Wait
from gbharness.backends.null_port import NullPort
from gbharness.buttons import Button
from gbharness.scheduler import Scheduler

_buttons = st.sets(st.sampled_from(list(Button)), min_size=1, max_size=4).map(frozenset)
_frames = st.integers(min_value=0, max_value=8)

_leaf: st.SearchStrategy[Action] = st.one_of(
    st.builds(Tap, _buttons, _frames, _frames),
    st.builds(Hold, _buttons, _frames),
    st.builds(Press, _buttons),
    st.builds(Release, st.one_of(st.just(frozenset()), _buttons)),
    st.builds(Wait, _frames),
)

actions: st.SearchStrategy[Action] = st.recursive(
    _leaf,
    lambda children: st.one_of(
        st.builds(Sequence, st.lists(children, max_size=4).map(tuple)),
        st.builds(Repeat, children, st.integers(min_value=1, max_value=3)),
    ),
    max_leaves=8,
)


def _contains_press(action: Action) -> bool:
    """Does this tree leave anything open-ended?"""
    match action:
        case Press():
            return True
        case Sequence():
            return any(_contains_press(i) for i in action.items)
        case Repeat():
            return _contains_press(action.item)
        case _:
            return False


@settings(max_examples=250, deadline=None)
@given(actions)
def test_scheduler_clock_never_diverges_from_the_port(action: Action) -> None:
    """The transcript's clock must match reality, or replay is fiction."""
    port = NullPort()
    sched = Scheduler(port)
    sched.run(action)
    assert sched.frame == port.frame


@settings(max_examples=250, deadline=None)
@given(actions)
def test_frames_advanced_equals_the_sum_of_the_parts(action: Action) -> None:
    port = NullPort()
    sched = Scheduler(port)
    result = sched.run(action)
    recorded = sum(e.detail for e in port.events if e.op == "advance")
    assert result.frames_advanced == recorded
    assert result.end_frame - result.start_frame == result.frames_advanced


@settings(max_examples=250, deadline=None)
@given(actions)
def test_edges_for_each_button_strictly_alternate(action: Action) -> None:
    """press, release, press, release ... never two of the same in a row.

    A double press without an intervening release means the held-state bookkeeping has
    drifted, which is how a button ends up stuck.
    """
    port = NullPort()
    Scheduler(port).run(action)
    for button in Button:
        ops = [op for _, op, b in port.edges if b == button]
        assert all(a != b for a, b in zip(ops, ops[1:], strict=False)), (
            f"{button} edges do not alternate: {ops}"
        )
        if ops:
            assert ops[0] == "press", f"{button} was released before it was pressed"


@settings(max_examples=250, deadline=None)
@given(actions)
def test_no_button_is_left_held_unless_the_tree_asked_for_it(action: Action) -> None:
    """The stuck-button guarantee.

    Only an explicit `Press` may outlive a run. Tap and Hold always clean up after
    themselves, including when emulation ends or an exception unwinds.
    """
    port = NullPort()
    sched = Scheduler(port)
    sched.run(action)
    if not _contains_press(action):
        assert port.pressed() == frozenset()


@settings(max_examples=250, deadline=None)
@given(actions)
def test_held_set_always_matches_the_ports_own_accounting(action: Action) -> None:
    port = NullPort()
    sched = Scheduler(port)
    result = sched.run(action)
    assert result.buttons_held == port.pressed()


@settings(max_examples=150, deadline=None)
@given(actions)
def test_execution_is_deterministic(action: Action) -> None:
    """Same action, same edges, same frames -- the premise replay rests on."""
    a, b = NullPort(), NullPort()
    Scheduler(a).run(action)
    Scheduler(b).run(action)
    assert a.events == b.events
    assert a.frame == b.frame


@settings(max_examples=150, deadline=None)
@given(actions, st.integers(min_value=1, max_value=20))
def test_ending_emulation_early_never_strands_a_button(action: Action, ends: int) -> None:
    port = NullPort(ends_after=ends)
    sched = Scheduler(port)
    sched.run(action)
    if not _contains_press(action):
        assert port.pressed() == frozenset()
