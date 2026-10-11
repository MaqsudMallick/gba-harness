"""Thread-safe human decisions for the turn-based linked-agent runner."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field

from gbharness.agent.contracts import Decision, DecisionRequest


@dataclass(slots=True)
class HumanActionSource:
    """Queue short controller macros from a responsive UI thread."""

    max_pending: int = 4
    _pending: deque[str] = field(default_factory=deque, init=False, repr=False)
    _condition: threading.Condition = field(
        default_factory=threading.Condition, init=False, repr=False
    )

    def submit(self, action: str) -> None:
        with self._condition:
            while len(self._pending) >= self.max_pending:
                self._pending.popleft()
            self._pending.append(action)
            self._condition.notify()

    def next_action(self, *, timeout: float) -> str:
        with self._condition:
            self._condition.wait_for(lambda: bool(self._pending), timeout=timeout)
            return self._pending.popleft() if self._pending else "wait"


@dataclass(slots=True)
class HumanDecisionProvider:
    """Adapt queued keyboard commands to the provider-neutral decision contract."""

    source: HumanActionSource
    timeout_seconds: float = 30.0

    @property
    def name(self) -> str:
        return "human:keyboard"

    def decide(self, request: DecisionRequest) -> Decision:
        allowed = {option.key for option in request.actions}
        action = self.source.next_action(timeout=self.timeout_seconds)
        if action not in allowed:
            action = "wait"
        return Decision(
            action=action,
            confidence=1.0,
            scene="human_input",
            note="Keyboard macro selected by the human participant.",
        )
