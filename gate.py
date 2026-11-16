from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GateState(str, Enum):
    LISTENING = "listening"
    RETRIEVING = "retrieving"
    INJECTING = "injecting"
    ANSWER_READY = "answer_ready"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class RetrievalTurn:
    token: int
    query: str


class KnowledgeGate:
    """Hard gate that keeps duplex generation listening during retrieval."""

    def __init__(self) -> None:
        self.state = GateState.LISTENING
        self._next_token = 1
        self._active_turn: RetrievalTurn | None = None
        self.evidence = ""

    @property
    def force_listen(self) -> bool:
        return self.state in {GateState.RETRIEVING, GateState.INJECTING}

    @property
    def force_speak(self) -> bool:
        return self.state is GateState.ANSWER_READY

    def begin_retrieval(self, query: str) -> RetrievalTurn:
        turn = RetrievalTurn(token=self._next_token, query=query)
        self._next_token += 1
        self._active_turn = turn
        self.evidence = ""
        self.state = GateState.RETRIEVING
        return turn

    def mark_injected(self, turn: RetrievalTurn, evidence: str) -> bool:
        if turn != self._active_turn or self.state is not GateState.RETRIEVING:
            return False
        self.evidence = evidence
        self.state = GateState.ANSWER_READY
        return True

    def consume_answer_ready(self) -> bool:
        if self.state is not GateState.ANSWER_READY:
            return False
        self.state = GateState.SPEAKING
        return True

    def finish_speaking(self) -> None:
        if self.state is GateState.SPEAKING:
            self.state = GateState.LISTENING
            self._active_turn = None
            self.evidence = ""

    def interrupt(self) -> None:
        self.state = GateState.LISTENING
        self._active_turn = None
        self.evidence = ""
