from gate import GateState, KnowledgeGate


def test_gate_blocks_answer_until_injection_finishes():
    gate = KnowledgeGate()
    turn = gate.begin_retrieval("q-1")

    assert gate.state is GateState.RETRIEVING
    assert gate.force_listen is True
    assert gate.force_speak is False

    assert gate.mark_injected(turn, "evidence") is True

    assert gate.state is GateState.ANSWER_READY
    assert gate.force_listen is False
    assert gate.force_speak is True
    assert gate.consume_answer_ready() is True
    assert gate.state is GateState.SPEAKING
    assert gate.force_speak is False


def test_interruption_invalidates_old_retrieval_result():
    gate = KnowledgeGate()
    old_turn = gate.begin_retrieval("q-1")

    gate.interrupt()

    assert gate.state is GateState.LISTENING
    assert gate.mark_injected(old_turn, "stale evidence") is False
    assert gate.state is GateState.LISTENING
