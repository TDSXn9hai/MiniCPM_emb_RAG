"""Review probes: passing asserts reproduce existing defects, not fixed behavior.

Run from minicpm_rag_test:
  ../.venv/bin/python -m pytest review_20261116/test_review_reproductions.py -q
No model weights or GPU are used.
"""
import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from asr import MockASR
from engine import KnowledgePipeline
from gate import GateState
from rag_core import KnowledgeBase, KnowledgeDocument
from runtime import MiniCPMDuplexRuntime, MockDuplexRuntime
from server import A2AConnection, Settings


def pipeline_for(runtime):
    return KnowledgePipeline(
        asr=MockASR("上海住宿"),
        knowledge_base=KnowledgeBase.from_documents(
            [KnowledgeDocument("travel", "上海住宿", "上海住宿六百元。")]
        ),
        runtime=runtime,
    )


def make_runtime(view):
    runtime = MiniCPMDuplexRuntime(
        model_path="unused", ref_audio_path="unused", system_prompt="first"
    )
    runtime._view = view
    return runtime


@pytest.mark.asyncio
async def test_answer_completion_leaves_gate_answer_ready():
    pipeline = pipeline_for(MockDuplexRuntime())
    await pipeline.process_query("上海住宿")
    pipeline.finish_speaking()
    # consume_answer_ready() is never called by server.
    assert pipeline.gate.state is GateState.ANSWER_READY


@pytest.mark.asyncio
async def test_interrupt_during_injection_returns_nonstale_result():
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowInjection:
        async def inject_knowledge(self, query, evidence):
            entered.set()
            await release.wait()

    pipeline = pipeline_for(SlowInjection())
    task = asyncio.create_task(pipeline.process_query("上海住宿"))
    await asyncio.wait_for(entered.wait(), 2)
    # Gate announces ready before injection completes.
    assert pipeline.gate.state is GateState.ANSWER_READY
    pipeline.interrupt()
    release.set()
    result = await task
    assert result.stale is False


@pytest.mark.asyncio
async def test_cancel_releases_async_lock_while_model_thread_still_runs():
    entered, release = threading.Event(), threading.Event()
    events = []

    class BlockingView:
        def generate(self, **kwargs):
            events.append("generation_started")
            entered.set()
            if not release.wait(5):
                raise TimeoutError("review worker was not released")
            events.append("generation_finished")
            return SimpleNamespace(
                text="完成。", audio_data=None, is_listen=False,
                end_of_turn=True, cost_llm_ms=0, cost_tts_ms=0,
                cost_token2wav_ms=0, cost_all_ms=0, n_tokens=1,
                n_tts_tokens=0,
            )

        def finalize(self):
            events.append("finalized")

        def inject_knowledge(self, **kwargs):
            events.append("injected_during_generation")
            return {}

    runtime = make_runtime(BlockingView())
    lock = asyncio.Lock()

    async def generate():
        async with lock:
            return await runtime.generate_answer()

    task = asyncio.create_task(generate())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()
        await runtime.inject_knowledge("新问题", "新资料")
        assert "injected_during_generation" in events
        assert "generation_finished" not in events
    finally:
        release.set()
        for _ in range(100):
            if "finalized" in events:
                break
            await asyncio.sleep(0.01)
        assert "finalized" in events


@pytest.mark.asyncio
async def test_prepare_existing_view_does_not_reset_session():
    class View:
        prepare_calls = 0

        def prepare(self, **kwargs):
            self.prepare_calls += 1

    view = View()
    runtime = make_runtime(view)
    await runtime.prepare("second session prompt")
    assert runtime.system_prompt == "second session prompt"
    assert view.prepare_calls == 0


@pytest.mark.asyncio
async def test_audio_backlog_is_not_drained_while_query_active():
    class Socket:
        async def send_json(self, payload):
            pass

    class Runtime(MockDuplexRuntime):
        calls = 0

        async def process_audio(self, audio, **kwargs):
            self.calls += 1
            return await super().process_audio(audio, **kwargs)

    runtime = Runtime()
    connection = A2AConnection(
        websocket=Socket(), settings=Settings(), runtime=runtime,
        asr=MockASR("上海住宿"),
        knowledge_base=pipeline_for(runtime).knowledge_base,
    )
    connection.model_buffer = np.zeros(48000, dtype=np.float32)
    blocker = asyncio.Event()
    connection.query_task = asyncio.create_task(blocker.wait())
    await connection.drain_model_audio()
    assert runtime.calls == 0
    assert connection.model_buffer.size == 48000
    blocker.set()
    await connection.query_task
