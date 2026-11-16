import asyncio

import numpy as np
import pytest

from engine import KnowledgePipeline
from gate import GateState
from rag_core import KnowledgeBase, KnowledgeDocument


class FakeASR:
    def __init__(self, text, wait: asyncio.Event | None = None):
        self.text = text
        self.wait = wait

    async def transcribe(self, audio, sample_rate=16000):
        if self.wait is not None:
            await self.wait.wait()
        return self.text


class FakeRuntime:
    def __init__(self):
        self.injected = []

    async def inject_knowledge(self, query, evidence):
        self.injected.append((query, evidence))
        return {"token_count": len(evidence)}


@pytest.mark.asyncio
async def test_pipeline_retrieves_and_injects_before_answer_ready():
    kb = KnowledgeBase.from_documents(
        [
            KnowledgeDocument(
                id="travel",
                title="差旅报销制度",
                content="上海住宿标准为每晚六百元。",
                source="travel.pdf#page=8",
            )
        ]
    )
    runtime = FakeRuntime()
    pipeline = KnowledgePipeline(
        asr=FakeASR("上海住宿能报多少"),
        knowledge_base=kb,
        runtime=runtime,
    )

    result = await pipeline.process_utterance(
        np.zeros(16000, dtype=np.float32)
    )

    assert result.stale is False
    assert result.query == "上海住宿能报多少"
    assert "六百元" in runtime.injected[0][1]
    assert result.citations[0].id == "K1"
    assert pipeline.gate.state is GateState.ANSWER_READY


@pytest.mark.asyncio
async def test_text_query_skips_asr_but_runs_the_same_gate():
    kb = KnowledgeBase.from_documents(
        [
            KnowledgeDocument(
                id="travel",
                title="差旅报销制度",
                content="上海住宿标准为每晚六百元。",
            )
        ]
    )
    runtime = FakeRuntime()
    pipeline = KnowledgePipeline(
        asr=FakeASR("不会被调用"),
        knowledge_base=kb,
        runtime=runtime,
    )

    result = await pipeline.process_query("上海住宿标准")

    assert result.query == "上海住宿标准"
    assert result.stale is False
    assert runtime.injected
    assert pipeline.gate.state is GateState.ANSWER_READY


@pytest.mark.asyncio
async def test_interrupted_pipeline_does_not_inject_stale_evidence():
    released = asyncio.Event()
    runtime = FakeRuntime()
    pipeline = KnowledgePipeline(
        asr=FakeASR("很慢的问题", wait=released),
        knowledge_base=KnowledgeBase.from_documents(
            [
                KnowledgeDocument(
                    id="slow",
                    title="很慢的问题",
                    content="很慢的问题答案。",
                )
            ]
        ),
        runtime=runtime,
    )

    task = asyncio.create_task(
        pipeline.process_utterance(np.zeros(16000, dtype=np.float32))
    )
    await asyncio.sleep(0)
    pipeline.interrupt()
    released.set()
    result = await task

    assert result.stale is True
    assert runtime.injected == []
    assert pipeline.gate.state is GateState.LISTENING
