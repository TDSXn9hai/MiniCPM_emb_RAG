from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from asr import ASRBackend
from gate import KnowledgeGate
from rag_core import (
    Citation,
    EvidenceCard,
    KnowledgeBase,
    build_evidence_card,
)


class KnowledgeInjectingRuntime(Protocol):
    async def inject_knowledge(self, query: str, evidence: str) -> dict: ...


@dataclass(frozen=True)
class PipelineResult:
    query: str
    evidence: EvidenceCard
    citations: tuple[Citation, ...]
    stale: bool


class KnowledgePipeline:
    def __init__(
        self,
        *,
        asr: ASRBackend,
        knowledge_base: KnowledgeBase,
        runtime: KnowledgeInjectingRuntime,
        top_k: int = 3,
        min_score: float = 0.08,
    ) -> None:
        self.asr = asr
        self.knowledge_base = knowledge_base
        self.runtime = runtime
        self.top_k = top_k
        self.min_score = min_score
        self.gate = KnowledgeGate()
        self.last_query = ""

    async def process_utterance(self, audio: np.ndarray) -> PipelineResult:
        self.last_query = ""
        turn = self.gate.begin_retrieval("")
        transcript = await self.asr.transcribe(audio, 16000)
        return await self._retrieve_and_inject(turn, transcript)

    async def process_query(self, query: str) -> PipelineResult:
        self.last_query = ""
        turn = self.gate.begin_retrieval(query)
        return await self._retrieve_and_inject(turn, query)

    async def _retrieve_and_inject(self, turn, transcript: str) -> PipelineResult:
        self.last_query = transcript

        hits = self.knowledge_base.search(
            transcript,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        if hits:
            card = build_evidence_card(hits)
        else:
            card = EvidenceCard(
                text=(
                    "知识库没有检索到足够相关的内容。"
                    "请明确告诉用户当前知识库没有足够依据。"
                ),
                citations=(),
            )

        if not self.gate.mark_injected(turn, card.text):
            return PipelineResult(
                query=transcript,
                evidence=card,
                citations=card.citations,
                stale=True,
            )

        await self.runtime.inject_knowledge(transcript, card.text)
        return PipelineResult(
            query=transcript,
            evidence=card,
            citations=card.citations,
            stale=False,
        )

    def interrupt(self) -> None:
        self.gate.interrupt()

    def finish_speaking(self) -> None:
        self.gate.finish_speaking()
