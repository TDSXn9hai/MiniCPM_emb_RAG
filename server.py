from __future__ import annotations

import asyncio
import base64
import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from asr import ASRBackend, create_asr_backend
from engine import KnowledgePipeline
from rag_core import KnowledgeBase
from runtime import (
    MiniCPMDuplexRuntime,
    MockDuplexRuntime,
    RuntimeOutput,
)


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Settings:
    runtime_backend: str = os.environ.get("RAG_RUNTIME", "mock")
    asr_backend: str = os.environ.get("RAG_ASR", "whisper")
    asr_model: str = os.environ.get("RAG_ASR_MODEL", "openai/whisper-small")
    asr_language: str = os.environ.get("RAG_ASR_LANGUAGE", "zh")
    mock_transcript: str = os.environ.get(
        "RAG_MOCK_TRANSCRIPT", "上海住宿能报多少"
    )
    model_path: str = os.environ.get(
        "RAG_MODEL_PATH", "../minicpm_demo/model"
    )
    ref_audio_path: str = os.environ.get(
        "RAG_REF_AUDIO", "assets/ref_audio/ref_minicpm_signature.wav"
    )
    knowledge_path: str = os.environ.get(
        "RAG_KNOWLEDGE_PATH", "knowledge/sample.json"
    )
    default_system_prompt: str = os.environ.get(
        "RAG_SYSTEM_PROMPT",
        (
            "Streaming Audio Conversation. 你是一名严谨的知识助手。"
            "回答以“先结论、后解释”为主，口语自然简洁。"
        ),
    )
    vad_silence_ms: int = int(os.environ.get("RAG_VAD_SILENCE_MS", "700"))
    vad_threshold: float = float(os.environ.get("RAG_VAD_THRESHOLD", "0.75"))
    max_evidence_chars: int = int(os.environ.get("RAG_MAX_EVIDENCE_CHARS", "900"))

    def resolve(self, path: str) -> str:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            return str(candidate)
        return str((ROOT / candidate).resolve())


def load_knowledge_base(path: str) -> KnowledgeBase:
    from rag_core import KnowledgeDocument

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    documents = [
        KnowledgeDocument(
            id=str(item["id"]),
            title=str(item["title"]),
            content=str(item["content"]),
            source=str(item.get("source") or ""),
        )
        for item in payload["documents"]
    ]
    return KnowledgeBase.from_documents(documents)


def create_runtime(settings: Settings):
    backend = settings.runtime_backend.strip().lower()
    if backend == "mock":
        return MockDuplexRuntime()
    if backend == "minicpm":
        return MiniCPMDuplexRuntime(
            model_path=settings.resolve(settings.model_path),
            ref_audio_path=settings.resolve(settings.ref_audio_path),
            system_prompt=settings.default_system_prompt,
        )
    raise ValueError(f"unsupported runtime backend: {settings.runtime_backend}")


def create_asr(settings: Settings) -> ASRBackend:
    return create_asr_backend(
        settings.asr_backend,
        model_name=settings.asr_model,
        language=settings.asr_language,
        transcript=settings.mock_transcript,
    )


class A2AConnection:
    def __init__(
        self,
        *,
        websocket: WebSocket,
        settings: Settings,
        runtime,
        asr: ASRBackend,
        knowledge_base: KnowledgeBase,
    ) -> None:
        self.websocket = websocket
        self.settings = settings
        self.runtime = runtime
        self.pipeline = KnowledgePipeline(
            asr=asr,
            knowledge_base=knowledge_base,
            runtime=runtime,
            top_k=3,
            min_score=0.08,
        )
        self.send_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self.query_task: asyncio.Task | None = None
        self.model_buffer = np.array([], dtype=np.float32)
        self.sequence = 0
        self._vad = None
        self._was_speaking = False

    async def send(self, event_type: str, **payload: Any) -> None:
        async with self.send_lock:
            await self.websocket.send_json({"type": event_type, **payload})

    async def run(self) -> None:
        await self.websocket.accept()
        await self.send(
            "server.ready",
            runtime=self.settings.runtime_backend,
            asr=self.settings.asr_backend,
            input_format="pcm_f32le_16000",
            output_format="pcm_f32le_24000",
            experimental_duplex_rag=True,
        )
        first = await self.websocket.receive_json()
        if first.get("type") != "session.start":
            await self.send(
                "error",
                code="invalid_start",
                message="first message must be session.start",
            )
            await self.websocket.close(code=1008)
            return

        system_prompt = str(
            first.get("system_prompt") or self.settings.default_system_prompt
        )
        await self.runtime.prepare(system_prompt)
        await self.send(
            "session.ready",
            runtime=self.settings.runtime_backend,
            sample_rate=16000,
            answer_sample_rate=24000,
            state="listening",
        )

        try:
            while True:
                message = await self.websocket.receive_json()
                kind = message.get("type")
                if kind == "text.query":
                    text = str(message.get("text") or "").strip()
                    if not text:
                        await self.send(
                            "error",
                            code="empty_query",
                            message="text.query requires text",
                        )
                        continue
                    await self.start_query(
                        lambda text=text: self.pipeline.process_query(text)
                    )
                    continue
                if kind == "audio.append":
                    await self.handle_audio(message)
                    continue
                if kind == "interrupt":
                    await self.interrupt()
                    continue
                if kind == "session.close":
                    break
                await self.send(
                    "error",
                    code="unsupported_message",
                    message=f"unsupported message type: {kind}",
                )
        except WebSocketDisconnect:
            pass
        finally:
            if self.query_task and not self.query_task.done():
                self.query_task.cancel()
            await self.runtime.close()

    async def start_query(self, factory) -> None:
        if self.query_task and not self.query_task.done():
            self.pipeline.interrupt()
            self.query_task.cancel()
        await self.send("state", state="retrieving", force_listen=True)
        self.query_task = asyncio.create_task(self._run_query(factory))

    async def _run_query(self, factory) -> None:
        try:
            result = await factory()
            if result.stale:
                await self.send("response.interrupted", reason="stale_query")
                return
            await self.send("asr.final", text=result.query)
            await self.send(
                "retrieval.result",
                query=result.query,
                citations=[
                    {
                        "id": item.id,
                        "title": item.title,
                        "source": item.source,
                        "score": round(item.score, 4),
                    }
                    for item in result.citations
                ],
            )
            await self.send(
                "state",
                state="injecting",
                force_listen=True,
            )
            await self.send(
                "knowledge.injected",
                evidence_chars=len(result.evidence.text),
            )
            await self.send(
                "state",
                state="answer_ready",
                force_listen=False,
                force_speak=True,
            )
            await self.send("answer.started")
            async with self.model_lock:
                answer = await self.runtime.generate_answer()
            await self.send_answer(answer)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send(
                "error",
                code="query_failed",
                message=str(exc),
            )

    async def send_answer(self, answer: RuntimeOutput) -> None:
        if answer.text:
            await self.send("answer.text", text=answer.text)
        if answer.audio_b64:
            await self.send(
                "answer.audio",
                audio=answer.audio_b64,
                sample_rate=24000,
            )
        await self.send(
            "answer.done",
            end_of_turn=answer.end_of_turn,
            metrics=answer.metrics,
        )
        self.pipeline.finish_speaking()
        await self.send("state", state="listening", force_listen=False)

    async def handle_audio(self, message: dict[str, Any]) -> None:
        try:
            raw = base64.b64decode(str(message["audio"]))
            samples = np.frombuffer(raw, dtype="<f4").copy()
        except Exception:
            await self.send(
                "error",
                code="invalid_audio",
                message="audio must be base64 float32 PCM",
            )
            return
        sequence = int(message.get("sequence", self.sequence))
        self.sequence = sequence + 1
        await self.send(
            "audio.ack",
            sequence=sequence,
            samples=int(samples.size),
        )

        if self._vad is None:
            from vad.vad import StreamingVAD, StreamingVadOptions

            self._vad = StreamingVAD(
                StreamingVadOptions(
                    threshold=self.settings.vad_threshold,
                    min_silence_duration_ms=self.settings.vad_silence_ms,
                )
            )
        speaking_before = self._vad.is_speaking
        segment = self._vad.feed(samples)
        speaking_now = self._vad.is_speaking
        if speaking_now != speaking_before:
            self._was_speaking = speaking_now
            await self.send("vad.state", speaking=speaking_now)
            await self.send(
                "state",
                state="listening" if speaking_now else "retrieving",
                force_listen=True,
            )

        self.model_buffer = np.concatenate([self.model_buffer, samples])
        await self.drain_model_audio()

        if segment is not None and segment.size:
            await self.start_query(
                lambda segment=segment: self.pipeline.process_utterance(segment)
            )

    async def drain_model_audio(self) -> None:
        if self.query_task and not self.query_task.done():
            return
        while self.model_buffer.size >= 16000:
            chunk = self.model_buffer[:16000]
            self.model_buffer = self.model_buffer[16000:]
            async with self.model_lock:
                output = await self.runtime.process_audio(
                    chunk,
                    force_listen=True,
                )
            if output.text:
                await self.send("listen.transcript", text=output.text)

    async def interrupt(self) -> None:
        self.pipeline.interrupt()
        await self.runtime.interrupt()
        if self.query_task and not self.query_task.done():
            self.query_task.cancel()
        self.query_task = None
        await self.send("response.interrupted", reason="client_interrupt")
        await self.send("state", state="listening", force_listen=False)


def create_app(
    *,
    settings: Settings | None = None,
    runtime=None,
    asr: ASRBackend | None = None,
    knowledge_base: KnowledgeBase | None = None,
) -> FastAPI:
    resolved = settings or Settings()
    actual_runtime = runtime
    actual_asr = asr
    actual_knowledge = knowledge_base

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal actual_runtime, actual_asr, actual_knowledge
        if actual_runtime is None:
            actual_runtime = create_runtime(resolved)
        if actual_asr is None:
            actual_asr = create_asr(resolved)
        if actual_knowledge is None:
            actual_knowledge = load_knowledge_base(
                resolved.resolve(resolved.knowledge_path)
            )
        await actual_runtime.prepare(resolved.default_system_prompt)
        app.state.settings = resolved
        app.state.runtime = actual_runtime
        app.state.asr = actual_asr
        app.state.knowledge_base = actual_knowledge
        yield
        await actual_runtime.close()

    app = FastAPI(title="MiniCPM Duplex RAG Test", lifespan=lifespan)
    static_dir = ROOT / "static"
    app.mount(
        "/static",
        StaticFiles(directory=static_dir, check_dir=False),
        name="static",
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "runtime": resolved.runtime_backend,
            "asr": resolved.asr_backend,
        }

    @app.get("/")
    async def index():
        return FileResponse(static_dir / "index.html")

    @app.websocket("/ws/a2a")
    async def a2a_ws(websocket: WebSocket):
        connection = A2AConnection(
            websocket=websocket,
            settings=app.state.settings,
            runtime=app.state.runtime,
            asr=app.state.asr,
            knowledge_base=app.state.knowledge_base,
        )
        await connection.run()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "server:app",
        host=os.environ.get("RAG_HOST", "0.0.0.0"),
        port=int(os.environ.get("RAG_PORT", "8010")),
        reload=False,
    )


if __name__ == "__main__":
    main()
