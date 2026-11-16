from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from duplex_generation import (
    KNOWLEDGE_ANSWER_MAX_UNITS,
    should_continue_knowledge_answer,
)


@dataclass(frozen=True)
class RuntimeOutput:
    text: str
    audio_b64: str | None
    is_listen: bool
    end_of_turn: bool
    metrics: dict[str, Any] = field(default_factory=dict)


def knowledge_duplex_config():
    from core.schemas.duplex import DuplexConfig

    return DuplexConfig(
        force_listen_count=0,
        temperature=0.3,
        top_k=20,
        top_p=0.75,
        length_penalty=1.2,
        max_new_speak_tokens_per_chunk=96,
    )


class MockDuplexRuntime:
    def __init__(self) -> None:
        self.system_prompt = ""
        self._pending_query = ""
        self._pending_evidence = ""

    async def prepare(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    async def process_audio(
        self,
        audio: np.ndarray,
        *,
        force_listen: bool,
    ) -> RuntimeOutput:
        return RuntimeOutput(
            text="",
            audio_b64=None,
            is_listen=True,
            end_of_turn=False,
        )

    async def inject_knowledge(self, query: str, evidence: str) -> dict[str, Any]:
        self._pending_query = query
        self._pending_evidence = evidence
        return {"token_count": len(evidence)}

    async def generate_answer(self) -> RuntimeOutput:
        evidence = self._pending_evidence
        snippet = evidence[:120].replace("\n", " ")
        text = (
            f"这是 mock 双工回答。针对“{self._pending_query}”，"
            f"检索到的依据是：{snippet}"
        )
        sample_rate = 24000
        duration_s = 0.6
        samples = np.arange(
            int(sample_rate * duration_s), dtype=np.float32
        )
        waveform = 0.02 * np.sin(
            2 * np.pi * 220 * samples / sample_rate
        )
        self._pending_query = ""
        self._pending_evidence = ""
        return RuntimeOutput(
            text=text,
            audio_b64=base64.b64encode(waveform.tobytes()).decode("ascii"),
            is_listen=False,
            end_of_turn=True,
        )

    async def interrupt(self) -> None:
        self._pending_query = ""
        self._pending_evidence = ""

    async def close(self) -> None:
        return None


class MiniCPMDuplexRuntime:
    """Experimental single-process MiniCPM-o duplex runtime."""

    def __init__(
        self,
        *,
        model_path: str,
        ref_audio_path: str,
        system_prompt: str,
        compile_model: bool = False,
        attn_implementation: str = "sdpa",
    ) -> None:
        self.model_path = str(Path(model_path).expanduser().resolve())
        self.ref_audio_path = str(
            Path(ref_audio_path).expanduser().resolve()
        )
        self.system_prompt = system_prompt
        self.compile_model = compile_model
        self.attn_implementation = attn_implementation
        self._processor = None
        self._view = None
        self._pending_query = ""
        self._pending_evidence = ""

    async def prepare(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        await asyncio.to_thread(self._prepare_sync)

    def _prepare_sync(self) -> None:
        if self._view is not None:
            return
        from core.processors.unified import UnifiedProcessor

        processor = UnifiedProcessor(
            model_path=self.model_path,
            compile=self.compile_model,
            attn_implementation=self.attn_implementation,
            chat_vocoder="token2wav",
            duplex_config=knowledge_duplex_config(),
        )
        view = processor.set_duplex_mode()
        view.prepare(
            system_prompt_text=self.system_prompt,
            ref_audio_path=self.ref_audio_path,
            prompt_wav_path=self.ref_audio_path,
        )
        self._processor = processor
        self._view = view

    @staticmethod
    def _to_output(result) -> RuntimeOutput:
        return RuntimeOutput(
            text=result.text or "",
            audio_b64=result.audio_data,
            is_listen=bool(result.is_listen),
            end_of_turn=bool(result.end_of_turn),
            metrics={
                "cost_llm_ms": result.cost_llm_ms,
                "cost_tts_ms": result.cost_tts_ms,
                "cost_token2wav_ms": result.cost_token2wav_ms,
                "cost_all_ms": result.cost_all_ms,
                "n_tokens": result.n_tokens,
                "n_tts_tokens": result.n_tts_tokens,
            },
        )

    def _process_audio_sync(
        self, audio: np.ndarray, force_listen: bool
    ) -> RuntimeOutput:
        if self._view is None:
            raise RuntimeError("runtime is not prepared")
        prefill = self._view.prefill(
            audio_waveform=audio.astype(np.float32, copy=False)
        )
        if isinstance(prefill, dict) and prefill.get("success") is False:
            raise RuntimeError(
                f"duplex prefill rejected audio: {prefill.get('reason')}"
            )
        result = self._view.generate(force_listen=force_listen)
        self._view.finalize()
        return self._to_output(result)

    async def process_audio(
        self,
        audio: np.ndarray,
        *,
        force_listen: bool,
    ) -> RuntimeOutput:
        return await asyncio.to_thread(
            self._process_audio_sync, audio, force_listen
        )

    def _inject_sync(self, query: str, evidence: str) -> dict[str, Any]:
        if self._view is None:
            raise RuntimeError("runtime is not prepared")
        self._pending_query = query
        self._pending_evidence = evidence
        return self._view.inject_knowledge(query=query, evidence=evidence)

    async def inject_knowledge(self, query: str, evidence: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._inject_sync, query, evidence)

    def _answer_sync(self) -> RuntimeOutput:
        if self._view is None:
            raise RuntimeError("runtime is not prepared")
        text_parts: list[str] = []
        audio_parts: list[bytes] = []
        metrics: dict[str, Any] = {}
        end_of_turn = False

        for unit_index in range(KNOWLEDGE_ANSWER_MAX_UNITS):
            result = self._view.generate(force_speak_override=True)
            self._view.finalize()
            output = self._to_output(result)
            if output.text:
                text_parts.append(output.text)
            if output.audio_b64:
                audio_parts.append(base64.b64decode(output.audio_b64))
            metrics = {**metrics, **output.metrics, "knowledge_units": unit_index + 1}
            end_of_turn = output.end_of_turn
            combined_text = "".join(text_parts)
            if not should_continue_knowledge_answer(
                combined_text,
                end_of_turn=end_of_turn,
                unit_index=unit_index,
                max_units=KNOWLEDGE_ANSWER_MAX_UNITS,
            ):
                break
            self._view.continue_knowledge(
                query=self._pending_query,
                evidence=self._pending_evidence,
                previous_text=combined_text,
            )

        combined_audio = b"".join(audio_parts)
        self._pending_query = ""
        self._pending_evidence = ""
        return RuntimeOutput(
            text="".join(text_parts),
            audio_b64=(
                base64.b64encode(combined_audio).decode("ascii")
                if combined_audio
                else None
            ),
            is_listen=not text_parts and not audio_parts,
            end_of_turn=end_of_turn,
            metrics=metrics,
        )

    async def generate_answer(self) -> RuntimeOutput:
        return await asyncio.to_thread(self._answer_sync)

    async def interrupt(self) -> None:
        if self._view is not None:
            await asyncio.to_thread(self._view.set_break)

    async def close(self) -> None:
        if self._view is not None:
            await asyncio.to_thread(self._view.cleanup)
            self._view = None
        if self._processor is not None:
            self._processor = None
