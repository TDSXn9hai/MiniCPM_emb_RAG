from __future__ import annotations

import asyncio
import re
from typing import Protocol

import numpy as np


class ASRBackend(Protocol):
    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str: ...


_NON_SPEECH_TAG = re.compile(
    r"\[(?:music|noise|applause|silence|laughter)\]|\((?:music|noise|applause)\)",
    re.IGNORECASE,
)


def clean_transcript(text: str) -> str:
    cleaned = _NON_SPEECH_TAG.sub(" ", text)
    cleaned = re.sub(r"<\|[^|>]+\|>", " ", cleaned)
    return " ".join(cleaned.split()).strip()


class NullASR:
    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        return ""


class MockASR:
    def __init__(self, transcript: str) -> None:
        self.transcript = transcript

    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        return clean_transcript(self.transcript)


class WhisperASR:
    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        device: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.language = language
        self._processor = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = WhisperProcessor.from_pretrained(self.model_name)
        self._model = (
            WhisperForConditionalGeneration.from_pretrained(self.model_name)
            .to(device)
            .eval()
        )
        self._torch = torch
        self.device = device

    def _transcribe_sync(self, audio: np.ndarray, sample_rate: int) -> str:
        self._load()
        import librosa

        if sample_rate != 16000:
            audio = librosa.resample(
                audio.astype(np.float32),
                orig_sr=sample_rate,
                target_sr=16000,
            )
        inputs = self._processor(
            audio.astype(np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(self.device)
        attention_mask = self._torch.ones(
            input_features.shape,
            dtype=self._torch.long,
            device=self.device,
        )
        generate_kwargs = {}
        if self.language:
            generate_kwargs["forced_decoder_ids"] = (
                self._processor.get_decoder_prompt_ids(
                    language=self.language,
                    task="transcribe",
                )
            )
        with self._torch.inference_mode():
            ids = self._model.generate(
                input_features,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        return clean_transcript(text)

    async def transcribe(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> str:
        return await asyncio.to_thread(
            self._transcribe_sync, audio, sample_rate
        )


def create_asr_backend(
    name: str,
    *,
    model_name: str = "openai/whisper-small",
    device: str | None = None,
    language: str | None = None,
    transcript: str = "",
) -> ASRBackend:
    normalized = name.strip().lower()
    if normalized in {"none", "off", "manual"}:
        return NullASR()
    if normalized == "mock":
        return MockASR(transcript)
    if normalized == "whisper":
        return WhisperASR(
            model_name=model_name,
            device=device,
            language=language,
        )
    raise ValueError(f"unsupported ASR backend: {name}")
