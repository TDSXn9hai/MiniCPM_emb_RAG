import asyncio
from pathlib import Path

import librosa
import numpy as np

from asr import WhisperASR
from engine import KnowledgePipeline
from rag_core import KnowledgeBase, KnowledgeDocument
from runtime import MockDuplexRuntime
from vad.vad import StreamingVAD, StreamingVadOptions


ROOT = Path(__file__).resolve().parents[1]
SOURCE_AUDIO = (
    ROOT.parent
    / "minicpm_demo"
    / "tests"
    / "cases"
    / "common"
    / "user_audio"
    / "当出现植物大战僵尸的时候提醒我.wav"
)


async def main() -> None:
    audio, sample_rate = librosa.load(
        SOURCE_AUDIO,
        sr=16000,
        mono=True,
    )
    vad = StreamingVAD(
        StreamingVadOptions(
            threshold=0.6,
            min_silence_duration_ms=500,
            speech_pad_ms=100,
        )
    )
    segments = []
    for start in range(0, len(audio), 1600):
        segment = vad.feed(audio[start : start + 1600])
        if segment is not None and segment.size:
            segments.append(segment)
    final = vad.flush()
    if final is not None and final.size:
        segments.append(final)

    assert segments, "VAD did not detect a speech segment"
    speech = np.concatenate(segments)
    print("[audio-asr] speech seconds:", round(len(speech) / 16000, 2))

    asr = WhisperASR(
        (ROOT / "models" / "whisper-small").as_posix(),
        device="cuda:0",
        language="zh",
    )
    transcript = await asr.transcribe(speech, sample_rate)
    print("[audio-asr] transcript:", transcript)
    assert "植物" in transcript or "僵尸" in transcript

    pipeline = KnowledgePipeline(
        asr=asr,
        knowledge_base=KnowledgeBase.from_documents(
            [
                KnowledgeDocument(
                    id="plant-zombies",
                    title="植物大战僵尸提醒事项",
                    content="当用户提到植物大战僵尸时，提醒检查阳光和坚果冷却。",
                    source="test/sample-user-audio",
                ),
                KnowledgeDocument(
                    id="travel",
                    title="差旅报销制度",
                    content="上海住宿标准为每晚六百元。",
                    source="travel-policy.pdf",
                ),
            ]
        ),
        runtime=MockDuplexRuntime(),
        top_k=1,
        min_score=0.02,
    )
    await pipeline.runtime.prepare("你是知识助手")
    result = await pipeline.process_utterance(speech)
    print("[audio-asr] query:", result.query)
    print("[audio-asr] citation:", result.citations[0].title)
    assert result.citations[0].id == "K1"
    assert "植物大战僵尸" in result.evidence.text


if __name__ == "__main__":
    asyncio.run(main())
