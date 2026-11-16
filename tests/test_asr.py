import numpy as np
import pytest

from asr import MockASR, clean_transcript, create_asr_backend
from server import Settings, create_asr


def test_clean_transcript_removes_non_speech_tags_and_spaces():
    assert clean_transcript(" [Music] 上海 出差  住宿标准。 [noise] ") == (
        "上海 出差 住宿标准。"
    )


@pytest.mark.asyncio
async def test_mock_asr_returns_configured_transcript():
    backend = MockASR("上海出差住宿能报多少")

    text = await backend.transcribe(np.zeros(16000, dtype=np.float32))

    assert text == "上海出差住宿能报多少"
    assert create_asr_backend("mock", transcript=text).__class__ is MockASR


@pytest.mark.asyncio
async def test_server_mock_asr_uses_configured_transcript():
    backend = create_asr(
        Settings(
            asr_backend="mock",
            mock_transcript="固定测试转写",
        )
    )

    text = await backend.transcribe(np.zeros(1600, dtype=np.float32))

    assert text == "固定测试转写"
