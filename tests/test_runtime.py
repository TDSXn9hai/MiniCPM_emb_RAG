import base64

import numpy as np
import pytest

from runtime import MockDuplexRuntime, knowledge_duplex_config


@pytest.mark.asyncio
async def test_mock_runtime_keeps_listening_then_answers_after_injection():
    runtime = MockDuplexRuntime()
    await runtime.prepare("你是知识助手")

    listen = await runtime.process_audio(
        np.zeros(16000, dtype=np.float32),
        force_listen=True,
    )
    assert listen.is_listen is True
    assert listen.audio_b64 is None

    await runtime.inject_knowledge(
        "上海住宿能报多少",
        "[K1] 上海住宿标准为每晚六百元。",
    )
    answer = await runtime.generate_answer()

    assert answer.is_listen is False
    assert "六百元" in answer.text
    audio = np.frombuffer(base64.b64decode(answer.audio_b64), dtype=np.float32)
    assert audio.size > 0


def test_knowledge_duplex_config_disables_startup_force_listen():
    config = knowledge_duplex_config()

    assert config.force_listen_count == 0
    assert config.temperature <= 0.5
