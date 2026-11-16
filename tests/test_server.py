import base64
from pathlib import Path

import librosa
import numpy as np
from fastapi.testclient import TestClient

from asr import MockASR
from rag_core import KnowledgeBase, KnowledgeDocument
from runtime import MockDuplexRuntime
from server import Settings, create_app


def build_test_app():
    return create_app(
        settings=Settings(
            runtime_backend="mock",
            asr_backend="mock",
            default_system_prompt="你是测试知识助手",
        ),
        runtime=MockDuplexRuntime(),
        asr=MockASR("上海住宿能报多少"),
        knowledge_base=KnowledgeBase.from_documents(
            [
                KnowledgeDocument(
                    id="travel",
                    title="差旅报销制度",
                    content="上海住宿标准为每晚六百元。",
                    source="travel.pdf#page=8",
                )
            ]
        ),
    )


def receive_until(ws, event_type, limit=12):
    events = []
    for _ in range(limit):
        event = ws.receive_json()
        events.append(event)
        if event["type"] == event_type:
            return events
    raise AssertionError(f"did not receive {event_type}: {events}")


def test_text_query_runs_retrieval_injection_and_answer():
    with TestClient(build_test_app()) as client:
        with client.websocket_connect("/ws/a2a") as ws:
            assert ws.receive_json()["type"] == "server.ready"
            ws.send_json({"type": "session.start"})
            assert ws.receive_json()["type"] == "session.ready"

            ws.send_json(
                {
                    "type": "text.query",
                    "text": "上海出差住宿能报多少",
                }
            )
            events = receive_until(ws, "answer.done")

            event_types = [event["type"] for event in events]
            assert "asr.final" in event_types
            assert "retrieval.result" in event_types
            assert "knowledge.injected" in event_types
            assert "answer.text" in event_types
            assert "answer.audio" in event_types
            retrieval = next(
                event for event in events if event["type"] == "retrieval.result"
            )
            assert retrieval["citations"][0]["title"] == "差旅报销制度"


def test_audio_append_is_acknowledged():
    samples = np.zeros(1600, dtype="<f4")
    encoded = base64.b64encode(samples.tobytes()).decode("ascii")
    with TestClient(build_test_app()) as client:
        with client.websocket_connect("/ws/a2a") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.start"})
            ws.receive_json()
            ws.send_json(
                {
                    "type": "audio.append",
                    "sequence": 1,
                    "audio": encoded,
                }
            )
            ack = ws.receive_json()
            assert ack["type"] == "audio.ack"
            assert ack["sequence"] == 1
            assert ack["samples"] == 1600


def test_speech_audio_triggers_vad_asr_retrieval_and_answer():
    app = create_app(
        settings=Settings(
            runtime_backend="mock",
            asr_backend="mock",
            vad_threshold=0.6,
            vad_silence_ms=500,
        ),
        runtime=MockDuplexRuntime(),
        asr=MockASR("上海住宿能报多少"),
        knowledge_base=KnowledgeBase.from_documents(
            [
                KnowledgeDocument(
                    id="travel",
                    title="差旅报销制度",
                    content="上海住宿标准为每晚六百元。",
                    source="travel.pdf#page=8",
                )
            ]
        ),
    )
    audio_path = (
        Path(__file__).resolve().parents[2]
        / "minicpm_demo"
        / "tests"
        / "cases"
        / "common"
        / "user_audio"
        / "当出现植物大战僵尸的时候提醒我.wav"
    )
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = np.concatenate(
        [audio, np.zeros(16000, dtype=np.float32)]
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/a2a") as ws:
            ws.receive_json()
            ws.send_json({"type": "session.start"})
            ws.receive_json()
            for index, start in enumerate(range(0, len(audio), 1600)):
                chunk = audio[start : start + 1600].astype("<f4")
                ws.send_json(
                    {
                        "type": "audio.append",
                        "sequence": index,
                        "audio": base64.b64encode(chunk.tobytes()).decode("ascii"),
                    }
                )

            events = receive_until(ws, "answer.done", limit=120)
            event_types = [event["type"] for event in events]
            assert "vad.state" in event_types
            assert "asr.final" in event_types
            assert "retrieval.result" in event_types
