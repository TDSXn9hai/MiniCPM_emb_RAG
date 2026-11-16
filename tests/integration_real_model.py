import asyncio
import base64
from pathlib import Path

import numpy as np

from runtime import MiniCPMDuplexRuntime


ROOT = Path(__file__).resolve().parents[1]


async def main() -> None:
    runtime = MiniCPMDuplexRuntime(
        model_path=str(ROOT.parent / "minicpm_demo" / "model"),
        ref_audio_path=str(ROOT / "assets" / "ref_audio" / "ref_minicpm_signature.wav"),
        system_prompt=(
            "Streaming Audio Conversation. 你是一名严谨的知识助手。"
            "请根据注入的知识证据简短回答。"
        ),
    )
    try:
        print("[integration] loading MiniCPM-o...")
        await runtime.prepare(
            "Streaming Audio Conversation. 你是一名严谨的知识助手。"
        )
        print("[integration] model ready; injecting evidence")
        await runtime.inject_knowledge(
            "上海出差住宿能报多少？",
            (
                "[K1]\n标题：差旅报销制度\n"
                "来源：travel-policy.pdf#page=8\n"
                "内容：上海属于一线城市，住宿标准为每晚六百元。"
            ),
        )
        answer = await runtime.generate_answer()
        samples = (
            np.frombuffer(base64.b64decode(answer.audio_b64), dtype=np.float32)
            if answer.audio_b64
            else np.array([], dtype=np.float32)
        )
        print("[integration] text:", answer.text)
        print(
            "[integration] audio:",
            f"{samples.size / 24000:.2f}s",
            "listen=",
            answer.is_listen,
        )
        print("[integration] metrics:", answer.metrics)
        assert answer.text.strip()
        assert "六百" in answer.text or "600" in answer.text
        assert samples.size > 0
    finally:
        await runtime.close()


if __name__ == "__main__":
    asyncio.run(main())
