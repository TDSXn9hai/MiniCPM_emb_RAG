import pytest

from knowledge_injection import (
    build_continuation_prompt,
    build_knowledge_prompt,
    inject_knowledge_context,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) % 97 + 100 for char in text]

    def convert_tokens_to_ids(self, token):
        return 9


class FakeDecoder:
    def __init__(self):
        self.fed = []
        self.started = 0

    def register_unit_start(self):
        self.started += 1

    def embed_tokens(self, token_ids):
        return token_ids

    def feed(self, embeds, return_logits=False):
        self.fed.append(list(embeds))
        return "logits", "hidden"


class FakeCapability:
    def __init__(self, needs_finalize=False):
        self.tokenizer = FakeTokenizer()
        self.decoder = FakeDecoder()
        self.unit_token_id = 7
        self.needs_finalize = needs_finalize
        self.pending_logits = None
        self.current_mode = None
        self.audio_chunk_idx = 4
        self.prefill_schema_tokens = []
        self.force_speak_override = False


def test_inject_knowledge_context_creates_a_forced_speak_unit():
    capability = FakeCapability()

    result = inject_knowledge_context(
        capability,
        query="上海住宿能报多少",
        evidence="[K1] 上海住宿标准为六百元。",
    )

    assert capability.decoder.started == 1
    assert capability.decoder.fed[0][0] == 7
    assert capability.pending_logits == "logits"
    assert capability.current_mode == "knowledge"
    assert capability.force_speak_override is True
    assert capability.audio_chunk_idx == 5
    assert result["token_count"] > 1


def test_inject_knowledge_context_rejects_open_finalize():
    capability = FakeCapability(needs_finalize=True)

    with pytest.raises(RuntimeError, match="finalize"):
        inject_knowledge_context(
            capability,
            query="问题",
            evidence="证据",
        )


def test_continuation_prompt_excludes_repetition():
    prompt = build_continuation_prompt(
        query="住宿能报多少",
        evidence="上海住宿六百元。",
        previous_text="上海出差住宿",
    )

    assert "上海出差住宿" in prompt
    assert "不要重复" in prompt
    assert "上海住宿六百元" in prompt
    assert "只补充一句" in prompt
    assert "不要重复用户问题" in prompt


def test_primary_prompt_requests_a_single_complete_spoken_answer():
    prompt = build_knowledge_prompt("住宿能报多少", "上海住宿六百元。")

    assert "只用一句话" in prompt
    assert "不要展开原因" in prompt
