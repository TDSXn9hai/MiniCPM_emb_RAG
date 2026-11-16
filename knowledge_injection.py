from __future__ import annotations

from typing import Any


def build_knowledge_prompt(query: str, evidence: str) -> str:
    return (
        f"知识库资料：\n{evidence.strip()}\n\n"
        f"用户问题：{query.strip()}\n"
        "请只用一句话回答，先给结论，不要展开原因，也不要重复问题。"
        "只根据以上资料回答；资料不足时直接说明。\n"
        "根据知识库资料，"
    )


def build_continuation_prompt(
    *,
    query: str,
    evidence: str,
    previous_text: str,
) -> str:
    return (
        f"知识库资料：\n{evidence.strip()}\n\n"
        f"用户问题：{query.strip()}\n"
        f"已经说过的内容：{previous_text.strip()}\n"
        "请从上一句之后继续，不要重复用户问题，也不要重复已经说过的内容。"
        "只补充一句，完成后立即结束，不要展开论证。"
    )


def inject_knowledge_context(
    capability: Any,
    *,
    query: str,
    evidence: str,
) -> dict[str, Any]:
    if capability.needs_finalize:
        raise RuntimeError("cannot inject knowledge before finalize completes")
    if not evidence.strip():
        raise ValueError("evidence must not be empty")

    text = build_knowledge_prompt(query, evidence)
    return _inject_prompt(capability, text)


def inject_continuation_context(
    capability: Any,
    *,
    query: str,
    evidence: str,
    previous_text: str,
) -> dict[str, Any]:
    text = build_continuation_prompt(
        query=query,
        evidence=evidence,
        previous_text=previous_text,
    )
    return _inject_prompt(capability, text)


def _inject_prompt(capability: Any, text: str) -> dict[str, Any]:
    if capability.needs_finalize:
        raise RuntimeError("cannot inject knowledge before finalize completes")
    if not text.strip():
        raise ValueError("knowledge prompt must not be empty")

    token_ids = [capability.unit_token_id]
    token_ids.extend(
        capability.tokenizer.encode(text, add_special_tokens=False)
    )

    capability.decoder.register_unit_start()
    embeds = capability.decoder.embed_tokens(token_ids)
    logits, _ = capability.decoder.feed(embeds, return_logits=True)

    capability.pending_logits = logits
    capability.current_mode = "knowledge"
    capability.audio_chunk_idx += 1
    capability.prefill_schema_tokens.append(list(token_ids))
    capability.force_speak_override = True

    return {
        "text": text,
        "token_count": len(token_ids),
    }
