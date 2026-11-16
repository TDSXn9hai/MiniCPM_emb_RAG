from __future__ import annotations


KNOWLEDGE_ANSWER_MAX_UNITS = 8


def force_speak_token(
    force_speak_override: bool,
    step: int,
    *,
    speak_token_id: int,
) -> int | None:
    if force_speak_override and step == 0:
        return speak_token_id
    return None


def knowledge_generation_limits(current_mode: str | None) -> tuple[int, int]:
    if current_mode == "knowledge":
        return 96, 220
    return 20, 28


def should_continue_knowledge_answer(
    text: str,
    *,
    end_of_turn: bool,
    unit_index: int,
    max_units: int = KNOWLEDGE_ANSWER_MAX_UNITS,
    min_chars: int = 12,
) -> bool:
    if end_of_turn or unit_index + 1 >= max_units:
        return False
    normalized = text.strip()
    if len(normalized) < min_chars:
        return True
    return normalized[-1] not in "。！？!?；;"
