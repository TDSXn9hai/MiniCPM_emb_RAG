from duplex_generation import (
    KNOWLEDGE_ANSWER_MAX_UNITS,
    force_speak_token,
    knowledge_generation_limits,
    should_continue_knowledge_answer,
)


def test_force_speak_only_overrides_the_first_generated_token():
    assert force_speak_token(True, 0, speak_token_id=42) == 42
    assert force_speak_token(True, 1, speak_token_id=42) is None
    assert force_speak_token(False, 0, speak_token_id=42) is None


def test_knowledge_unit_uses_a_longer_generation_budget():
    assert KNOWLEDGE_ANSWER_MAX_UNITS == 8
    assert knowledge_generation_limits("knowledge") == (96, 220)
    assert knowledge_generation_limits("audio") == (20, 28)


def test_short_unfinished_knowledge_answer_requests_another_unit():
    assert should_continue_knowledge_answer(
        "上海出差住宿",
        end_of_turn=False,
        unit_index=0,
        max_units=3,
    )
    assert not should_continue_knowledge_answer(
        "上海住宿标准为每晚六百元。",
        end_of_turn=False,
        unit_index=1,
        max_units=3,
        min_chars=8,
    )
    assert should_continue_knowledge_answer(
        "上海住宿标准是每晚六百元。因",
        end_of_turn=False,
        unit_index=0,
        max_units=3,
    )
