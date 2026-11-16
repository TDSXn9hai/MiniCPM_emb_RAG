from rag_core import KnowledgeBase, KnowledgeDocument, build_evidence_card


def test_chinese_query_returns_relevant_document():
    knowledge_base = KnowledgeBase.from_documents(
        [
            KnowledgeDocument(
                id="travel",
                title="差旅报销制度",
                content="上海属于一线城市，住宿标准为每晚六百元。",
                source="travel-policy.pdf#page=8",
            ),
            KnowledgeDocument(
                id="leave",
                title="休假制度",
                content="员工每年享有带薪年假。",
                source="leave-policy.pdf#page=2",
            ),
        ]
    )

    hits = knowledge_base.search("上海出差住宿能报多少", top_k=1)

    assert hits[0].document.id == "travel"
    assert hits[0].score > 0


def test_evidence_card_keeps_source_numbers_and_drops_weak_hits():
    knowledge_base = KnowledgeBase.from_documents(
        [
            KnowledgeDocument(
                id="travel",
                title="差旅报销制度",
                content="上海住宿标准为每晚六百元。",
                source="travel-policy.pdf#page=8",
            ),
            KnowledgeDocument(
                id="weather",
                title="天气说明",
                content="今天多云。",
                source="weather.txt",
            ),
        ]
    )

    hits = knowledge_base.search("上海住宿六百元", top_k=2, min_score=0.1)
    card = build_evidence_card(hits, max_chars=300)

    assert card.text.startswith("[K1]")
    assert "差旅报销制度" in card.text
    assert "travel-policy.pdf#page=8" in card.text
    assert "weather.txt" not in card.text
    assert len(card.text) <= 300
