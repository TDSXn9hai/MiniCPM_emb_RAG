from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class KnowledgeDocument:
    id: str
    title: str
    content: str
    source: str = ""


@dataclass(frozen=True)
class RetrievalHit:
    document: KnowledgeDocument
    score: float


@dataclass(frozen=True)
class Citation:
    id: str
    title: str
    source: str
    score: float


@dataclass(frozen=True)
class EvidenceCard:
    text: str
    citations: tuple[Citation, ...]


class KnowledgeBase:
    """Small offline knowledge base using character n-gram TF-IDF."""

    def __init__(
        self,
        documents: Sequence[KnowledgeDocument],
        vectorizer: TfidfVectorizer,
        matrix,
    ) -> None:
        self.documents = tuple(documents)
        self._vectorizer = vectorizer
        self._matrix = matrix

    @classmethod
    def from_documents(
        cls, documents: Sequence[KnowledgeDocument]
    ) -> "KnowledgeBase":
        docs = tuple(documents)
        if not docs:
            raise ValueError("at least one document is required")
        text = [f"{doc.title}\n{doc.content}" for doc in docs]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
        matrix = vectorizer.fit_transform(text)
        return cls(docs, vectorizer, matrix)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievalHit]:
        normalized = query.strip()
        if not normalized:
            return []
        query_vector = self._vectorizer.transform([normalized])
        scores = cosine_similarity(query_vector, self._matrix).ravel()
        ranked = sorted(
            enumerate(scores.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        return [
            RetrievalHit(document=self.documents[index], score=score)
            for index, score in ranked[: max(1, top_k)]
            if score >= min_score
        ]


def build_evidence_card(
    hits: Sequence[RetrievalHit],
    *,
    max_chars: int = 1200,
) -> EvidenceCard:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    sections: list[str] = []
    citations: list[Citation] = []
    current_length = 0
    for hit in hits:
        citation_id = f"K{len(citations) + 1}"
        doc = hit.document
        header = (
            f"[{citation_id}]\n"
            f"标题：{doc.title}\n"
            f"来源：{doc.source or '未提供'}\n"
            "内容："
        )
        available = max_chars - current_length - len(header) - 2
        if available <= 0:
            break
        content = doc.content.strip()[:available]
        if not content:
            continue
        section = f"{header}{content}"
        sections.append(section)
        citations.append(
            Citation(
                id=citation_id,
                title=doc.title,
                source=doc.source,
                score=hit.score,
            )
        )
        current_length += len(section) + 2

    return EvidenceCard(
        text="\n\n".join(sections),
        citations=tuple(citations),
    )
