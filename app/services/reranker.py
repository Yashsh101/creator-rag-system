import logging
import re
from abc import ABC, abstractmethod

from app.core.config import settings
from app.services.citation_formatter import RetrievedChunk

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        raise NotImplementedError


class NoOpReranker(Reranker):
    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return results[:top_k]


class LexicalReranker(Reranker):
    """Lightweight lexical re-scoring (not a cross-encoder).

    Blends the upstream vector/RRF score with query-term overlap and an exact
    phrase bonus. Zero-cost, zero-dependency, deterministic — a defensible
    default; set ``reranker_provider="cohere"`` for a true cross-encoder.
    """

    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        query_terms = _tokens(query)
        if not query_terms:
            return results[:top_k]

        lexical_weight = settings.local_reranker_lexical_weight
        semantic_weight = 1.0 - lexical_weight

        scored: list[RetrievedChunk] = []
        for result in results:
            text_terms = _tokens(result.chunk.text)
            lexical_score = len(query_terms & text_terms) / max(1, len(query_terms))
            phrase_bonus = 0.05 if query.lower() in result.chunk.text.lower() else 0.0
            rerank_score = min(1.0, (semantic_weight * result.score) + (lexical_weight * lexical_score) + phrase_bonus)
            scored.append(RetrievedChunk(chunk=result.chunk, score=rerank_score, source=f"{result.source}+rerank"))

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]


class CohereReranker(Reranker):
    """True cross-encoder reranking via the Cohere Rerank API.

    Falls back to lexical re-scoring if the SDK/key is unavailable or the call
    fails, so a missing key never breaks retrieval during a live demo.
    """

    def __init__(self, fallback: Reranker):
        self._fallback = fallback

    def rerank(self, query: str, results: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not results:
            return []
        try:
            import cohere

            client = cohere.Client(settings.cohere_api_key)
            documents = [result.chunk.text for result in results]
            response = client.rerank(
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                model=settings.cohere_rerank_model,
            )
            ranked: list[RetrievedChunk] = []
            for item in response.results:
                original = results[item.index]
                ranked.append(
                    RetrievedChunk(
                        chunk=original.chunk,
                        score=float(item.relevance_score),
                        source=f"{original.source}+cohere",
                    )
                )
            return ranked
        except Exception:
            logger.warning("cohere_rerank_failed_falling_back", exc_info=True, extra={"event": "cohere_rerank_failed_falling_back"})
            return self._fallback.rerank(query, results, top_k)


# Backward-compatible alias (older imports referenced LocalReranker).
LocalReranker = LexicalReranker


def build_reranker() -> Reranker:
    if not settings.reranking_enabled or settings.reranker_provider == "none":
        return NoOpReranker()
    if settings.reranker_provider == "cohere" and settings.cohere_api_key:
        return CohereReranker(fallback=LexicalReranker())
    return LexicalReranker()


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text)}

