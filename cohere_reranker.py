"""Optional Cohere reranking for ChromaDB search results."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, TypeVar

logger = logging.getLogger(__name__)
_Document = TypeVar("_Document")
_REQUEST_TIMEOUT_SECONDS = 20.0


def _create_client(api_key: str) -> Any:
    import cohere

    return cohere.ClientV2(api_key=api_key, timeout=_REQUEST_TIMEOUT_SECONDS)


def _document_text(document: Any) -> str | None:
    if isinstance(document, dict):
        text = document.get("text")
    else:
        text = getattr(document, "page_content", None)
    return text if isinstance(text, str) and text else None


def rerank_documents(
    question: str,
    documents: Sequence[_Document],
    top_k: int,
    *,
    api_key: str | None,
    model: str,
) -> list[_Document]:
    """Return original documents in Cohere relevance order, or ChromaDB order on failure."""

    original_documents = list(documents)
    if not original_documents:
        logger.info("No ChromaDB documents to rerank")
        return original_documents
    if top_k < 1:
        raise ValueError("Reranking top_k must be at least 1.")
    fallback_documents = original_documents[:top_k]
    if not api_key or not api_key.strip():
        logger.warning(
            "Cohere reranking skipped because COHERE_API_KEY is unset; "
            "using the first %d ChromaDB result(s) from %d candidate(s)",
            len(fallback_documents),
            len(original_documents),
        )
        return fallback_documents
    if not model or not model.strip():
        logger.warning(
            "Cohere reranking skipped because COHERE_RERANK_MODEL is empty; "
            "using the first %d ChromaDB result(s) from %d candidate(s)",
            len(fallback_documents),
            len(original_documents),
        )
        return fallback_documents

    texts = [_document_text(document) for document in original_documents]
    if any(text is None for text in texts):
        logger.warning(
            "Cohere reranking skipped because a retrieved document has no text; "
            "using the first %d ChromaDB result(s) from %d candidate(s)",
            len(fallback_documents),
            len(original_documents),
        )
        return fallback_documents

    requested_count = min(top_k, len(original_documents))
    logger.info(
        "Submitting %d ChromaDB document(s) to Cohere reranking",
        len(original_documents),
    )
    try:
        client = _create_client(api_key.strip())
        response = client.rerank(
            model=model.strip(),
            query=question,
            documents=texts,
            top_n=requested_count,
        )
        results = response.get("results") if isinstance(response, dict) else response.results
        if not isinstance(results, (list, tuple)) or not results:
            raise ValueError("Cohere returned no reranking results.")

        indices: list[int] = []
        for result in results:
            index = result.get("index") if isinstance(result, dict) else getattr(result, "index", None)
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("Cohere returned a result with an invalid document index.")
            if index < 0 or index >= len(original_documents) or index in indices:
                raise ValueError("Cohere returned an out-of-range or duplicate document index.")
            indices.append(index)
        if len(indices) > requested_count:
            raise ValueError("Cohere returned more documents than requested.")

        reranked = [original_documents[index] for index in indices]
        logger.info(
            "Cohere reranking succeeded: submitted %d document(s), returned %d",
            len(original_documents),
            len(reranked),
        )
        return reranked
    except Exception as error:
        logger.warning(
            "Cohere reranking failed (%s); falling back to ChromaDB similarity-ranked "
            "results (%d document(s))",
            type(error).__name__,
            len(fallback_documents),
        )
        return fallback_documents
