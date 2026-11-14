"""Question-time retrieval subgraph for Equity Lens."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph


class RAGGraphState(TypedDict, total=False):
    """State used by the retrieval-and-answer subgraph."""

    question: str
    filters: dict[str, str | list[str]]
    retrieved_chunks: list[dict[str, Any]]
    document_evidence: list[dict[str, Any]]
    answer: str
    sources: list[dict[str, Any]]
    error_message: str | None
    is_error: bool
    processed_result: dict[str, Any]
    defer_answer: bool
    image_bytes: bytes | None


def _format_source_label(metadata: dict[str, Any]) -> str:
    company = metadata.get("company")
    ticker = metadata.get("ticker")
    identity = f"{company} ({ticker}) - " if company else ""
    period_parts = [
        metadata.get("fiscal_year"),
        metadata.get("quarter"),
    ]
    period = ", ".join(part for part in period_parts if part)
    period_label = f" ({period})" if period else ""
    exchange = f"{metadata.get('exchange')} - " if metadata.get("exchange") else ""
    published = f", published {metadata['publication_date']}" if metadata.get("publication_date") else ""
    title = metadata.get("original_title", metadata.get("source", "Unknown file"))
    return (
        f"{identity}{exchange}{title}{period_label}{published} "
        f"({metadata.get('location', 'location unavailable')})"
    )


def build_rag_graph(
    *,
    collection: Any,
    model: Any,
    config: Any,
    retrieval_runner: Callable[..., list[dict[str, Any]]],
    client_provider: Callable[[], Any],
    answer_runner: Callable[[str, list[dict[str, Any]], list[dict[str, str]], Any, Any], str],
    history_provider: Callable[[str], list[dict[str, str]]] | None = None,
    question_processor: Callable[..., dict[str, Any]] | None = None,
    defer_answer: bool = False,
):
    """Build a small subgraph that retrieves evidence and returns an answer."""

    history_provider = history_provider or (lambda _question: [])

    def prepare_question_node(state: RAGGraphState) -> dict[str, Any]:
        question = str(state.get("question", "")).strip()
        return {
            "question": question,
            "filters": state.get("filters") or {},
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "error_message": None,
            "is_error": False,
            "defer_answer": defer_answer,
        }

    def retrieval_node(state: RAGGraphState) -> dict[str, Any]:
        question = state.get("question") or ""
        if not question:
            return {
                "answer": "Enter a question to search the indexed exchange filings.",
                "sources": [],
                "is_error": False,
            }
        if question_processor is not None:
            result = question_processor(
                question=question,
                collection=collection,
                model=model,
                history=history_provider(question),
                config=config,
                filters=state.get("filters") or {},
            )
            return {"processed_result": result}
        filters = state.get("filters") or {}
        try:
            matches = retrieval_runner(
                question, collection, model, config.top_k, filters=filters
            )
        except Exception as error:
            return {
                "retrieved_chunks": [],
                "document_evidence": [],
                "sources": [],
                "error_message": f"{type(error).__name__}: {error}",
                "is_error": True,
            }
        if not matches:
            requested_scope = ", ".join(
                f"{field.replace('_', ' ')}: {', '.join(value) if isinstance(value, list) else value}"
                for field, value in filters.items()
            )
            answer = (
                f"No indexed documents match the requested filters ({requested_scope})."
                if requested_scope
                else "I could not find relevant text in the indexed filings to answer that question."
            )
            return {
                "answer": answer,
                "retrieved_chunks": [],
                "document_evidence": [],
                "sources": [],
                "is_error": False,
            }
        return {"retrieved_chunks": matches, "document_evidence": matches, "sources": [{
            "label": _format_source_label(match["metadata"]),
            "snippet": match["text"][:600],
            "source_url": match["metadata"].get("source_url", ""),
        } for match in matches]}

    def answer_node(state: RAGGraphState) -> dict[str, Any]:
        processed_result = state.get("processed_result")
        if processed_result is not None:
            return {
                "answer": str(processed_result.get("answer", "")),
                "sources": processed_result.get("sources") or [],
                "document_evidence": processed_result.get("document_evidence") or [],
                "is_error": bool(processed_result.get("is_error", False)),
                "error_message": processed_result.get("answer") if processed_result.get("is_error") else None,
                "image_bytes": processed_result.get("image_bytes"),
            }
        if state.get("is_error"):
            return {
                "answer": "",
                "document_evidence": state.get("document_evidence") or [],
                "sources": state.get("sources") or [],
                "is_error": True,
                "error_message": state.get("error_message"),
            }
        question = state.get("question") or ""
        matches = state.get("retrieved_chunks") or []
        if not matches:
            return {
                "answer": "" if state.get("defer_answer") else state.get("answer", "No evidence found."),
                "document_evidence": [],
                "sources": [],
                "is_error": False,
            }
        if state.get("defer_answer"):
            return {
                "answer": "",
                "document_evidence": matches,
                "sources": state.get("sources", []),
                "is_error": False,
            }
        history = history_provider(question)
        try:
            client = client_provider()
            answer = answer_runner(question, matches, history, config, client)
        except ValueError as exc:
            return {"answer": str(exc), "sources": state.get("sources", []), "is_error": True, "error_message": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive fallback for API failures.
            return {"answer": str(exc), "sources": state.get("sources", []), "is_error": True, "error_message": str(exc)}
        return {
            "answer": answer,
            "document_evidence": matches,
            "sources": state.get("sources", []),
            "is_error": False,
        }

    builder = StateGraph(RAGGraphState)
    builder.add_node("prepare_question_node", prepare_question_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("answer_node", answer_node)
    builder.add_edge(START, "prepare_question_node")
    builder.add_edge("prepare_question_node", "retrieval_node")
    builder.add_edge("retrieval_node", "answer_node")
    builder.add_edge("answer_node", END)
    return builder.compile()
