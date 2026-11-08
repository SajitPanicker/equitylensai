"""Parent orchestration graph for Equity Lens."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path
from typing import Annotated, Any, Callable, TypedDict

import streamlit as st
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from stock_rag_chatbot import (
    answer_with_financial_context,
    classify_financial_intent,
    is_image_generation_request,
    needs_visual_clarification,
)


class EquityLensState(TypedDict, total=False):
    """Serializable state for one question and its conversation."""

    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    filters: dict[str, str | list[str]]
    result: dict[str, Any] | None
    answer: str
    sources: list[dict[str, Any]]
    image_path: str | None
    error_message: str | None
    is_error: bool
    unmatched_filters: dict[str, list[str]]
    company_name: str
    ticker: str
    analysis_type: str
    use_rag: bool
    use_yahoo_finance: bool
    classification_reason: str
    classification_error: str | None
    rag_result: dict[str, Any]
    yahoo_result: dict[str, Any]
    document_evidence: list[dict[str, Any]]
    market_data: dict[str, Any]


class SqliteCheckpointResource:
    """Keep one SQLite saver connection open for the Streamlit app lifetime."""

    def __init__(self, database_path: str) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._context_manager = SqliteSaver.from_conn_string(str(path))
        self.checkpointer = self._context_manager.__enter__()

    def close(self) -> None:
        self._context_manager.__exit__(None, None, None)


@st.cache_resource(on_release=lambda resource: resource.close())
def get_sqlite_checkpointer(database_path: str) -> SqliteCheckpointResource:
    """Open and cache the checkpointer instead of reconnecting for each question."""

    return SqliteCheckpointResource(database_path)


def new_thread_id() -> str:
    """Return a UUID identifying one conversation."""

    return str(uuid.uuid4())


def _write_image_reference(image_bytes: bytes | None) -> str | None:
    """Keep image bytes out of checkpoint state and retain only a local file path."""

    if not image_bytes:
        return None
    image_directory = Path(tempfile.gettempdir()) / "equitylens_generated_images"
    image_directory.mkdir(parents=True, exist_ok=True)
    image_path = image_directory / f"{uuid.uuid4()}.png"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def _history_for_answer(messages: list[AnyMessage], question: str) -> list[dict[str, str]]:
    """Convert checkpoint messages to the history format used by the existing RAG."""

    history_messages = messages
    if messages and isinstance(messages[-1], HumanMessage) and messages[-1].content == question:
        history_messages = messages[:-1]

    history: list[dict[str, str]] = []
    for message in history_messages:
        if not isinstance(message.content, str):
            continue
        if isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, HumanMessage):
            role = "user"
        else:
            continue
        history.append({"role": role, "content": message.content})
    return history


def route_by_intent(state: EquityLensState) -> str:
    """Choose a graph branch from validated booleans; invalid combinations use RAG."""

    use_rag = state.get("use_rag") is True
    use_yahoo_finance = state.get("use_yahoo_finance") is True
    if use_rag and use_yahoo_finance:
        return "both"
    if use_yahoo_finance:
        return "yahoo"
    return "rag"


def build_equitylens_graph(
    *,
    collection: Any,
    model_provider: Callable[[], Any],
    config: Any,
    process_user_question: Callable[..., dict[str, Any]],
    find_unmatched_filter_values: Callable[[Any, dict[str, str | list[str]]], dict[str, list[str]]],
    checkpointer: SqliteSaver,
    embedding_model_provider: Callable[[], Any] | None = None,
    classification_helper: Callable[[str, Any, Any], dict[str, Any]] | None = None,
    financial_analysis_runner: Callable[
        [str, list[dict[str, str]], list[dict[str, Any]], dict[str, Any], list[str], Any, Any],
        str,
    ] | None = None,
    rag_graph_builder: Callable[..., Any] | None = None,
    yahoo_graph_builder: Callable[..., Any] | None = None,
):
    """Build the parent graph that classifies and coordinates the two source subgraphs."""

    embedding_model_provider = embedding_model_provider or model_provider
    classification_helper = classification_helper or classify_financial_intent
    financial_analysis_runner = financial_analysis_runner or answer_with_financial_context

    def input_node(state: EquityLensState) -> dict[str, Any]:
        return {
            "question": str(state.get("question", "")).strip(),
            "filters": state.get("filters") or {},
            "result": None,
            "answer": "",
            "sources": [],
            "image_path": None,
            "error_message": None,
            "is_error": False,
            "unmatched_filters": {},
            "company_name": state.get("company_name", ""),
            "ticker": state.get("ticker", ""),
            "analysis_type": state.get("analysis_type", "rag"),
            "use_rag": True,
            "use_yahoo_finance": False,
            "classification_reason": "",
            "classification_error": None,
            "rag_result": {},
            "yahoo_result": {},
            "document_evidence": [],
            "market_data": {},
        }

    def validation_node(state: EquityLensState) -> dict[str, Any]:
        question = state.get("question", "")
        if not question:
            answer = "Enter a question to search the indexed exchange filings."
            return {
                "error_message": answer,
                "result": {"answer": answer, "sources": [], "is_error": True},
            }

        filters = state.get("filters") or {}
        unmatched_filters = (
            find_unmatched_filter_values(collection, filters)
            if filters and collection.count()
            else {}
        )
        return {"unmatched_filters": unmatched_filters}

    def route_after_validation(state: EquityLensState) -> str:
        return "response_node" if state.get("result") is not None else "intent_classification_node"

    def intent_classification_node(state: EquityLensState) -> dict[str, Any]:
        question = state.get("question") or ""
        if not question:
            return {
                "analysis_type": "rag",
                "use_rag": True,
                "use_yahoo_finance": False,
                "classification_reason": "An empty question cannot be classified; using RAG as a safe default.",
            }

        try:
            # Keep the LLM prompt/API call in a reusable helper; this node owns state updates.
            classification = classification_helper(question, config, model_provider())
            analysis_type = classification.get("analysis_type")
            valid_flags = {
                "rag": (True, False),
                "yahoo": (False, True),
                "both": (True, True),
            }
            if analysis_type not in valid_flags:
                raise ValueError("The classifier returned an unsupported analysis type.")
            use_rag, use_yahoo_finance = valid_flags[analysis_type]
            if (
                classification.get("use_rag") is not use_rag
                or classification.get("use_yahoo_finance") is not use_yahoo_finance
            ):
                raise ValueError("The classifier returned inconsistent source flags.")
            return {
                "company_name": str(classification.get("company_name") or "").strip(),
                "ticker": str(classification.get("ticker") or "").strip(),
                "analysis_type": analysis_type,
                "use_rag": use_rag,
                "use_yahoo_finance": use_yahoo_finance,
                "classification_reason": str(classification.get("classification_reason") or "").strip(),
                "classification_error": None,
            }
        except Exception as error:
            # A classifier failure must not stop research; use the filing workflow and disclose it.
            return {
                "company_name": "",
                "ticker": "",
                "analysis_type": "rag",
                "use_rag": True,
                "use_yahoo_finance": False,
                "classification_reason": "Intent classification failed; safely routing to RAG.",
                "classification_error": f"{type(error).__name__}: {error}",
            }

    def run_rag_subgraph(state: EquityLensState) -> dict[str, Any]:
        question = state.get("question", "")
        model = embedding_model_provider()
        history_provider = lambda current_question: _history_for_answer(
            state.get("messages", []), current_question
        )
        graph_builder = rag_graph_builder
        if graph_builder is None:
            from rag_graph import build_rag_graph

            graph_builder = build_rag_graph
        rag_options = {
            "collection": collection,
            "model": model,
            "config": config,
            "history_provider": history_provider,
        }
        if is_image_generation_request(question) or needs_visual_clarification(question):
            rag_options["question_processor"] = process_user_question
        else:
            rag_options["defer_answer"] = True
        rag_graph = graph_builder(**rag_options)
        raw_result = rag_graph.invoke(
            {"question": question, "filters": state.get("filters") or {}}
        )
        raw_result = dict(raw_result)
        image_path = _write_image_reference(raw_result.pop("image_bytes", None))
        if image_path:
            raw_result["image_path"] = image_path
        return {
            "rag_result": raw_result,
            "document_evidence": raw_result.get("document_evidence") or [],
        }

    def run_yahoo_subgraph(state: EquityLensState) -> dict[str, Any]:
        if yahoo_graph_builder is None:
            return {
                "yahoo_result": {
                    "answer": "Yahoo Finance is not configured in the parent graph.",
                    "error_message": "Yahoo Finance is not configured in the parent graph.",
                    "market_data": {},
                    "sources": [],
                    "is_error": True,
                },
                "market_data": {},
            }
        yahoo_graph = yahoo_graph_builder()
        yahoo_result = yahoo_graph.invoke(
            {
                "question": state.get("question", ""),
                "company_name": state.get("company_name", ""),
                "ticker": state.get("ticker", ""),
                "request_type": state.get("analysis_type", ""),
                "requested_metrics": ["latest_market_data", "financial_summary"],
            }
        )
        yahoo_result = dict(yahoo_result)
        return {
            "yahoo_result": yahoo_result,
            "market_data": yahoo_result.get("market_data") or {},
        }

    def rag_processing_node(state: EquityLensState) -> dict[str, Any]:
        return run_rag_subgraph(state)

    def yahoo_processing_node(state: EquityLensState) -> dict[str, Any]:
        return run_yahoo_subgraph(state)

    def parallel_dispatch_node(_: EquityLensState) -> dict[str, Any]:
        # Two outgoing edges below start both source branches in the same graph step.
        return {}

    def rag_parallel_node(state: EquityLensState) -> dict[str, Any]:
        return run_rag_subgraph(state)

    def yahoo_parallel_node(state: EquityLensState) -> dict[str, Any]:
        return run_yahoo_subgraph(state)

    def financial_synthesis_node(state: EquityLensState) -> dict[str, Any]:
        """Analyze selected source evidence after all selected branches finish."""

        rag_result = state.get("rag_result") or {}
        yahoo_result = state.get("yahoo_result") or {}
        document_evidence = state.get("document_evidence") or []
        market_data = state.get("market_data") or {}
        sources: list[dict[str, Any]] = []
        source_status: list[str] = []
        if state.get("use_rag"):
            sources.extend(rag_result.get("sources") or [])
            if rag_result.get("is_error"):
                source_status.append(
                    "Official NSE/BSE document retrieval failed: "
                    + str(rag_result.get("error_message") or rag_result.get("answer") or "unknown error")
                )
            elif not document_evidence:
                source_status.append("No relevant official NSE/BSE document evidence was retrieved.")
        if state.get("use_yahoo_finance"):
            sources.extend(yahoo_result.get("sources") or [])
            if yahoo_result.get("is_error") or not market_data:
                source_status.append(
                    "Yahoo Finance market data could not be retrieved: "
                    + str(yahoo_result.get("error_message") or yahoo_result.get("answer") or "no market data returned")
                )

        is_special_rag_request = is_image_generation_request(
            state.get("question", "")
        ) or needs_visual_clarification(state.get("question", ""))
        rag_image_result = rag_result if is_special_rag_request else None
        if rag_image_result:
            answer = str(rag_image_result.get("answer") or "")
        else:
            history = _history_for_answer(
                state.get("messages", []), state.get("question", "")
            )
            answer = financial_analysis_runner(
                state.get("question", ""),
                history,
                document_evidence if state.get("use_rag") else [],
                market_data if state.get("use_yahoo_finance") else {},
                source_status,
                config,
                model_provider(),
            )
        classification_error = state.get("classification_error")
        if classification_error:
            answer = (
                "Intent classification failed, so this answer uses RAG as a safe fallback. "
                + answer
            )
        selected_source_count = int(bool(state.get("use_rag"))) + int(
            bool(state.get("use_yahoo_finance"))
        )
        failed_source_count = len(source_status)
        all_sources_failed = (
            selected_source_count > 0 and failed_source_count == selected_source_count
        )
        if is_special_rag_request:
            all_sources_failed = bool(rag_result.get("is_error", False))
        return {
            "result": {
                "answer": answer,
                "sources": sources,
                "image_path": rag_result.get("image_path"),
                "is_error": all_sources_failed,
            },
        }

    def response_node(state: EquityLensState) -> dict[str, Any]:
        result = state.get("result") or {}
        answer = str(result.get("answer", "The question could not be processed."))
        sources = result.get("sources") or []
        is_error = bool(result.get("is_error", False))
        image_path = result.get("image_path")
        return {
            "result": {"answer": answer, "sources": sources, "image_path": image_path, "is_error": is_error},
            "answer": answer,
            "sources": sources,
            "image_path": image_path,
            "is_error": is_error,
            "error_message": answer if is_error else None,
        }

    def conversation_update_node(state: EquityLensState) -> dict[str, Any]:
        answer = state.get("answer", "")
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], AIMessage) and messages[-1].content == answer:
            return {}
        return {
            "messages": [
                AIMessage(
                    content=answer,
                    additional_kwargs={
                        "sources": state.get("sources", []),
                        "image_path": state.get("image_path"),
                        "is_error": state.get("is_error", False),
                    },
                )
            ]
        }

    builder = StateGraph(EquityLensState)
    builder.add_node("input_node", input_node)
    builder.add_node("validation_node", validation_node)
    builder.add_node("intent_classification_node", intent_classification_node)
    builder.add_node("rag_processing_node", rag_processing_node)
    builder.add_node("yahoo_processing_node", yahoo_processing_node)
    builder.add_node("parallel_dispatch_node", parallel_dispatch_node)
    builder.add_node("rag_parallel_node", rag_parallel_node)
    builder.add_node("yahoo_parallel_node", yahoo_parallel_node)
    builder.add_node("financial_synthesis_node", financial_synthesis_node)
    builder.add_node("response_node", response_node)
    builder.add_node("conversation_update_node", conversation_update_node)
    builder.add_edge(START, "input_node")
    builder.add_edge("input_node", "validation_node")
    builder.add_conditional_edges(
        "validation_node",
        route_after_validation,
        {
            "intent_classification_node": "intent_classification_node",
            "response_node": "response_node",
        },
    )
    # Conditional edges map the classifier's intent label to graph nodes; Python
    # routing, not the LLM, controls which workflow runs.
    builder.add_conditional_edges(
        "intent_classification_node",
        route_by_intent,
        {
            "rag": "rag_processing_node",
            "yahoo": "yahoo_processing_node",
            "both": "parallel_dispatch_node",
        },
    )
    builder.add_edge("rag_processing_node", "financial_synthesis_node")
    builder.add_edge("yahoo_processing_node", "financial_synthesis_node")
    builder.add_edge("parallel_dispatch_node", "rag_parallel_node")
    builder.add_edge("parallel_dispatch_node", "yahoo_parallel_node")
    # A multi-source edge is a synchronization barrier: synthesis waits for both.
    builder.add_edge(
        ["rag_parallel_node", "yahoo_parallel_node"],
        "financial_synthesis_node",
    )
    builder.add_edge("financial_synthesis_node", "response_node")
    builder.add_edge("response_node", "conversation_update_node")
    builder.add_edge("conversation_update_node", END)
    return builder.compile(checkpointer=checkpointer)
