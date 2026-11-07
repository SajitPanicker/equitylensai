"""LangGraph question-answering workflow for Equity Lens."""

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


def _history_for_answer(
    messages: list[AnyMessage], question: str
) -> list[dict[str, str]]:
    """Convert checkpoint messages to the history format used by the existing RAG."""

    history_messages = messages
    if (
        messages
        and isinstance(messages[-1], HumanMessage)
        and messages[-1].content == question
    ):
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


def build_equitylens_graph(
    *,
    collection: Any,
    model_provider: Callable[[], Any],
    config: Any,
    process_user_question: Callable[..., dict[str, Any]],
    find_unmatched_filter_values: Callable[
        [Any, dict[str, str | list[str]]], dict[str, list[str]]
    ],
    checkpointer: SqliteSaver,
):
    """Build the small question-answering graph around existing application logic."""

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
        }

    def validation_node(state: EquityLensState) -> dict[str, Any]:
        question = state.get("question", "")
        if not question:
            answer = "Enter a question to search the indexed exchange filings."
            return {
                "error_message": answer,
                "result": {
                    "answer": answer,
                    "sources": [],
                    "is_error": True,
                },
            }

        if collection.count() == 0:
            return {
                "result": {
                    "answer": (
                        "No exchange filings are indexed yet. Discover filings above "
                        "and index selected documents first."
                    ),
                    "sources": [],
                    "is_error": False,
                }
            }

        filters = state.get("filters") or {}
        unmatched_filters = (
            find_unmatched_filter_values(collection, filters) if filters else {}
        )
        return {"unmatched_filters": unmatched_filters}

    def route_after_validation(state: EquityLensState) -> str:
        return "response_node" if state.get("result") is not None else "rag_processing_node"

    def rag_processing_node(state: EquityLensState) -> dict[str, Any]:
        question = state["question"]
        raw_result = process_user_question(
            question=question,
            collection=collection,
            model=model_provider(),
            history=_history_for_answer(state.get("messages", []), question),
            config=config,
            filters=state.get("filters") or {},
        )
        image_path = _write_image_reference(raw_result.get("image_bytes"))
        serializable_result = {
            key: value for key, value in raw_result.items() if key != "image_bytes"
        }
        serializable_result["image_path"] = image_path
        return {"result": serializable_result}

    def response_node(state: EquityLensState) -> dict[str, Any]:
        result = state.get("result") or {}
        answer = str(result.get("answer", "The question could not be processed."))
        sources = result.get("sources") or []
        is_error = bool(result.get("is_error", False))
        image_path = result.get("image_path")
        return {
            "result": {
                "answer": answer,
                "sources": sources,
                "image_path": image_path,
                "is_error": is_error,
            },
            "answer": answer,
            "sources": sources,
            "image_path": image_path,
            "is_error": is_error,
            "error_message": answer if is_error else None,
        }

    def conversation_update_node(state: EquityLensState) -> dict[str, Any]:
        answer = state.get("answer", "")
        messages = state.get("messages", [])
        if (
            messages
            and isinstance(messages[-1], AIMessage)
            and messages[-1].content == answer
        ):
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
    builder.add_node("rag_processing_node", rag_processing_node)
    builder.add_node("response_node", response_node)
    builder.add_node("conversation_update_node", conversation_update_node)
    builder.add_edge(START, "input_node")
    builder.add_edge("input_node", "validation_node")
    builder.add_conditional_edges(
        "validation_node",
        route_after_validation,
        {
            "rag_processing_node": "rag_processing_node",
            "response_node": "response_node",
        },
    )
    builder.add_edge("rag_processing_node", "response_node")
    builder.add_edge("response_node", "conversation_update_node")
    builder.add_edge("conversation_update_node", END)
    return builder.compile(checkpointer=checkpointer)
