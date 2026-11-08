"""Yahoo Finance subgraph for Equity Lens."""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from yahoo_finance_source import fetch_yahoo_finance_data, resolve_ticker


class YahooFinanceState(TypedDict, total=False):
    """State for the Yahoo Finance workflow."""

    question: str
    company_name: str
    ticker: str
    request_type: str
    requested_metrics: list[str]
    yahoo_result: dict[str, Any]
    market_data: dict[str, Any]
    answer: str
    sources: list[dict[str, Any]]
    error_message: str | None
    is_error: bool


def _contains_numeric_data(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_numeric_data(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_numeric_data(item) for item in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def build_yahoo_finance_graph():
    """Create the Yahoo Finance workflow graph."""

    def ticker_resolution_node(state: YahooFinanceState) -> dict[str, Any]:
        company_name = str(state.get("company_name") or state.get("question") or "").strip()
        if not company_name:
            return {"error_message": "Please provide a company name before fetching market data.", "is_error": True}
        try:
            ticker = resolve_ticker(company_name, state.get("ticker"))
        except (TypeError, ValueError) as error:
            return {"error_message": str(error), "is_error": True}
        return {"company_name": company_name, "ticker": ticker, "requested_metrics": state.get("requested_metrics") or ["latest_market_data", "financial_summary"]}

    def yahoo_data_retrieval_node(state: YahooFinanceState) -> dict[str, Any]:
        company_name = str(state.get("company_name") or "").strip()
        ticker = str(state.get("ticker") or "").strip()
        if not company_name and not ticker:
            return {"error_message": "No company or ticker was available to look up market data.", "is_error": True}
        try:
            question = str(state.get("question") or "")
            is_historical = bool(
                re.search(
                    r"\b(?:growth|trend|last|past|previous|historical|year-by-year|"
                    r"over time|multi[- ]year)\b|\b\d+\s*(?:years?|months?)\b",
                    question,
                    re.IGNORECASE,
                )
            )
            result = fetch_yahoo_finance_data(
                company_name or ticker,
                ticker=ticker or None,
                period="5y" if is_historical else "5d",
                interval="1mo" if is_historical else "1d",
                include_history=True,
            )
        except Exception as exc:  # pragma: no cover - runtime observed by calling code.
            return {"error_message": str(exc), "is_error": True}
        return {"yahoo_result": result}

    def data_validation_node(state: YahooFinanceState) -> dict[str, Any]:
        if state.get("is_error"):
            return {"is_error": True}
        result = state.get("yahoo_result") or {}
        if not result:
            return {"error_message": "Yahoo Finance returned no data for the requested company.", "is_error": True}
        has_market_data = _contains_numeric_data(
            {
                "latest_market_data": result.get("latest_market_data"),
                "historical_prices": (result.get("historical_prices") or {}).get("points"),
                "financial_summary": result.get("financial_summary"),
                "annual_financial_trends": result.get("annual_financial_trends"),
            }
        )
        if not has_market_data:
            return {"error_message": "The requested market data was unavailable from Yahoo Finance.", "is_error": True}
        return {
            "market_data": result,
            "yahoo_result": result,
            "is_error": False,
            "error_message": None,
        }

    def yahoo_result_node(state: YahooFinanceState) -> dict[str, Any]:
        if state.get("is_error"):
            message = state.get("error_message") or "Yahoo Finance data could not be retrieved."
            return {
                "answer": message,
                "market_data": {},
                "yahoo_result": {},
                "sources": [],
                "is_error": True,
                "error_message": message,
            }
        result = state.get("market_data") or {}
        retrieved_at = result.get("data_timestamp") or "timestamp unavailable"
        return {
            "answer": "Yahoo Finance data retrieved for financial synthesis.",
            "market_data": result,
            "yahoo_result": result,
            "sources": [
                {
                    "label": f"Yahoo Finance ({result.get('ticker', 'unknown')})",
                    "snippet": f"{result.get('source', 'Yahoo Finance')}; retrieved at {retrieved_at}.",
                }
            ],
            "is_error": False,
            "error_message": None,
        }

    builder = StateGraph(YahooFinanceState)
    builder.add_node("ticker_resolution_node", ticker_resolution_node)
    builder.add_node("yahoo_data_retrieval_node", yahoo_data_retrieval_node)
    builder.add_node("data_validation_node", data_validation_node)
    builder.add_node("yahoo_result_node", yahoo_result_node)
    builder.add_edge(START, "ticker_resolution_node")
    builder.add_edge("ticker_resolution_node", "yahoo_data_retrieval_node")
    builder.add_edge("yahoo_data_retrieval_node", "data_validation_node")
    builder.add_edge("data_validation_node", "yahoo_result_node")
    builder.add_edge("yahoo_result_node", END)
    return builder.compile()
