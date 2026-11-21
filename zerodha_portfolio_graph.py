"""Independent LangGraph subgraph for read-only Zerodha portfolio data."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from zerodha_source import (
    MCPAuthenticationError,
    MCPConnectionError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaError,
    ZerodhaMCPClient,
)


class ZerodhaPortfolioState(TypedDict, total=False):
    question: str
    portfolio_data: dict[str, Any]
    zerodha_result: dict[str, Any]
    zerodha_auth_status: str
    zerodha_error: str | None
    answer: str
    sources: list[dict[str, Any]]
    is_error: bool


def build_zerodha_portfolio_graph(
    client_factory: Callable[[], ZerodhaMCPClient] | None = None,
):
    """Build a read-only workflow with an isolated MCP client per invocation."""

    client_factory = client_factory or ZerodhaMCPClient

    def auth_status_node(state: ZerodhaPortfolioState) -> dict[str, Any]:
        return {
            "zerodha_auth_status": "checking",
            "zerodha_error": None,
            "portfolio_data": {},
            "zerodha_result": {},
        }

    def retrieve_portfolio_node(state: ZerodhaPortfolioState) -> dict[str, Any]:
        async def retrieve() -> dict[str, Any]:
            async with client_factory() as client:
                await client.discover_tools()
                question = str(state.get("question") or "")
                lower = question.casefold()
                calls: dict[str, Any] = {}

                # Holdings are the default for personal-portfolio questions.
                if re.search(r"\b(holding|holdings|allocation|invested|portfolio)\b", lower):
                    calls["holdings"] = await client.get_holdings()
                if re.search(r"\b(position|positions|open trade|open position)\b", lower):
                    calls["positions"] = await client.get_positions()
                if re.search(r"\b(margin|margins|available cash|available balance)\b", lower):
                    calls["margins"] = await client.get_margins()
                if re.search(r"\b(profile|account)\b", lower):
                    calls["profile"] = await client.get_profile()
                if not calls:
                    calls["holdings"] = await client.get_holdings()
                return calls

        try:
            portfolio_data = asyncio.run(retrieve())
            return {
                "portfolio_data": portfolio_data,
                "zerodha_auth_status": "connected",
                "zerodha_error": None,
            }
        except MCPAuthenticationError as error:
            return {
                "portfolio_data": {},
                "zerodha_auth_status": "authentication_required",
                "zerodha_error": str(error),
            }
        except MCPConnectionError as error:
            return {
                "portfolio_data": {},
                "zerodha_auth_status": "disconnected",
                "zerodha_error": str(error),
            }
        except (
            ToolExecutionError,
            ToolNotFoundError,
            ToolSchemaError,
        ) as error:
            return {
                "portfolio_data": {},
                "zerodha_auth_status": "connected",
                "zerodha_error": str(error),
            }
        except Exception as error:
            return {
                "portfolio_data": {},
                "zerodha_auth_status": "disconnected",
                "zerodha_error": f"{type(error).__name__}: {error}",
            }

    def result_node(state: ZerodhaPortfolioState) -> dict[str, Any]:
        portfolio_data = state.get("portfolio_data") or {}
        auth_status = state.get("zerodha_auth_status") or "disconnected"
        error = state.get("zerodha_error")
        if auth_status == "authentication_required":
            answer = "Zerodha authentication is required. Connect to Zerodha to access personal portfolio data."
            is_error = True
        elif error:
            answer = f"Zerodha portfolio data could not be retrieved: {error}"
            is_error = True
        elif not portfolio_data:
            answer = "Zerodha returned no portfolio fields for this request."
            is_error = True
        else:
            answer = "Zerodha personal portfolio data retrieved for financial synthesis."
            is_error = False

        result = {
            "answer": answer,
            "portfolio_data": portfolio_data,
            "source": "Zerodha Kite MCP (authenticated personal account)",
            "auth_status": auth_status,
            "is_error": is_error,
            "error": error,
        }
        return {
            "zerodha_result": result,
            "answer": answer,
            "sources": [{
                "label": "Zerodha Kite MCP (personal portfolio)",
                "snippet": "Authenticated account holdings, positions, profile, or margins.",
            }] if not is_error else [],
            "is_error": is_error,
            "zerodha_error": error,
        }

    builder = StateGraph(ZerodhaPortfolioState)
    builder.add_node("auth_status_node", auth_status_node)
    builder.add_node("retrieve_portfolio_node", retrieve_portfolio_node)
    builder.add_node("result_node", result_node)
    builder.add_edge(START, "auth_status_node")
    builder.add_edge("auth_status_node", "retrieve_portfolio_node")
    builder.add_edge("retrieve_portfolio_node", "result_node")
    builder.add_edge("result_node", END)
    return builder.compile()
