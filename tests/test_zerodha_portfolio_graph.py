"""Offline tests for the independent Zerodha portfolio subgraph."""

import unittest

from zerodha_portfolio_graph import build_zerodha_portfolio_graph
from zerodha_source import MCPAuthenticationError, MCPConnectionError, ToolExecutionError


class SuccessfulClient:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def discover_tools(self):
        return []

    async def get_holdings(self):
        return {"data": [{"tradingsymbol": "TEST", "quantity": 10}]}

    async def get_positions(self):
        return {"data": [{"tradingsymbol": "TEST", "pnl": 100}]}

    async def get_margins(self):
        return {"equity": {"available": 5000}}

    async def get_profile(self):
        return {"profile": "safe"}


class ZerodhaPortfolioGraphTests(unittest.TestCase):
    def test_holdings_retrieval(self):
        graph = build_zerodha_portfolio_graph(lambda: SuccessfulClient())
        state = graph.invoke({"question": "Show my Zerodha holdings"})

        self.assertEqual(state["zerodha_auth_status"], "connected")
        self.assertFalse(state["is_error"])
        self.assertIn("holdings", state["portfolio_data"])

    def test_positions_and_margins_retrieval(self):
        graph = build_zerodha_portfolio_graph(lambda: SuccessfulClient())
        state = graph.invoke({"question": "Show my positions and available margins"})

        self.assertIn("positions", state["portfolio_data"])
        self.assertIn("margins", state["portfolio_data"])

    def test_authentication_failure_is_explicit(self):
        class AuthFailureClient(SuccessfulClient):
            async def discover_tools(self):
                raise MCPAuthenticationError("Authentication required")

        state = build_zerodha_portfolio_graph(lambda: AuthFailureClient()).invoke(
            {"question": "Show my holdings"}
        )
        self.assertEqual(state["zerodha_auth_status"], "authentication_required")
        self.assertTrue(state["is_error"])
        self.assertIn("authentication", state["answer"].lower())

    def test_connection_and_tool_failures_are_explicit(self):
        class ConnectionFailureClient(SuccessfulClient):
            async def discover_tools(self):
                raise MCPConnectionError("Server unavailable")

        class ToolFailureClient(SuccessfulClient):
            async def get_holdings(self):
                raise ToolExecutionError("Holdings failed")

        connection = build_zerodha_portfolio_graph(lambda: ConnectionFailureClient()).invoke(
            {"question": "Show my holdings"}
        )
        tool = build_zerodha_portfolio_graph(lambda: ToolFailureClient()).invoke(
            {"question": "Show my holdings"}
        )
        self.assertEqual(connection["zerodha_auth_status"], "disconnected")
        self.assertTrue(connection["is_error"])
        self.assertTrue(tool["is_error"])
        self.assertIn("holdings", tool["zerodha_error"].lower())


if __name__ == "__main__":
    unittest.main()
