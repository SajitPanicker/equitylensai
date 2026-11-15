"""Offline tests for the read-only Zerodha MCP client."""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from mcp import MCPError

from manual_zerodha_mcp_test import _redact
from zerodha_source import (
    MCPAuthenticationError,
    MCPConnectionError,
    READ_ONLY_TOOLS,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaError,
    ZerodhaMCPClient,
)


def make_tool(name, schema=None, description="test tool"):
    return SimpleNamespace(
        name=name,
        description=description,
        inputSchema=schema
        or {"type": "object", "properties": {}, "required": []},
    )


def make_result(value=None, *, error=False, message=None):
    if message is not None:
        text = message
    else:
        text = json.dumps(value)
    return SimpleNamespace(
        isError=error,
        content=[SimpleNamespace(type="text", text=text)],
        structuredContent=None,
    )


class FakeSession:
    def __init__(self, tools, results=None, pages=None):
        self.tools = tools
        self.results = results or {}
        self.pages = pages
        self.calls = []
        self.list_calls = []

    async def list_tools(self, *, params=None):
        self.list_calls.append(params)
        if self.pages:
            return self.pages.pop(0)
        return SimpleNamespace(tools=self.tools, next_cursor=None)

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        result = self.results.get((name, arguments.get("mode")), self.results.get(name))
        if isinstance(result, list):
            return result.pop(0)
        return result


class ZerodhaSourceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.schemas = [
            make_tool(
                "portfolio",
                {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": ["profile", "margins", "holdings", "positions"],
                        },
                        "type": {
                            "type": "string",
                            "enum": ["full", "summary", "compact"],
                            "default": "full",
                        },
                        "from": {"type": "integer", "minimum": 0},
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 500,
                            "default": 100,
                        },
                    },
                    "required": ["mode"],
                },
            ),
        ]
        self.client = ZerodhaMCPClient(browser_opener=lambda url: bool(url))
        self.session = FakeSession(
            self.schemas,
            {
                "login": make_result(
                    message="[Login to Kite](https://kite.trade/login?token=redacted)"
                ),
                ("portfolio", "profile"): make_result(
                    {"profile": "safe test result"}
                ),
                ("portfolio", "holdings"): make_result(
                    {
                        "data": [{"tradingsymbol": "TEST"}],
                        "pagination": {
                            "from": 0,
                            "limit": 500,
                            "total": 1,
                            "has_more": False,
                            "returned": 1,
                        },
                    }
                ),
                ("portfolio", "positions"): make_result(
                    {
                        "data": [{"tradingsymbol": "TEST"}],
                        "pagination": {
                            "from": 0,
                            "limit": 500,
                            "total": 1,
                            "has_more": False,
                            "returned": 1,
                        },
                    }
                ),
                ("portfolio", "margins"): make_result(
                    {"equity": {"available": 1000}}
                ),
            },
        )
        self.client._session = self.session

    async def asyncSetUp(self):
        await self.client.discover_tools()

    async def test_discovers_tools_with_descriptions_and_input_schemas(self):
        names = {tool.name for tool in await self.client.discover_tools()}

        self.assertIn("portfolio", names)
        self.assertEqual(self.client._tools["portfolio"].description, "test tool")
        self.assertIn("mode", self.client._tools["portfolio"].inputSchema["properties"])

    async def test_tool_discovery_follows_server_pagination(self):
        first_page = self.schemas[:2]
        second_page = [make_tool("get_profile")]
        self.session.list_calls.clear()
        self.session.pages = [
            SimpleNamespace(tools=first_page, next_cursor="next-page"),
            SimpleNamespace(tools=second_page, next_cursor=None),
        ]

        discovered = await self.client.discover_tools()

        self.assertEqual(
            {tool.name for tool in discovered}, {"portfolio", "get_profile"}
        )
        self.assertEqual(self.session.list_calls[0], None)
        self.assertEqual(self.session.list_calls[1].cursor, "next-page")

    async def test_read_only_calls_use_discovered_schemas_and_return_payloads(self):
        profile = await self.client.get_profile()
        holdings = await self.client.get_holdings()
        positions = await self.client.get_positions()
        margins = await self.client.get_margins()

        self.assertEqual(profile, {"profile": "safe test result"})
        self.assertEqual(
            holdings,
            {
                "data": [{"tradingsymbol": "TEST"}],
                "pagination": {
                    "from": 0,
                    "limit": 500,
                    "total": 1,
                    "has_more": False,
                    "returned": 1,
                },
            },
        )
        self.assertEqual(
            positions,
            {
                "data": [{"tradingsymbol": "TEST"}],
                "pagination": {
                    "from": 0,
                    "limit": 500,
                    "total": 1,
                    "has_more": False,
                    "returned": 1,
                },
            },
        )
        self.assertEqual(margins, {"equity": {"available": 1000}})
        self.assertEqual(
            self.session.calls,
            [
                ("portfolio", {"mode": "profile"}),
                (
                    "portfolio",
                    {"mode": "holdings", "type": "full", "from": 0, "limit": 500},
                ),
                (
                    "portfolio",
                    {"mode": "positions", "from": 0, "limit": 500},
                ),
                ("portfolio", {"mode": "margins"}),
            ],
        )

    async def test_missing_tool_is_reported_without_calling_it(self):
        self.client._tools.pop("portfolio")

        with self.assertRaises(ToolNotFoundError):
            await self.client.get_holdings()

        self.assertEqual(self.session.calls, [])

    async def test_login_does_not_guess_required_arguments(self):
        self.client._tools["login"] = make_tool(
            "login",
            {
                "type": "object",
                "properties": {"confirm": {"type": "boolean"}},
                "required": ["confirm"],
            },
        )

        with self.assertRaises(ToolSchemaError):
            await self.client.login()

        self.assertEqual(self.session.calls, [])

    async def test_authentication_failure_is_explicit(self):
        self.session.results[("portfolio", "profile")] = make_result(
            error=True, message="Authentication required; please login again"
        )

        with self.assertRaises(MCPAuthenticationError):
            await self.client.get_profile()

    async def test_retrieval_error_is_explicit_and_does_not_return_fake_data(self):
        self.session.results[("portfolio", "holdings")] = make_result(
            error=True, message="Failed to retrieve portfolio holdings"
        )

        with self.assertRaises(ToolExecutionError):
            await self.client.get_holdings()

    async def test_protocol_error_does_not_trigger_a_silent_new_session(self):
        async def expired_tool_call(name, arguments):
            self.assertEqual(name, "portfolio")
            self.assertEqual(arguments, {"mode": "profile"})
            raise MCPError(code=-32000, message="Session expired")

        self.session.call_tool = expired_tool_call
        with self.assertRaises(MCPConnectionError):
            await self.client.get_profile()

        self.assertIs(self.client._session, self.session)

    async def test_pagination_arguments_can_request_a_following_page(self):
        await self.client.get_holdings(from_index=2, limit=5)

        self.assertEqual(
            self.session.calls,
            [
                (
                    "portfolio",
                    {"mode": "holdings", "type": "full", "from": 2, "limit": 5},
                )
            ],
        )

    async def test_portfolio_preserves_live_pagination_metadata(self):
        response = {
            "data": [{"tradingsymbol": "TEST"}],
            "pagination": {"from": 0, "limit": 500, "has_more": True},
        }
        self.session.results[("portfolio", "holdings")] = make_result(response)

        holdings = await self.client.get_holdings()

        self.assertEqual(holdings, response)
        self.assertEqual(
            self.session.calls,
            [
                (
                    "portfolio",
                    {"mode": "holdings", "type": "full", "from": 0, "limit": 500},
                )
            ],
        )

    async def test_trading_tool_names_are_outside_allowlist(self):
        self.assertEqual(
            READ_ONLY_TOOLS,
            {
                "login",
                "portfolio",
                "get_profile",
                "get_holdings",
                "get_positions",
                "get_margins",
            },
        )
        with self.assertRaises(ToolNotFoundError):
            await self.client._call_tool("place_order", {})
        with self.assertRaises(ToolNotFoundError):
            await self.client._call_tool("portfolio", {"mode": "orders"})

        self.assertEqual(self.session.calls, [])

    def test_manual_output_redacts_identity_fields(self):
        display_data = _redact(
            {
                "user_id": "private-user",
                "email": "private@example.test",
                "user_shortname": "private-name",
                "access_token": "private-token",
                "holdings": [{"tradingsymbol": "TEST"}],
            }
        )

        self.assertEqual(display_data["user_id"], "[REDACTED]")
        self.assertEqual(display_data["email"], "[REDACTED]")
        self.assertEqual(display_data["user_shortname"], "[REDACTED]")
        self.assertEqual(display_data["access_token"], "[REDACTED]")
        self.assertEqual(display_data["holdings"], [{"tradingsymbol": "TEST"}])


if __name__ == "__main__":
    unittest.main()
