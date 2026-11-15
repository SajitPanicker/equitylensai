"""Read-only client for Zerodha's hosted Kite MCP server.

Each instance owns one MCP session and one in-memory OAuth token store. Do not
share an instance between Equity Lens users or persist its authentication state.
"""

from __future__ import annotations

import asyncio
import json
import re
import webbrowser
from contextlib import AsyncExitStack
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

import httpx2
from mcp import ClientSession, MCPError
from mcp.client.auth import (
    AuthorizationCodeResult,
    OAuthClientProvider,
    OAuthFlowError,
    TokenStorage,
)
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from mcp.types import PaginatedRequestParams, Tool
from pydantic import AnyUrl


ZERODHA_MCP_URL = "https://mcp.kite.trade/mcp"

# This explicit allowlist makes it impossible for this module to invoke trading tools.
READ_ONLY_TOOLS = frozenset(
    {
        "login",
        "portfolio",
        "get_profile",
        "get_holdings",
        "get_positions",
        "get_margins",
    }
)
PORTFOLIO_READ_MODES = frozenset({"profile", "margins", "holdings", "positions"})


class ZerodhaMCPError(RuntimeError):
    """Base error for a failed connection or read-only tool call."""


class MCPConnectionError(ZerodhaMCPError):
    """The hosted MCP server could not be reached or the MCP session expired."""


class MCPAuthenticationError(ZerodhaMCPError):
    """The MCP or Kite authentication flow is missing, denied, or expired."""


class ToolNotFoundError(ZerodhaMCPError):
    """The connected server does not advertise a requested tool."""


class ToolSchemaError(ZerodhaMCPError):
    """A discovered tool schema requires arguments this client cannot infer."""


class ToolExecutionError(ZerodhaMCPError):
    """The server returned an error for a tool call."""


class _MemoryTokenStorage(TokenStorage):
    """Keep MCP OAuth material only for the lifetime of this client instance."""

    def __init__(self) -> None:
        self._tokens: OAuthToken | None = None
        self._client_info: OAuthClientInformationFull | None = None

    async def get_tokens(self) -> OAuthToken | None:
        return self._tokens

    async def set_tokens(self, tokens: OAuthToken) -> None:
        self._tokens = tokens

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        return self._client_info

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        self._client_info = client_info


class ZerodhaMCPClient:
    """One user's connection to the official hosted Kite MCP server.

    The SDK performs the server's OAuth/PKCE flow during initialization. Keep
    this object open for any advertised Kite login step and every subsequent
    portfolio request; a new MCP session is not assumed to inherit auth state.
    """

    def __init__(
        self,
        endpoint: str = ZERODHA_MCP_URL,
        browser_opener: Callable[[str], bool] | None = None,
    ) -> None:
        self.endpoint = endpoint
        self._browser_opener = browser_opener or (lambda url: webbrowser.open(url, new=2))
        self._stack: AsyncExitStack | None = None
        self._callback_server: asyncio.AbstractServer | None = None
        self._callback_result: asyncio.Future[AuthorizationCodeResult] | None = None
        self._session: ClientSession | None = None
        self._tools: dict[str, Tool] = {}

    async def __aenter__(self) -> ZerodhaMCPClient:
        await self.connect()
        return self

    async def __aexit__(
        self, _exc_type: Any, _exc: Any, _traceback: Any
    ) -> None:
        _ = (_exc_type, _exc, _traceback)
        await self.close()

    async def connect(self) -> None:
        """Create and initialize exactly one OAuth-authenticated MCP session."""
        if self._session is not None:
            raise MCPConnectionError("This client already has an active MCP session.")

        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        try:
            # Bind a loopback callback before registering the OAuth redirect URI.
            self._callback_result = asyncio.get_running_loop().create_future()
            self._callback_server = await asyncio.start_server(
                self._handle_oauth_callback, host="127.0.0.1", port=0
            )
            callback_port = self._callback_server.sockets[0].getsockname()[1]
            redirect_uri = AnyUrl(f"http://127.0.0.1:{callback_port}/callback")

            oauth = OAuthClientProvider(
                server_url=self.endpoint,
                client_metadata=OAuthClientMetadata(
                    client_name="Equity Lens read-only portfolio client",
                    redirect_uris=[redirect_uri],
                    scope="default",
                ),
                storage=_MemoryTokenStorage(),
                redirect_handler=self._open_oauth_page,
                callback_handler=self._wait_for_oauth_callback,
            )
            http_client = await self._stack.enter_async_context(
                httpx2.AsyncClient(
                    auth=oauth,
                    timeout=httpx2.Timeout(30.0, read=300.0),
                )
            )
            read_stream, write_stream = await self._stack.enter_async_context(
                streamable_http_client(self.endpoint, http_client=http_client)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
        except OAuthFlowError as error:
            await self.close()
            raise MCPAuthenticationError(
                "Authorization for the hosted MCP server failed or was declined. "
                "Run the connection again to restart its browser flow."
            ) from error
        except (httpx2.HTTPError, MCPError, TimeoutError) as error:
            await self.close()
            raise MCPConnectionError(
                "Could not connect to the hosted Zerodha MCP server. "
                "Check your internet connection and try again."
            ) from error
        except Exception:
            await self.close()
            raise

    async def close(self) -> None:
        """Close the current transport and local callback listener."""
        if self._stack is not None:
            stack, self._stack = self._stack, None
            self._session = None
            self._tools.clear()
            await stack.aclose()
        if self._callback_server is not None:
            self._callback_server.close()
            await self._callback_server.wait_closed()
            self._callback_server = None

    async def discover_tools(self) -> list[Tool]:
        """Return the tools and schemas advertised by this live server session."""
        session = self._require_session()
        tools: list[Tool] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()

        try:
            while True:
                params = (
                    PaginatedRequestParams(cursor=cursor) if cursor is not None else None
                )
                response = await session.list_tools(params=params)
                tools.extend(response.tools)
                cursor = response.next_cursor
                if cursor is None:
                    break
                if cursor in seen_cursors:
                    raise MCPConnectionError(
                        "The MCP server repeated a tool-list cursor; discovery stopped."
                    )
                seen_cursors.add(cursor)
        except (httpx2.HTTPError, MCPError, TimeoutError) as error:
            raise MCPConnectionError(
                "Tool discovery failed. The MCP session may have expired; "
                "start a new test and complete the browser authorization again."
            ) from error

        self._tools = {tool.name: tool for tool in tools}
        return tools

    async def login(self) -> None:
        """Start Kite's browser login without displaying its session-bound URL."""
        tool = self._require_tool("login")
        self._require_no_unknown_required_arguments(tool)
        result = await self._call_tool("login", {})
        login_url = _extract_login_url(_tool_text(result))
        if login_url is None:
            raise MCPAuthenticationError(
                "The login tool did not return a recognizable Kite authorization link."
            )
        if not self._browser_opener(login_url):
            raise MCPAuthenticationError(
                "Could not open the Kite login page. Set a default browser and retry; "
                "the session-bound authorization link was not printed."
            )

    async def get_profile(self) -> Any:
        if "portfolio" in self._tools:
            return _payload(
                await self._call_portfolio_mode("profile")
            )
        return await self._call_without_arguments("get_profile")

    async def get_holdings(
        self, *, from_index: int | None = None, limit: int | None = None
    ) -> Any:
        if "portfolio" in self._tools:
            return await self._call_portfolio_rows(
                "holdings", from_index=from_index, limit=limit
            )
        return await self._call_paginated(
            "get_holdings", from_index=from_index, limit=limit
        )

    async def get_positions(
        self, *, from_index: int | None = None, limit: int | None = None
    ) -> Any:
        if "portfolio" in self._tools:
            return await self._call_portfolio_rows(
                "positions", from_index=from_index, limit=limit
            )
        return await self._call_paginated(
            "get_positions", from_index=from_index, limit=limit
        )

    async def get_margins(self) -> Any:
        if "portfolio" in self._tools:
            return _payload(await self._call_portfolio_mode("margins"))
        return await self._call_without_arguments("get_margins")

    async def _open_oauth_page(self, authorization_url: str) -> None:
        # OAuth URLs contain transient state and PKCE parameters, so never print them.
        if not self._browser_opener(authorization_url):
            raise MCPAuthenticationError(
                "Could not open the hosted MCP authorization page in a browser."
            )

    async def _wait_for_oauth_callback(self) -> AuthorizationCodeResult:
        if self._callback_result is None:
            raise MCPAuthenticationError("The OAuth callback listener is unavailable.")
        return await self._callback_result

    async def _handle_oauth_callback(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Receive the OAuth redirect locally without logging its authorization code."""
        try:
            request_line = await asyncio.wait_for(reader.readline(), timeout=15)
            method, path, *_ = request_line.decode("ascii", errors="replace").split()
            while await asyncio.wait_for(reader.readline(), timeout=15) not in (
                b"\r\n",
                b"\n",
                b"",
            ):
                pass

            parsed_url = urlsplit(path)
            query = parse_qs(parsed_url.query)
            if method != "GET" or parsed_url.path != "/callback":
                await _send_callback_response(writer, 404, "Not found.")
                return

            if "error" in query:
                if self._callback_result and not self._callback_result.done():
                    self._callback_result.set_exception(
                        MCPAuthenticationError(
                            "Hosted MCP authorization was declined or failed."
                        )
                    )
                await _send_callback_response(
                    writer, 400, "Authorization was not completed. You may close this tab."
                )
                return

            code = query.get("code", [None])[0]
            if not code:
                if self._callback_result and not self._callback_result.done():
                    self._callback_result.set_exception(
                        MCPAuthenticationError(
                            "The hosted MCP authorization callback did not include a code."
                        )
                    )
                await _send_callback_response(
                    writer, 400, "Authorization was not completed. You may close this tab."
                )
                return

            if self._callback_result and not self._callback_result.done():
                self._callback_result.set_result(
                    AuthorizationCodeResult(
                        code=code,
                        state=query.get("state", [None])[0],
                        iss=query.get("iss", [None])[0],
                    )
                )
            await _send_callback_response(
                writer, 200, "MCP authorization received. You may return to the terminal."
            )
        except (asyncio.TimeoutError, UnicodeDecodeError, ValueError):
            await _send_callback_response(
                writer, 400, "Invalid authorization callback. You may close this tab."
            )
        finally:
            writer.close()
            await writer.wait_closed()

    async def _call_without_arguments(self, name: str) -> Any:
        tool = self._require_tool(name)
        self._require_no_unknown_required_arguments(tool)
        return _payload(await self._call_tool(name, {}))

    async def _call_paginated(
        self, name: str, *, from_index: int | None, limit: int | None
    ) -> Any:
        tool = self._require_tool(name)
        schema = _tool_schema(tool)
        required = set(schema.get("required", []))
        properties = schema.get("properties", {})
        arguments: dict[str, int] = {}

        for argument_name, value in (("from", from_index), ("limit", limit)):
            if value is not None:
                if argument_name not in properties:
                    raise ToolSchemaError(
                        f"The live {name} schema does not accept {argument_name!r}."
                    )
                arguments[argument_name] = value

        missing = required - arguments.keys()
        if missing:
            raise ToolSchemaError(
                f"The live {name} schema requires arguments this client cannot infer: "
                f"{', '.join(sorted(missing))}."
            )
        return _payload(await self._call_tool(name, arguments))

    async def _call_portfolio_mode(
        self, mode: str, extra_arguments: dict[str, int | str] | None = None
    ) -> Any:
        arguments = self._portfolio_arguments(mode, extra_arguments or {})
        return await self._call_tool("portfolio", arguments)

    async def _call_portfolio_rows(
        self, mode: str, *, from_index: int | None, limit: int | None
    ) -> Any:
        tool = self._require_tool("portfolio")
        properties = _tool_schema(tool).get("properties", {})
        base_arguments: dict[str, int | str] = {}

        if mode == "holdings" and "type" in properties:
            allowed_types = properties["type"].get("enum")
            if allowed_types is not None and "full" not in allowed_types:
                raise ToolSchemaError(
                    "The live portfolio schema does not offer full holdings."
                )
            base_arguments["type"] = "full"

        if from_index is not None:
            base_arguments["from"] = from_index
        elif "from" in properties:
            base_arguments["from"] = properties["from"].get("default", 0)

        if limit is not None:
            base_arguments["limit"] = limit
        elif "limit" in properties:
            limit_schema = properties["limit"]
            page_size = limit_schema.get(
                "maximum", limit_schema.get("default")
            )
            if isinstance(page_size, int) and page_size > 0:
                base_arguments["limit"] = page_size

        return _payload(await self._call_portfolio_mode(mode, base_arguments))

    def _portfolio_arguments(
        self, mode: str, extra_arguments: dict[str, int | str]
    ) -> dict[str, int | str]:
        tool = self._require_tool("portfolio")
        schema = _tool_schema(tool)
        properties = schema.get("properties", {})
        mode_schema = properties.get("mode", {})
        available_modes = mode_schema.get("enum")

        if mode not in PORTFOLIO_READ_MODES or (
            available_modes is not None and mode not in available_modes
        ):
            raise ToolNotFoundError(
                f"The live portfolio schema does not support read-only mode {mode!r}."
            )
        if "mode" not in properties:
            raise ToolSchemaError("The live portfolio schema does not define mode.")

        for argument_name in extra_arguments:
            if argument_name not in properties:
                raise ToolSchemaError(
                    f"The live portfolio schema does not accept "
                    f"{argument_name!r}."
                )

        arguments = {"mode": mode, **extra_arguments}
        missing = set(schema.get("required", [])) - arguments.keys()
        if missing:
            raise ToolSchemaError(
                "The live portfolio schema requires arguments this client "
                f"cannot infer: {', '.join(sorted(missing))}."
            )
        return arguments

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in READ_ONLY_TOOLS:
            raise ToolNotFoundError(
                f"Tool {name!r} is outside the read-only Phase 1 allowlist."
            )
        if name == "portfolio" and arguments.get("mode") not in PORTFOLIO_READ_MODES:
            raise ToolNotFoundError(
                "Only profile, margins, holdings, and positions portfolio modes "
                "are allowed in Phase 1."
            )
        session = self._require_session()
        try:
            result = await session.call_tool(name, arguments)
        except (httpx2.HTTPError, MCPError, TimeoutError) as error:
            raise MCPConnectionError(
                "The MCP connection failed. Its server-side session may have expired; "
                "this client will not reconnect automatically because authentication "
                "may not survive a new session."
            ) from error

        if _field(result, "isError", "is_error", default=False):
            message = _tool_text(result)
            if _looks_like_authentication_error(message):
                raise MCPAuthenticationError(
                    "Authentication is missing or expired. Complete the "
                    "server-advertised authorization flow and retry."
                )
            raise ToolExecutionError(
                f"The hosted MCP server reported an error while calling {name}."
            )
        return result

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPConnectionError(
                "No MCP session is active. Use this client as an async context manager."
            )
        return self._session

    def _require_tool(self, name: str) -> Tool:
        if name not in READ_ONLY_TOOLS:
            raise ToolNotFoundError(
                f"Tool {name!r} is outside the read-only Phase 1 allowlist."
            )
        if name not in self._tools:
            raise ToolNotFoundError(
                f"The connected server has not advertised {name!r}. "
                "Run discover_tools() first and check the live tool list."
            )
        return self._tools[name]

    @staticmethod
    def _require_no_unknown_required_arguments(tool: Tool) -> None:
        required = _tool_schema(tool).get("required", [])
        if required:
            raise ToolSchemaError(
                f"The live {tool.name} schema requires arguments this client "
                f"cannot infer: {', '.join(required)}."
            )


async def _send_callback_response(
    writer: asyncio.StreamWriter, status: int, message: str
) -> None:
    body = (
        "<!doctype html><html><meta charset='utf-8'>"
        f"<p>{message}</p></html>"
    ).encode("utf-8")
    reason = "OK" if status == 200 else "Bad Request" if status == 400 else "Not Found"
    writer.write(
        f"HTTP/1.1 {status} {reason}\r\n"
        "Content-Type: text/html; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n".encode("ascii")
        + body
    )
    await writer.drain()


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _tool_schema(tool: Any) -> dict[str, Any]:
    schema = _field(tool, "inputSchema", "input_schema", default={})
    return schema if isinstance(schema, dict) else {}


def _tool_text(result: Any) -> str:
    content = _field(result, "content", default=[]) or []
    return "\n".join(
        text
        for item in content
        if isinstance((text := _field(item, "text")), str)
    )


def _payload(result: Any) -> Any:
    structured = _field(result, "structuredContent", "structured_content")
    if structured is not None:
        return structured

    text = _tool_text(result).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _extract_login_url(text: str) -> str | None:
    match = re.search(r"\[Login to Kite\]\((https://[^)\s]+)\)", text)
    if match is None:
        return None
    return match.group(1)


def _looks_like_authentication_error(message: str) -> bool:
    normalized = message.casefold()
    return any(
        phrase in normalized
        for phrase in (
            "please log in",
            "please login",
            "login required",
            "authentication required",
            "not authenticated",
            "unauthorized",
            "invalid token",
            "token expired",
            "session expired",
            "session not found",
        )
    )
