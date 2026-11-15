"""Interactive, live integration test for the hosted Zerodha MCP server.

This script is intentionally not named ``test_*.py`` so normal unittest
discovery never opens a browser or makes broker requests.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from zerodha_source import (
    MCPAuthenticationError,
    MCPConnectionError,
    ToolExecutionError,
    ToolNotFoundError,
    ToolSchemaError,
    ZerodhaMCPClient,
)


_PRIVATE_FIELDS = {
    "access_token",
    "account_no",
    "account_number",
    "address",
    "api_key",
    "api_secret",
    "authorization",
    "authorization_code",
    "authorization_url",
    "client_id",
    "client_code",
    "client_secret",
    "demat_account_id",
    "dp_id",
    "dob",
    "email",
    "full_name",
    "login_url",
    "mobile",
    "pan",
    "password",
    "phone",
    "phone_number",
    "refresh_token",
    "request_token",
    "secret",
    "session_id",
    "state",
    "token",
    "totp",
    "user_id",
    "user_name",
    "user_shortname",
    "username",
}


def _redact(value: Any) -> Any:
    """Redact common identity fields before displaying broker responses."""
    if isinstance(value, dict):
        redacted = {}
        for key, item in value.items():
            if str(key).casefold() in _PRIVATE_FIELDS:
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _display_result(label: str, value: Any) -> None:
    print(f"\n[OK] {label}")
    print(json.dumps(_redact(value), indent=2, ensure_ascii=False, default=str))


async def _try_operation(
    label: str,
    operation: Callable[[], Awaitable[Any]],
    *,
    optional: bool = False,
) -> bool:
    try:
        value = await operation()
    except ToolNotFoundError:
        print(f"\n[UNSUPPORTED] {label}: the hosted server did not advertise this tool.")
        return optional
    except MCPAuthenticationError as error:
        print(f"\n[FAILED] {label}: {error}")
        return False
    except (MCPConnectionError, ToolExecutionError, ToolSchemaError) as error:
        print(f"\n[FAILED] {label}: {error}")
        return False
    else:
        _display_result(label, value)
        return True


async def _run_manual_test() -> int:
    print(
        "Connecting to the official hosted Kite MCP server. "
        "The first browser step authorizes this MCP client; credentials are "
        "entered only on the official authorization page."
    )
    client = ZerodhaMCPClient()

    try:
        async with client:
            tools = await client.discover_tools()
            print(f"\nDiscovered {len(tools)} live tool(s):")
            for tool in tools:
                name = tool.name
                description = getattr(tool, "description", None) or ""
                schema = getattr(tool, "inputSchema", None)
                if schema is None:
                    schema = getattr(tool, "input_schema", {})
                print(f"\n- {name}: {description}")
                print("  Input schema:")
                print(
                    json.dumps(
                        schema or {},
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )
                )

            tool_names = {tool.name for tool in tools}
            authentication_ready = False
            if "login" in tool_names:
                print(
                    "\nStarting the server's separate Kite login tool. Its "
                    "session-bound link will be opened in your browser and not printed."
                )
                try:
                    await client.login()
                except (MCPAuthenticationError, ToolExecutionError, ToolSchemaError) as error:
                    print(f"\n[FAILED] Kite login: {error}")
                else:
                    authentication_ready = True
                    print(
                        "\n[STARTED] Complete Zerodha's secure login and 2FA in "
                        "the browser. Do not enter your password or TOTP in this "
                        "terminal. After the browser confirms success, return here "
                        "and press Enter."
                    )
                    await asyncio.to_thread(input)
            elif "portfolio" in tool_names:
                authentication_ready = True
                print(
                    "\n[OK] The live server exposes a unified portfolio tool and "
                    "no separate login tool. MCP browser authorization completed "
                    "during connection; continuing with its read-only modes."
                )
            else:
                print(
                    "\n[FAILED] No supported login or portfolio tool was advertised."
                )

            operations = (
                ("Profile", client.get_profile, False),
                ("Equity holdings", client.get_holdings, False),
                ("Positions", client.get_positions, False),
                ("Margins", client.get_margins, True),
            )
            operation_results = []
            for label, operation, optional in operations:
                operation_results.append(
                    await _try_operation(label, operation, optional=optional)
                )

            print(
                "\nFinished. No broker response or authentication token was written "
                "to disk by this test."
            )
            return 0 if authentication_ready and all(operation_results) else 1
    except MCPAuthenticationError as error:
        print(f"\n[FAILED] Hosted MCP authorization: {error}", file=sys.stderr)
        return 2
    except MCPConnectionError as error:
        print(f"\n[FAILED] Hosted MCP connection: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n[STOPPED] Integration test cancelled.")
        return 130
    except Exception as error:
        # Do not echo arbitrary SDK/server exceptions, which can include URL data.
        print(
            f"\n[FAILED] Unexpected {type(error).__name__}. "
            "No exception details were printed to avoid exposing authorization data.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_run_manual_test()))
