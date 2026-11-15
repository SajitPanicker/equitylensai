# Zerodha Kite MCP: Phase 1

Phase 1 is an isolated, read-only connection to the official hosted Kite MCP
server. It does not change Equity Lens routing, LangGraph checkpoints, the
Streamlit dashboard, or any existing data source.

## What MCP does here

The Model Context Protocol (MCP) standardizes how a client discovers tools and
calls them. In this integration, `zerodha_source.py` is an MCP client and
Zerodha's `https://mcp.kite.trade/mcp` is the MCP server. The server exposes
account and portfolio tools; this client permits only the legacy login tool
and read-only profile, holdings, positions, and margins calls.

The official Kite MCP repository documents the hosted `/mcp` endpoint as
Streamable HTTP. The hosted endpoint currently requires transport-level OAuth:
it advertises a protected-resource document and an authorization server with
authorization-code flow, PKCE (`S256`), and dynamic client registration. The
official MCP Python SDK handles discovery, registration, and the short-lived
OAuth callback. See the [official Kite MCP repository](https://github.com/zerodha/kite-mcp-server),
the [MCP Streamable HTTP specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports),
and the [official Python SDK](https://github.com/modelcontextprotocol/python-sdk).

**The deployed tool list may differ from the repository README and source.**
The live schema shared for the hosted server advertises a unified `portfolio`
tool with a required `mode` of `profile`, `holdings`, `positions`, or
`margins`; it does not advertise a separate `login` tool or separate
`get_profile`/`get_holdings` tools. The client follows the live schema. The MCP
OAuth browser authorization happens while the transport session initializes.
If a future/older live server does advertise a separate `login` tool, the
client checks that tool's schema before calling it.

The OAuth callback runs on loopback (`127.0.0.1`) and returns to the running
test process. Complete any sign-in and 2FA prompts only on the official browser
page; never enter credentials in the terminal.

The script does not ask for or collect a password or TOTP. It does not print
the MCP authorization URL, the session-bound Kite link, OAuth tokens, or raw
server exceptions. Profile identity fields are redacted before results are
shown. Portfolio results are displayed in the terminal for the user and are
not written to files or logs.

## Hosted endpoint credentials

The official repository's hosted-version instructions state that no API keys
are required on the client. The client does not need `KITE_API_KEY` or
`KITE_API_SECRET`, so no Zerodha values belong in `.env` or `.env.example`.
Those variables in the upstream README apply to self-hosting the server and
are not used here. MCP OAuth registration is performed by the SDK; tokens and
registration details remain in memory for the lifetime of one client process.

## Session lifecycle and re-authentication

The client keeps one MCP transport session open from initialization through all
portfolio calls. For holdings and positions, the observed live `portfolio`
schema allows pages up to 500 records. The client requests the largest
schema-allowed page and returns the server's pagination metadata unchanged; it
does not guess how to interpret or advance that metadata. If more records are
available, call `get_holdings(from_index=..., limit=...)` or
`get_positions(from_index=..., limit=...)` for a following page, using the
offset indicated by the live response. These calls still use the same MCP
session.

The MCP protocol permits servers to expire sessions. This client reports
connection/session errors and does **not** silently initialize another MCP
session. The live schema includes a `session` tool with status and logout
modes, but this integration does not call either mode. The shared live schema
does not specify a session lifetime, so do not assume a fixed expiry. Start a
fresh test and repeat browser authorization if the connection expires or a tool
reports that authentication is required.

Each user must have a separate `ZerodhaMCPClient` instance. Do not place an
authenticated instance in a module-level singleton, shared Streamlit state,
or a LangGraph checkpoint. This phase does not connect the client to a graph
or persist portfolio snapshots.

## Install

From the project directory in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The project README currently documents Python 3.11 or 3.12. Confirm the
interpreter before creating a new environment:

```powershell
python --version
```

## Offline tests

The unit tests mock MCP tool discovery and tool results; they do not contact
Zerodha or require a browser:

```powershell
python -m unittest discover -s tests -p "test_zerodha_source.py" -v
python -m unittest discover -s tests -v
```

## Manual hosted integration test

This is the only test that contacts the hosted server and requests live account
data:

```powershell
python tests\manual_zerodha_mcp_test.py
```

The test initializes hosted OAuth, then lists the live tool names, descriptions,
and schemas. With the currently observed server it uses `portfolio` modes
`profile`, `holdings`, `positions`, and `margins`; no separate login tool is
expected. If a separate `login` tool appears in the live list, its input schema
is checked before any call. Unsupported data modes are reported as such. A
successful mocked test is not evidence that a live account has authenticated.

## Known limitations and troubleshooting

- The hosted repository's README currently documents separate `login`,
  `get_profile`, `get_holdings`, `get_positions`, and `get_margins` tools. The
  live schema shared for this deployment instead advertises `portfolio` with
  a required `mode` and optional `type`, `from`, and `limit`. Runtime
  `tools/list` is authoritative; never infer live arguments from the README.
- If the hosted OAuth page does not open, configure a default browser and rerun
  the script. It intentionally does not print transient authorization URLs.
- If the OAuth callback does not return to the test, finish the flow in that
  browser before closing the terminal.
- If a call says authentication is required, restart the manual test and
  complete the browser authorization in the new session.
- The hosted server may change its tools, schemas, auth process, or session
  policy. Tool discovery is authoritative for the running session. This phase
  does not invoke order, GTT, or other trading tools, even if they appear in
  the discovery list.
