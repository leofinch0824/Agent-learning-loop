"""A one-tool MCP stdio server -- the protocol-boundary twin of ``unit_convert``.

L3 compares TWO ways of calling the exact same function:

(a) direct: import ``unit_convert`` and call it in-process, and
(b) via MCP: spawn this file as a subprocess, do the ``initialize`` handshake,
    discover tools with ``tools/list``, then call ``tools/call`` -- every call
    crosses a process boundary as JSON-RPC.

Run it standalone (it then speaks MCP on stdin/stdout, so you normally drive it
from a client such as main.py DEMO 6 or the mcp python-sdk client):

    poetry run python lessons/l3_tools_and_execution/mcp_server.py

Probed facts about the installed ``mcp`` 2.2.0 that shape this file:

- The server class is ``mcp.server.mcpserver.MCPServer`` (the old FastMCP name
  is gone; ``mcp.server.fastmcp`` does NOT exist in 2.x).
- ``@server.tool(...)`` returns the ORIGINAL function unchanged, so the
  decorated ``unit_convert`` below is still directly importable and callable
  for path (a).
- A ``TypedDict`` return annotation makes the server publish an ``outputSchema``
  and lets the client read a typed ``structured_content`` from the result;
  with a bare ``dict`` annotation the client only gets JSON text.
- ``ToolAnnotations(read_only_hint=..., destructive_hint=..., idempotent_hint=...)
  `` is how the read/write classification travels THROUGH the protocol.
- If the tool raises, the client receives ``is_error=True`` with the generic
  text "Error executing tool unit_convert" -- the precise message only shows
  up in the server's stderr. The boundary drops error detail by design.
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

# Pure conversion table: the whole tool is side-effect free and idempotent,
# which is exactly what the retry policy in main.py keys on.
LENGTHS: dict[str, float] = {
    "m": 1.0,
    "cm": 0.01,
    "mm": 0.001,
    "km": 1000.0,
    "in": 0.0254,
    "ft": 0.3048,
}


class ConvResult(TypedDict):
    """Typed return -> output schema + structured_content on the client side."""

    value: float
    unit: str


server = MCPServer(name="l3-units", version="0.1.0")


@server.tool(
    description="Convert a length between supported units (m, cm, mm, km, in, ft).",
    annotations=ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
    ),
)
def unit_convert(value: float, from_unit: str, to_unit: str) -> ConvResult:
    """Convert a length. Raises ValueError with a precise message on unknown units.

    Watch DEMO 6 in main.py compare what happens to that message when the call
    crosses the protocol boundary: direct -> full ValueError; MCP -> generic
    "Error executing tool" text.
    """
    if from_unit not in LENGTHS:
        raise ValueError(f"unknown from_unit {from_unit!r}; supported: {sorted(LENGTHS)}")
    if to_unit not in LENGTHS:
        raise ValueError(f"unknown to_unit {to_unit!r}; supported: {sorted(LENGTHS)}")
    return {"value": value * LENGTHS[from_unit] / LENGTHS[to_unit], "unit": to_unit}


if __name__ == "__main__":
    # run_stdio_async serves the MCP protocol on stdin/stdout until the client
    # closes the connection (or the process is terminated).
    asyncio.run(server.run_stdio_async())
