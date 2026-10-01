"""MCP stdio connector lifecycle and tool dispatch for the ReAct agent."""
from __future__ import annotations

import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[2]


def _server_configs() -> dict[str, dict[str, Any]]:
    """Build local MCP server commands using the project root as their base."""
    docs_dir = ROOT / "data" / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    memory_file = ROOT / "data" / "agent_memory.jsonl"
    return {
        "filesystem": {
            "command": "npx",
            "args": ["-y", "--silent", "@modelcontextprotocol/server-filesystem", str(docs_dir)],
        },
        "fetch": {"command": "uvx", "args": ["mcp-server-fetch"]},
        "git": {
            "command": "uvx",
            "args": ["mcp-server-git", "--repository", str(ROOT)],
        },
        "memory": {
            "command": "npx",
            "args": ["-y", "--silent", "@modelcontextprotocol/server-memory"],
            "env": {"MEMORY_FILE_PATH": str(memory_file)},
        },
    }


def _is_read_only_tool(server_name: str, tool_name: str) -> bool:
    """Keep external connectors read-only except for deliberate memory writes."""
    if server_name == "filesystem":
        return tool_name in {
            "list_allowed_directories", "list_directory", "directory_tree",
            "read_file", "read_text_file", "read_multiple_files",
            "get_file_info", "search_files",
        }
    if server_name == "fetch":
        return tool_name == "fetch"
    if server_name == "git":
        return tool_name in {
            "git_status", "git_diff_unstaged", "git_diff_staged", "git_diff",
            "git_log", "git_show", "git_branch",
        }
    if server_name == "memory":
        return tool_name in {
            "create_entities", "create_relations", "add_observations",
            "read_graph", "search_nodes", "open_nodes",
        }
    return False


class MCPToolRegistry:
    """Connect four stdio MCP servers and expose their read-focused tools."""

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, tuple[str, Any]] = {}

    async def __aenter__(self) -> "MCPToolRegistry":
        try:
            for server_name, config in _server_configs().items():
                params = StdioServerParameters(
                    command=config["command"],
                    args=config["args"],
                    env=config.get("env"),
                    cwd=str(ROOT),
                )
                read_stream, write_stream = await self._stack.enter_async_context(
                    stdio_client(params)
                )
                session = await self._stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                self.sessions[server_name] = session
                listed = await session.list_tools()
                for tool in listed.tools:
                    if _is_read_only_tool(server_name, tool.name):
                        key = f"{server_name}.{tool.name}"
                        self.tools[key] = (server_name, tool)
            return self
        except BaseException:
            await self._stack.aclose()
            raise

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self._stack.aclose()

    def prompt_description(self) -> str:
        descriptions = []
        for key, (_, tool) in self.tools.items():
            schema = json.dumps(tool.inputSchema, ensure_ascii=False)
            descriptions.append(f"- {key}: {tool.description or ''} Input schema: {schema}")
        return "\n".join(descriptions)

    async def call(self, name: str, raw_input: str) -> str:
        if name not in self.tools:
            return f"Error: MCP tool '{name}' is unavailable."
        server_name, tool = self.tools[name]
        try:
            arguments = json.loads(raw_input)
        except json.JSONDecodeError:
            # Keep the simple ReAct format usable when a model emits a plain string.
            required = tool.inputSchema.get("required", [])
            properties = tool.inputSchema.get("properties", {})
            field = required[0] if required else next(iter(properties), None)
            if not field:
                arguments = {}
            else:
                arguments = {field: raw_input.strip().strip('"')}
        if not isinstance(arguments, dict):
            return "Error: MCP Action Input must be a JSON object."

        try:
            result = await self.sessions[server_name].call_tool(tool.name, arguments)
            parts = [getattr(item, "text", "") for item in result.content]
            output = "\n".join(part for part in parts if part)
            if not output and result.structuredContent is not None:
                output = json.dumps(result.structuredContent, ensure_ascii=False)
            if result.isError:
                return f"MCP tool error: {output or 'unknown error'}"
            return output[:12000] or "Tool completed without text output."
        except Exception as exc:  # Return tool errors to the model as observations.
            return f"MCP tool error: {type(exc).__name__}: {exc}"
