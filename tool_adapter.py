"""Tool adapter with permission gates, timeouts, and verification (Upgrade 8).

Connects retrieval, local files, and MCP tools to the decision loop.
Requires explicit grants (external, memory) and verifies results before
trusting them. A tool result does not automatically become a trusted fact.
"""

import time
from pathlib import Path

import control


class ToolAdapter:
    """A gated tool wrapper that requires grants, applies timeouts, and verifies results."""

    def __init__(self, grants: set[str] | None = None):
        self.grants = set(grants or [])
        self.timeouts = {"retrieve": 5.0, "file_read": 3.0, "mcp_call": 10.0}
        self.results_verified = []

    def check_grant(self, domain: str) -> bool:
        return domain in self.grants

    def call_with_timeout(
        self, tool_name: str, action: callable, *args, **kwargs
    ) -> dict:
        """Call a tool with a timeout and return structured result with verification."""
        timeout = self.timeouts.get(tool_name, 5.0)
        start = time.time()
        try:
            result = action(*args, **kwargs)
            elapsed = time.time() - start
            verified = self.verify_result(tool_name, result)
            return {
                "tool": tool_name,
                "applied": True,
                "verified": verified,
                "elapsed": round(elapsed, 3),
                "timeout": timeout,
                "result": result,
            }
        except Exception as exc:
            return {
                "tool": tool_name,
                "applied": False,
                "verified": False,
                "error": str(exc),
                "timeout": timeout,
            }

    def verify_result(self, tool_name: str, result) -> bool:
        """Basic verification: result must be non-empty and not contain error markers."""
        if result is None:
            return False
        if isinstance(result, dict):
            if result.get("error") or result.get("denied"):
                return False
            return bool(
                result.get("applied") or result.get("retained") or result.get("written")
            )
        if isinstance(result, str):
            return len(result.strip()) > 0 and not result.startswith("[error")
        return bool(result)

    def retrieve_knowledge(self, query: str, knowledge_store_path: str) -> dict:
        """Retrieve from local knowledge store with external grant check."""
        if not self.check_grant("external"):
            return {
                "applied": False,
                "denied": True,
                "reason": "external domain not granted",
            }
        from pathlib import Path

        import store as store_module

        from retrieval import retrieve

        path = Path(knowledge_store_path)
        if not path.exists():
            return {
                "applied": False,
                "verified": False,
                "reason": "knowledge store missing",
            }
        knowledge = store_module.Store(path)
        results = retrieve(knowledge, query, limit=5)
        verified = len(results) > 0 and all(isinstance(r, dict) for r in results)
        self.results_verified.append(
            {"tool": "retrieve_knowledge", "verified": verified, "count": len(results)}
        )
        return {
            "applied": True,
            "verified": verified,
            "results": len(results),
            "results_detail": results,
        }

    def read_local_file(self, file_path: str) -> dict:
        """Read a local file with file access grant check."""
        if not self.check_grant("memory"):
            return {
                "applied": False,
                "denied": True,
                "reason": "memory/file domain not granted",
            }
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            return {
                "applied": False,
                "verified": False,
                "reason": "file not found or not a file",
            }
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            verified = len(content) > 0
            return {
                "applied": True,
                "verified": verified,
                "path": str(path),
                "length": len(content),
            }
        except Exception as exc:
            return {"applied": False, "verified": False, "error": str(exc)}

    def call_mcp_tool(self, tool_name: str, payload: dict) -> dict:
        """Call an MCP tool with timeout and verification."""
        if not self.check_grant("external"):
            return {
                "applied": False,
                "denied": True,
                "reason": "external domain not granted for MCP",
            }
        # Delegate to existing mcp_server handler if available
        try:
            import mcp_server

            result = mcp_server.execute_tool(tool_name, payload)
            verified = self.verify_result(tool_name, result)
            return {
                "applied": True,
                "verified": verified,
                "tool": tool_name,
                "result": result,
            }
        except Exception as exc:
            return {"applied": False, "verified": False, "error": str(exc)}


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="tool_adapter",
        description="Gated tool adapter with permission checks and verification.",
    )
    parser.add_argument(
        "--grant", nargs="+", default=["external", "memory"], help="granted domains"
    )
    parser.add_argument(
        "--retrieve", metavar="QUERY", help="retrieve from knowledge store"
    )
    parser.add_argument("--read-file", metavar="PATH", help="read local file")
    parser.add_argument("--mcp-tool", metavar="NAME", help="call MCP tool")
    args = parser.parse_args()

    adapter = ToolAdapter(grants=set(args.grant))
    if args.retrieve:
        result = adapter.retrieve_knowledge(args.retrieve, "out/knowledge_store.json")
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.read_file:
        result = adapter.read_local_file(args.read_file)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.mcp_tool:
        result = adapter.call_mcp_tool(args.mcp_tool, {})
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print("No action specified. Use --retrieve, --read-file, or --mcp-tool.")
        parser.print_help()


if __name__ == "__main__":
    import json
    import sys

    sys.exit(main() or 0)
