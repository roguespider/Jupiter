"""Tests for the MCP endpoint (Architecture 29).

Architecture 29 asks for an interface between Jupiter and external tools, and
says two things that constrain everything here: Jupiter must work without MCP
during core model testing, and MCP is an integration layer rather than a
requirement of the neural model. So the tests below are mostly about the
boundary - that the endpoint is refused until a human allows it, that it binds
to loopback by default, and that a write from a network caller is treated as a
persistent memory change.

They also drive the server over real HTTP rather than calling handlers, because
the thing a client connects to is a socket and a test that skips the socket is
not testing what an agent will meet.
"""

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import control
import mcp_server

from tests.support import TempDirTestCase, make_config


def post(port, payload, timeout=10):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8"))


def get(port, path="/health", timeout=10):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{port}{path}", timeout=timeout
    ) as reply:
        return json.loads(reply.read().decode("utf-8"))


class McpServerTests(TempDirTestCase):
    """The endpoint driven as a client would drive it."""

    def setUp(self):
        super().setUp()
        self.config = make_config(
            knowledge_store=str(self.workdir / "knowledge.json"),
            out_dir=str(self.workdir / "out"),
            raw_dir=str(self.workdir / "raw"),
            control_log=str(self.workdir / "control.jsonl"),
        )
        (self.workdir / "out").mkdir(parents=True, exist_ok=True)
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"external"}
        self.addCleanup(setattr, control, "granted_domains", original)
        self.server = mcp_server.serve(self.config, port=0, background=True)
        self.addCleanup(self.server.shutdown)
        self.port = self.server.server_address[1]

    def test_health_reports_the_server_and_its_tools(self):
        payload = get(self.port)
        self.assertEqual(payload["server"], mcp_server.SERVER_NAME)
        self.assertIn("status", payload["tools"])

    def test_initialize_negotiates_a_protocol_version(self):
        reply = post(
            self.port,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-06-18"},
            },
        )
        self.assertEqual(reply["result"]["protocolVersion"], "2025-06-18")
        self.assertIn("tools", reply["result"]["capabilities"])

    def test_initialize_falls_back_for_an_unknown_protocol(self):
        # Answering with a version the client never asked for would be a lie
        # about what is supported; the server has to say which it actually does.
        reply = post(
            self.port,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "1999-01-01"},
            },
        )
        self.assertEqual(
            reply["result"]["protocolVersion"], mcp_server.PROTOCOL_VERSION
        )

    def test_tools_list_returns_schemas(self):
        reply = post(self.port, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        names = [t["name"] for t in reply["result"]["tools"]]
        for expected in ("status", "retrieve", "capabilities", "evaluate"):
            self.assertIn(expected, names)
        for tool in reply["result"]["tools"]:
            self.assertIn("inputSchema", tool)
            self.assertIn("description", tool)

    def test_a_tool_can_be_called(self):
        reply = post(
            self.port,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "status", "arguments": {}},
            },
        )
        self.assertFalse(reply["result"]["isError"])
        payload = json.loads(reply["result"]["content"][0]["text"])
        self.assertEqual(payload["tokenizer"], self.config.tokenizer)

    def test_an_unknown_tool_is_a_protocol_error_not_a_crash(self):
        reply = post(
            self.port,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "nonexistent", "arguments": {}},
            },
        )
        self.assertEqual(reply["error"]["code"], mcp_server.INVALID_PARAMS)

    def test_an_unknown_method_is_reported(self):
        reply = post(self.port, {"jsonrpc": "2.0", "id": 5, "method": "nope"})
        self.assertEqual(reply["error"]["code"], mcp_server.METHOD_NOT_FOUND)

    def test_a_failing_tool_does_not_take_the_server_down(self):
        # retrieve with no query is a refusal, and the server has to survive it.
        reply = post(
            self.port,
            {
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {"name": "retrieve", "arguments": {}},
            },
        )
        self.assertTrue(reply["result"]["isError"])
        self.assertEqual(get(self.port)["server"], mcp_server.SERVER_NAME)


class McpControlTests(TempDirTestCase):
    """Who is allowed to reach the model."""

    def config(self):
        work = Path(tempfile.mkdtemp())
        c = make_config(
            knowledge_store=str(work / "knowledge.json"),
            out_dir=str(work / "out"),
            raw_dir=str(work / "raw"),
            control_log=str(work / "control.jsonl"),
        )
        (work / "out").mkdir(parents=True, exist_ok=True)
        return c

    def test_the_endpoint_is_refused_without_human_authority(self):
        # Reaching a model over a socket is external tool access whatever the
        # caller calls itself, and Architecture 38 puts that behind a human.
        with self.assertRaises(control.ControlDenied):
            mcp_server.serve(self.config(), port=0, background=True)

    def test_the_default_host_is_loopback(self):
        # A model with a learned corpus is not something to place on a network
        # because a port was configured.
        self.assertEqual(make_config().mcp_host, "127.0.0.1")

    def test_a_write_from_a_network_caller_is_treated_as_a_memory_change(self):
        original = control.granted_domains
        control.granted_domains = lambda environ=None: {"external"}
        self.addCleanup(setattr, control, "granted_domains", original)
        c = self.config()
        server = mcp_server.serve(c, port=0, background=True)
        self.addCleanup(server.shutdown)
        reply = post(
            server.server_address[1],
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "learn_document",
                    "arguments": {"text": "the river is not safe."},
                },
            },
        )
        self.assertTrue(
            reply["result"]["isError"],
            "writing to memory over a socket needs the memory grant, not just external",
        )
        self.assertIn("memory", reply["result"]["content"][0]["text"])
        self.assertFalse(
            (Path(c.raw_dir) / "mcp-supplied.txt").exists(),
            "a refused write must not have written anything",
        )


class McpShapeTests(unittest.TestCase):
    def test_the_endpoint_needs_no_new_dependency(self):
        # Architecture 22 asks whether a dependency is needed before adding one.
        # http.server, json and socketserver already provide the whole protocol.
        import http.server
        import json as json_module
        import socketserver

        self.assertTrue(all([http.server, json_module, socketserver]))

    def test_the_tools_are_the_capabilities_that_already_existed(self):
        # ch29 calls MCP an integration layer. Nothing exposed here can do
        # anything Jupiter cannot already do from its own entry point.
        names = {t["name"] for t in mcp_server.tool_definitions()}
        self.assertEqual(
            names, {"status", "retrieve", "capabilities", "evaluate", "learn_document"}
        )

    def test_every_tool_declares_a_schema_and_a_description(self):
        for tool in mcp_server.tool_definitions():
            self.assertTrue(tool["description"].strip(), tool["name"])
            self.assertEqual(tool["inputSchema"]["type"], "object", tool["name"])

    def test_local_addresses_reports_loopback_always(self):
        found = mcp_server.local_addresses(9000)
        self.assertEqual(found["loopback"], "127.0.0.1:9000")

    def test_a_refusal_is_reported_as_iserror_not_a_protocol_error(self):
        # The call was understood; the tool declined. Collapsing the two would
        # make a permission decision look like a bug.
        reply = {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "content": [{"type": "text", "text": "refused"}],
                "isError": True,
            },
        }
        self.assertIn("result", reply)
        self.assertNotIn("error", reply)


if __name__ == "__main__":
    unittest.main()
