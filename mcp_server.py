"""MCP integration: an endpoint an agent can reach once Jupiter has a port.

Architecture 29 lists what MCP should provide - retrieving information, reading
files, writing approved information, querying databases, interacting with tools,
accessing RoBoT services - and says two things that shape everything here.
Jupiter must remain capable of operating without MCP during core model testing,
and MCP is an integration layer rather than a fundamental requirement for the
neural model. So this is a separate module that the default chain does not run,
and the model, the learning loop and the tokenizer are untouched by it.

The transport is HTTP with JSON-RPC 2.0, which is what "an IP address and a
port" means, and it is built on the standard library alone. Architecture 22 asks
whether a dependency is needed before one is added, and http.server, json and
socketserver already provide everything the protocol requires, so adding a
framework would have been a large dependency for no capability.

Two defaults are deliberate and both are about who can reach the model.

The listener binds to the loopback address, not to every interface. A model
with a learned corpus and a knowledge store is not something to put on a network
because a port happened to be configured; binding everywhere is one word of
mistake away. Someone who wants it reachable sets the host.

And the endpoint is behind the Architecture 38 external-access control, for the
same reason. Reaching a model over a socket is external tool access whatever the
caller calls itself, and the control layer already knows how to refuse something
until a human says otherwise.

The tools are the capabilities that already exist rather than new ones. Nothing
here can do anything Jupiter cannot already do from its own entry point, which
is what keeps ch 29's "integration layer" honest.
"""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import control

PROTOCOL_VERSION = "2025-06-18"
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")

SERVER_NAME = "jupiter"
SERVER_VERSION = "0.1"

# JSON-RPC error codes. The specification's own numbers, so a client can tell a
# refused call from a broken one.
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def _text(text):
    """The content shape MCP uses for a text result."""
    return [{"type": "text", "text": str(text)}]


class ToolError(Exception):
    """A tool refused or could not answer. Carried to the caller as isError."""


class JupiterTools:
    """The capabilities exposed, each delegating to code that already exists.

    Read tools are available once the endpoint is running. Tools that change
    anything require the memory grant, because writing to the knowledge layer
    from a network caller is a persistent memory change and Architecture 38
    puts that behind a human.
    """

    def __init__(self, c):
        self.config = c

    # -- read -------------------------------------------------------------

    def status(self, arguments):
        """What is loaded, and what the model currently is."""
        from config import ARCHITECTURE_VERSION

        return {
            "server": SERVER_NAME,
            "architecture_version": ARCHITECTURE_VERSION,
            "tokenizer": getattr(self.config, "tokenizer", "unknown"),
            "block_size": getattr(self.config, "block_size", None),
            "out_dir": getattr(self.config, "out_dir", "out"),
            "checkpoint_present": _checkpoint_exists(self.config),
            "policy": (
                "Jupiter answers from what it has learned. It is a research "
                "instrument, not a general assistant."
            ),
        }

    def retrieve(self, arguments):
        """Search the knowledge store for material relevant to a query."""
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ToolError("retrieve needs a 'query'")
        from pathlib import Path

        import store as store_module

        import retrieval

        path = Path(self.config.knowledge_store)
        if not path.exists():
            return {"results": [], "note": "no knowledge store yet"}
        knowledge = store_module.Store(path)
        results = retrieval.retrieve(
            knowledge, query, limit=int(arguments.get("limit", 5))
        )
        return {
            "results": [
                {
                    "collection": item["collection"],
                    "id": item["id"],
                    "score": item["score"],
                    "excerpt": item.get("excerpt", ""),
                    "matched": item.get("matched", []),
                }
                for item in results
            ],
            "note": "matching is lexical, so a paraphrase sharing no vocabulary is not found",
        }

    def evaluate(self, arguments):
        """The Architecture 34 report: what is measured and what is not."""
        import contextlib
        import io

        import evaluation

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = evaluation.report(self.config)
        return {
            "measured": list(evaluation.MEASURED),
            "coverage": dict(evaluation.COVERAGE),
            "report": result,
        }

    def capabilities(self, arguments):
        """What Architecture 34 says can and cannot be measured, in one line."""
        import evaluation

        lines = [
            f"{name}: {'measured' if name in evaluation.MEASURED else 'not measured'}"
            for name in evaluation.COVERAGE
        ]
        return {"capabilities": lines}

    # -- write ------------------------------------------------------------

    def learn_document(self, arguments):
        """Queue a document for learning. Requires the memory grant.

        This does not learn synchronously. A cycle takes a bounded amount of
        time and is judged before anything is kept, so a network caller queues
        material and the governed controller picks it up. Accepting a document
        from a socket is a persistent memory change, which Architecture 38 puts
        behind a human, so it is refused without the memory grant.
        """
        source = str(arguments.get("path", "")).strip()
        text = str(arguments.get("text", ""))
        if not source and not text:
            raise ToolError("learn_document needs a 'path' or 'text'")
        control.check(
            "memory",
            "accept a document into the corpus from a network caller",
            detail=source or "inline text",
            c=self.config,
        )
        from pathlib import Path

        target = (
            Path(source) if source else Path(self.config.raw_dir) / "mcp-supplied.txt"
        )
        # The path here comes from the caller, so the memory grant above is not
        # enough on its own: without this, a network caller holding the memory
        # grant could name any path on the machine and have Jupiter write to it.
        # A document belongs in the corpus, and the corpus is inside data/.
        control.check_write(
            self.config,
            target,
        )
        if text:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        return {
            "queued": str(target),
            "note": (
                "written into data/raw. It is curated like any other document "
                "and judged by the acceptance gate before it is kept."
            ),
        }


def _checkpoint_exists(c):
    from pathlib import Path

    return (Path(c.out_dir) / "Jupiter_latest.safetensors").exists()


def tool_definitions():
    """The tool schemas, in the shape tools/list expects."""
    string = {"type": "string"}
    return [
        {
            "name": "status",
            "description": "What model is loaded, which tokenizer, and whether a checkpoint exists.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "retrieve",
            "description": (
                "Search Jupiter's learned knowledge for material relevant to a "
                "query. Matching is lexical, so results say which terms matched."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to look for"},
                    "limit": {"type": "integer", "description": "maximum results"},
                },
                "required": ["query"],
            },
        },
        {
            "name": "capabilities",
            "description": "Which Architecture 34 categories Jupiter can and cannot measure.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "evaluate",
            "description": "The full evaluation report: metrics per category.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "learn_document",
            "description": (
                "Queue a document for learning from its path or inline text. "
                "Refused unless a human has authorised memory changes."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {**string, "description": "a file to learn"},
                    "text": {**string, "description": "text to learn directly"},
                },
            },
        },
    ]


class McpHandler(BaseHTTPRequestHandler):
    """One request. Kept deliberately small: it speaks JSON-RPC and nothing else."""

    protocol_version = "HTTP/1.1"
    server_version = f"{SERVER_NAME}/{SERVER_VERSION}"
    tools = None

    def log_message(self, format, *args):
        """Stay quiet. The console is the learning report, and Architecture 35
        asks for the run to be observable, which is not the same as narrating
        every packet."""

    def _respond(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, request_id, code, message):
        self._respond(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {"code": code, "message": message},
            }
        )

    def _result(self, request_id, result):
        self._respond({"jsonrpc": "2.0", "id": request_id, "result": result})

    def do_GET(self):
        # A GET is how a human checks the port is alive, which is the first
        # thing anyone does after starting the server.
        if self.path.rstrip("/") in ("/health", ""):
            self._respond(
                {
                    "server": SERVER_NAME,
                    "version": SERVER_VERSION,
                    "protocol": PROTOCOL_VERSION,
                    "tools": [t["name"] for t in tool_definitions()],
                }
            )
            return
        self._error(None, METHOD_NOT_FOUND, f"no such path: {self.path}")

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
            request = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, TypeError) as failure:
            self._error(None, INVALID_REQUEST, f"malformed request: {failure}")
            return
        self._dispatch(request)

    def _dispatch(self, request):
        if not isinstance(request, dict):
            self._error(None, INVALID_REQUEST, "request must be an object")
            return
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}
        if not isinstance(params, dict):
            self._error(request_id, INVALID_PARAMS, "params must be an object")
            return

        if method == "initialize":
            requested = str(params.get("protocolVersion", PROTOCOL_VERSION))
            version = (
                requested if requested in SUPPORTED_PROTOCOLS else PROTOCOL_VERSION
            )
            self._result(
                request_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            )
            return
        if method in ("notifications/initialized", "initialized"):
            # A notification carries no id and expects no reply.
            self._respond({}, status=202)
            return
        if method == "tools/list":
            self._result(request_id, {"tools": tool_definitions()})
            return
        if method == "tools/call":
            self._call(request_id, params)
            return
        if method == "ping":
            self._result(request_id, {})
            return
        self._error(request_id, METHOD_NOT_FOUND, f"no such method: {method}")

    def _call(self, request_id, params):
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            self._error(request_id, INVALID_PARAMS, "arguments must be an object")
            return
        handler = getattr(self.tools, str(name), None)
        if handler is None or str(name).startswith("_"):
            self._error(request_id, INVALID_PARAMS, f"no such tool: {name}")
            return
        try:
            value = handler(arguments)
        except ToolError as failure:
            # A refused call is a result with isError, not a protocol error. The
            # call was understood; the tool declined.
            self._result(request_id, {"content": _text(str(failure)), "isError": True})
            return
        except control.ControlDenied as failure:
            self._result(request_id, {"content": _text(str(failure)), "isError": True})
            return
        except Exception as failure:  # a tool must not take the server down
            # INTERNAL_ERROR is the JSON-RPC code for a failure the server could
            # not have anticipated, which is exactly what an exception escaping a
            # tool is. The response is still a result carrying isError rather
            # than a protocol error, because the request was understood and
            # dispatched; what failed is the work, not the call. The code is in
            # the result so a client can tell an unexpected failure from a tool
            # that deliberately declined, which are different problems with
            # different responses.
            self._result(
                request_id,
                {
                    "content": _text(f"{type(failure).__name__}: {failure}"),
                    "isError": True,
                    "code": INTERNAL_ERROR,
                },
            )
            return
        self._result(
            request_id,
            {
                "content": _text(json.dumps(value, indent=2, default=str)),
                "isError": False,
            },
        )


def local_addresses(port):
    """The addresses this machine can be reached on.

    An agent connecting over a network needs an address, and asking the machine
    for its own is more reliable than a person reading it off a log.
    """
    found = {"loopback": f"127.0.0.1:{port}"}
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            found["lan"] = f"{probe.getsockname()[0]}:{port}"
    except OSError:
        # No route to a public address means no LAN address to advertise, which
        # is a normal state on an isolated machine and not a failure.
        found["lan"] = None
    return found


def serve(c, host=None, port=None, background=True):
    """Start the endpoint.

    Refused without the external-access grant, because reaching a model over a
    socket is external tool access whatever the caller calls itself.
    """
    host = host or getattr(c, "mcp_host", "127.0.0.1")
    port = int(port if port is not None else getattr(c, "mcp_port", 8765))
    control.check(
        "external",
        "expose the model over a network socket",
        detail=f"{host}:{port}",
        c=c,
    )
    McpHandler.tools = JupiterTools(c)
    server = ThreadingHTTPServer((host, port), McpHandler)
    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
    # When not running in the background the caller owns the serve loop, so the
    # server is returned unserved. serve() used to call serve_forever() itself
    # in that case, which meant main() blocked inside this function and never
    # reached the banner below: the endpoint worked and the operator was told
    # nothing, so there was no address to connect to. main() now starts the loop
    # itself, after it has printed.
    return server


def main(c=None, host=None, port=None):
    from config import load_config

    c = c or load_config()
    port = int(port if port is not None else getattr(c, "mcp_port", 8765))
    try:
        server = serve(c, host=host, port=port, background=False)
    except control.ControlDenied as failure:
        print(f"Jupiter did not start the MCP endpoint.\n\n{failure}")
        return 1
    addresses = local_addresses(port)
    print("Jupiter MCP endpoint")
    print(f"  protocol : {PROTOCOL_VERSION}")
    print(f"  loopback : {addresses['loopback']}")
    print(f"  network  : {addresses['lan'] or 'unavailable on this machine'}")
    print(f"  tools    : {', '.join(t['name'] for t in tool_definitions())}")
    print()
    print("  An agent connects to the network address. It is the loopback")
    print("  address unless a human set mcp_host to something wider.")
    try:
        # Started here rather than inside serve(), so the address is on screen
        # before anything blocks.
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
