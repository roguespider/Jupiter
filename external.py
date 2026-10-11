"""Knowledge from outside the corpus, and the decision to keep it (Architecture 28).

Architecture 28 says external information may come from MCP servers, databases,
files, APIs, documentation and search systems, and then says the thing that
matters: "External knowledge should not automatically become permanent internal
knowledge. The system should evaluate whether information deserves retention."

So this is two halves, and the second is the one that keeps it safe. Anything can
be fetched; almost nothing should be kept. Jupiter already has a large corpus
and a knowledge layer that took real cycles to build, and a system that
absorbs everything it reads will fill both with whatever it happened to be shown
next. Retention therefore has to be a judgement with stated grounds, and the
default has to be no.

What is supported here, and why:

  Files and documentation are read directly, because they are what a project
  actually has. Reading its own architecture is the concrete case: Jupiter can
  consult the specification it is being built against, which is material no
  corpus contains.

  The knowledge store is queried as the database Architecture 28 names. It
  already is one, and wrapping it in a second system would be ceremony.

  An MCP client is included over HTTP, because the protocol needs no dependency
  and because an agent is the most likely source of anything Jupiter does not
  have. A client is the mirror of the server in mcp_server.py, so the two ends
  can be checked against each other.

  APIs and search systems are not implemented. Both need a dependency or a
  network policy, and neither is a capability Jupiter can be given honestly by
  guessing. They are named here so the gap is visible rather than assumed away.

Everything here that leaves the process is gated on the Architecture 38 external
domain. That gate did not exist when this module was written, and it was not
missing a call: nothing in the project ever called the functions that reach
outside, because the only production use of this module was census(), which
reports rather than retrieves. The moment fetch() became reachable from the
decision layer, the gate had to arrive with it, because an external source that
Jupiter may read on its own initiative is external tool access whether or not
anyone called it that name.
"""

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import control

FILE = "file"
DOCUMENTATION = "documentation"
DATABASE = "database"
MCP = "mcp"

SOURCE_KINDS = (FILE, DOCUMENTATION, DATABASE, MCP)

# Documents worth keeping. The floor is deliberately high: an external item
# that is not about anything Jupiter already has a term for is noise, however
# interesting it is to a reader.
RETENTION_MIN_TERMS = 2
RETENTION_MIN_SCORE = 0.34

TERM = re.compile(r"[a-z][a-z0-9_-]{2,}")
STOPWORDS = frozenset(
    """the and for that with this from have has had not are was were you your they
    their there here what when where which who whom will would should could can
    may might must been being than then them these those into over under about
    also more most other some such only own same very just each both few many
    does did done use used using new old one two three""".split()
)


def terms(text):
    """The significant words in a piece of text."""
    return {
        word
        for word in TERM.findall(str(text).lower())
        if word not in STOPWORDS and len(word) > 3
    }


def read_document(path, limit=200_000):
    """Read a text document, or None when it is not one.

    A file that cannot be decoded is skipped rather than guessed at. The
    malformed register in Architecture 27 exists because decoding something
    that is not text produces plausible nonsense, and a retriever that feeds
    that to a knowledge layer is worse than one that returns nothing.
    """
    try:
        raw = Path(path).read_bytes()[:limit]
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


class FileSource:
    """A directory of documents, or a single file."""

    def __init__(self, name, location, kind=FILE):
        self.name = name
        self.location = Path(location)
        self.kind = kind

    def documents(self, limit=200):
        if self.location.is_file():
            return [self.location]
        if not self.location.is_dir():
            return []
        found = []
        for pattern in ("*.md", "*.txt", "*.rst"):
            found.extend(sorted(self.location.rglob(pattern)))
        return found[:limit]

    def fetch(self, query, limit=5):
        wanted = terms(query)
        if not wanted:
            return []
        scored = []
        for path in self.documents():
            text = read_document(path)
            if text is None:
                continue
            found = terms(text)
            overlap = wanted & found
            if not overlap:
                continue
            scored.append(
                {
                    "source": self.name,
                    "kind": self.kind,
                    "reference": str(path),
                    "score": len(overlap) / len(wanted),
                    "matched": sorted(overlap),
                    "excerpt": _excerpt(text, overlap),
                }
            )
        scored.sort(key=lambda row: -row["score"])
        return scored[:limit]


def _excerpt(text, overlap, width=320):
    """A passage around the first matching term, so a result can be judged."""
    lowered = text.lower()
    for word in sorted(overlap):
        at = lowered.find(word)
        if at >= 0:
            start = max(0, at - width // 2)
            return text[start : start + width].strip()
    return text[:width].strip()


class McpClient:
    """A minimal MCP client over HTTP.

    The mirror of mcp_server.py, and deliberately no larger than it. It speaks
    the same JSON-RPC the server does, so the two can be pointed at each other
    and the round trip checked rather than assumed.
    """

    def __init__(self, name, endpoint, timeout=15):
        self.name = name
        self.endpoint = endpoint
        self.timeout = timeout
        self.tools = None
        # census() reports every source by kind, and a client built here has no
        # kind to report. Anything configured as an MCP source therefore raised
        # AttributeError inside the census rather than describing itself, which
        # is the one thing a census is for.
        self.kind = MCP

    def _rpc(self, method, params=None, request_id=1):
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
        }
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as reply:
            return json.loads(reply.read().decode("utf-8"))

    def available(self):
        """Whether the endpoint answers, and what it offers.

        Architecture 38 asks for human control over external tool access, and
        opening a socket to an agent Jupiter did not start is that, so the gate
        is asked before the connection rather than after it. A refusal is raised
        rather than folded into the False this returns for an unreachable
        endpoint: those two mean opposite things to a reader, and reporting a
        grant that was never given as a server that is down would send whoever
        reads the report looking in the wrong place.
        """
        control.check(
            "external",
            "probe an external MCP agent",
            detail=self.endpoint,
        )
        try:
            self.tools = self._rpc("tools/list")["result"]["tools"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
            self.tools = None
        return self.tools is not None

    def call(self, name, arguments):
        reply = self._rpc("tools/call", {"name": name, "arguments": arguments})
        result = reply.get("result", {})
        if result.get("isError"):
            raise RuntimeError(
                (result.get("content") or [{}])[0].get("text", "tool refused")
            )
        return result

    def fetch(self, query, limit=5):
        """Query an external agent, if it offers a tool that can answer."""
        if self.tools is None and not self.available():
            return []
        names = [t.get("name") for t in self.tools]
        tool = next(
            (n for n in names if n in ("retrieve", "search", "query", "recall")), None
        )
        if tool is None:
            return []
        try:
            payload = self.call(tool, {"query": query, "limit": limit})
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
            return []
        return [{"source": self.name, "kind": MCP, "tool": tool, "payload": payload}]


def sources(c):
    """The external sources a configuration names, ready to fetch from."""
    found = []
    for entry in getattr(c, "external_sources", []) or []:
        if isinstance(entry, str):
            found.append(FileSource(Path(entry).stem, entry))
        elif isinstance(entry, dict):
            kind = entry.get("kind", FILE)
            if kind == MCP:
                found.append(
                    McpClient(
                        entry.get("name", "mcp"),
                        entry.get("endpoint", "http://127.0.0.1:8765/"),
                    )
                )
            else:
                found.append(
                    FileSource(
                        entry.get("name", Path(entry.get("location", ".")).stem),
                        entry.get("location", "."),
                        kind=kind,
                    )
                )
    return found


def fetch(c, query, limit=5):
    """Ask every configured source, and return what any of them had.

    This is the one function that brings material from outside the process into
    Jupiter, so it carries the Architecture 38 gate rather than leaving it to
    each source. A refusal propagates: a caller that was told to consult the
    world and could not has to be able to say so, and returning an empty list
    instead would be indistinguishable from a world with nothing to say.
    """
    control.check(
        "external",
        "consult external knowledge sources",
        detail=f"query={query!r} limit={limit}",
        c=c,
    )
    results = []
    for source in sources(c):
        try:
            results.extend(source.fetch(query, limit=limit))
        except control.ControlDenied:
            # ControlDenied is a PermissionError, so it is an OSError, so the
            # handler below would swallow it and report an empty result. That is
            # the specific dishonesty this module exists to avoid: a refusal
            # would be indistinguishable from a world with nothing to say.
            raise
        except (urllib.error.URLError, OSError, ValueError, RuntimeError):
            # One unreachable source is not a reason to fail the question. It is
            # recorded as absent rather than as an error the caller cannot act on.
            continue
    results.sort(key=lambda row: -row.get("score", 0.0))
    return results[:limit]


def worth_keeping(result, knowledge_terms, existing=None):
    """Whether an external result deserves to become internal knowledge.

    Architecture 28's test, and it is deliberately strict. The result must
    overlap material Jupiter already holds, and not already be held. An item
    that is interesting but unrelated is a thing to read, not a thing to
    become part of the model, and a system that cannot say no to those fills
    its knowledge layer with whatever it was shown.
    """
    found = set(result.get("matched", []))
    if not found:
        return False, "no matching terms were recorded, so there is nothing to judge"
    overlap = found & set(knowledge_terms)
    if len(overlap) < RETENTION_MIN_TERMS:
        return (
            False,
            f"only {len(overlap)} of its terms appear in what Jupiter already holds",
        )
    score = float(result.get("score", 0.0))
    if score < RETENTION_MIN_SCORE:
        return False, f"score {score:.3f} is below the retention floor"
    if existing and result.get("reference") in existing:
        return False, "already held"
    return (
        True,
        f"{len(overlap)} shared terms and a score of {score:.3f}; "
        "retained pending the acceptance gate",
    )


def retain(store, result, record):
    """File an external result as material to be learned, not as knowledge.

    The distinction matters. Writing it into the knowledge store now would make
    it permanent before any cycle has judged it, which is precisely what
    Architecture 28 forbids. It goes into the raw corpus, where curation and the
    acceptance gate decide whether it is worth anything.
    """
    from pathlib import Path

    directory = Path(store.path).parent / "external"
    directory.mkdir(parents=True, exist_ok=True)
    name = (
        str(result.get("reference", result.get("source", "external")))
        .replace("/", "_")
        .replace("\\", "_")
    )
    target = directory / f"{name}.txt"
    stamp = time.strftime("%Y%m%dT%H%M%S")
    counter = 0
    while target.exists():
        counter += 1
        target = directory / f"{name}-{stamp}-{counter}.txt"
    target.write_text(
        f"Source: {result.get('reference', result.get('source'))}\n"
        f"Retained because: {record}\n\n{result.get('excerpt', '')}\n",
        encoding="utf-8",
    )
    return target


def census(c):
    """Which external sources are configured and reachable."""
    rows = []
    for source in sources(c):
        row = {"name": source.name, "kind": source.kind, "reachable": True}
        if isinstance(source, McpClient):
            row["endpoint"] = source.endpoint
            try:
                row["reachable"] = source.available()
            except control.ControlDenied:
                # Reaching another agent is external tool access, so without the
                # grant it is not attempted. That is reported as unknown rather
                # than as False, because "never asked" and "asked and it did not
                # answer" are different facts and the report is read by someone
                # deciding whether to grant the domain.
                row["reachable"] = None
                row["probe"] = (
                    "not attempted: external access is not granted, so this is "
                    f"unknown rather than false. Set {control.GRANT_VARIABLE} "
                    "to include 'external' to let Jupiter probe it."
                )
        else:
            row["location"] = str(source.location)
            row["documents"] = len(source.documents())
        rows.append(row)
    return {
        "sources": rows,
        "kinds": list(SOURCE_KINDS),
        "note": (
            "Files, documentation and the knowledge store are read directly; "
            "an MCP agent is spoken to over HTTP. APIs and search systems are "
            "not implemented and need a dependency or a network policy. "
            "Nothing here becomes internal knowledge without passing the "
            "retention test, and even then it is filed for the acceptance gate "
            "rather than written into the knowledge layer."
        ),
    }
