"""Programming knowledge as structure, not as memorized text.

Twelve programming capabilities were asked for, and one thing about them
governs all of them: "Programming knowledge should be treated as structured
knowledge rather than simply memorized text." Everything else follows from that.
Without it a program is bytes, and the only thing Jupiter can say about it is
what it has seen; it cannot say what a function is, what it calls, or which file
a name came from.

So this module reads code into records. A symbol, a dependency, a diagnostic and
a language are things with fields, and they live in the knowledge store beside
concepts and assertions rather than in the training stream.

Honesty about what this is and is not:

  Python is parsed with the standard library's own ast module, so a Python symbol
  is real structure with a real line number and real parameter names. That is
  not a heuristic and is not labelled as one.

  Other languages are read with narrow syntax patterns and are labelled
  heuristics everywhere they are used, for the same reason the concept extractor
  is. The patterns are deliberately conservative: a wrong symbol is a fabricated
  fact about a program, and a fabricated fact about code is worse than an
  absent one.

  Nothing here executes code, infers behaviour from a body, or proves a program
  correct. It says what is written, not what it does. The
  semantics, debugging and code generation are not achieved by this and are not
  claimed to be.

What it does buy is concrete: cross-language reasoning becomes possible because
"a function that opens a file" is a record, so a Python one and a JavaScript one
can be recognised as the same kind of thing. That is the groundwork, and the
groundwork is what is built here.
"""

import ast
import re
import time

SYMBOLS = "code_symbols"
DEPENDENCIES = "code_dependencies"
DIAGNOSTICS = "code_diagnostics"

# The kinds of definition Jupiter recognises. Checked against when a symbol is
# recorded, so that adding one here makes it reachable rather than leaving a
# declaration that nothing reads.
DEFINITION_KINDS = ("function", "method", "class", "constant", "variable")
LANGUAGES = "code_languages"

# Heuristic language detection. Ordered, and the first match wins, so a file
# that looks like more than one language is reported as the first rather than
# being double counted.
LANGUAGE_SIGNS = (
    (
        "python",
        (
            # The declaration patterns deliberately do not require a trailing
            # colon. A file with a syntax error in it still declares functions,
            # and failing to recognise it as Python means the error is never
            # reported, which is exactly when it would matter most.
            r"^\s*(async\s+)?def\s+\w+\s*\(",
            r"^\s*class\s+\w+\s*[\(:]",
            r"^\s*import\s+\w+",
            r"^\s*from\s+[\w.]+\s+import\b",
            r"^\s*@\w+",
        ),
    ),
    (
        "javascript",
        (r"function\s+\w+\s*\(", r"=>\s*\{", r"\b(const|let|var)\s+\w+\s*="),
    ),
    ("typescript", (r":\s*(string|number|boolean)\b", r"interface\s+\w+\s*\{")),
    ("c", (r"^\s*#include\s*<", r"^\s*int\s+\w+\s*\([^)]*\)\s*\{", r"printf\(")),
    ("cpp", (r"^\s*#include\s*<", r"\bstd::", r"\bcout\b")),
    (
        "java",
        (r"\bpublic\s+(class|static|void)\b", r"^\s*import\s+java\.", r"System\.out"),
    ),
    ("rust", (r"\bfn\s+\w+\s*\(", r"\blet\s+mut\b", r"\bprintln!")),
    ("go", (r"\bfunc\s+\w+\s*\(", r"\bpackage\s+\w+", r":=")),
    (
        "sql",
        (
            r"^\s*SELECT\s+.*\s+FROM\b",
            r"^\s*(INSERT|UPDATE|DELETE)\s+INTO\b",
            r"\bJOIN\b",
        ),
    ),
    ("lisp", (r"^\s*\((defun|def|let)\s", r"^\s*\(car\s|^\s*\(cdr\s")),
    ("shell", (r"^\s*#!.*\b(bash|sh)\b", r"^\s*(if|for|while)\s.*;\s*then\b")),
    ("html", (r"<html", r"<!DOCTYPE\s+html", r"<div\b")),
    ("json", (r"^\s*[\[{]", r"^\s*\"[\w-]+\"\s*:")),
)

# Diagnostic text. Error and warning interpretation are wanted;
# this recognises the shapes they arrive in, which is the part that can be done
# without a compiler, and says nothing about the underlying fault.
DIAGNOSTIC_PATTERNS = (
    (
        "error",
        re.compile(r"\b(error|exception|traceback|failed|failure|fatal)\b", re.I),
    ),
    ("warning", re.compile(r"\b(warning|warn|deprecated|caution)\b", re.I)),
    ("hint", re.compile(r"\b(hint|note|consider|try)\b", re.I)),
)


def detect_language(text):
    """Which language a document is written in, by the first sign that matches.

    Returns None rather than "unknown" being passed off as a finding. A document
    that matches nothing has not been identified, and saying so is different
    from saying it is not a program.
    """
    sample = "\n".join(text.splitlines()[:400])
    for name, signs in LANGUAGE_SIGNS:
        for sign in signs:
            if re.search(sign, sample, re.MULTILINE):
                return name
    return None


def parse_python(text):
    """Python symbols and imports, from the standard library's own parser.

    A syntax error is reported rather than raised. A corpus of code contains
    fragments that do not parse, and one unparseable file must not stop the rest
    being read.
    """
    symbols = []
    imports = []
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError) as failure:
        return symbols, imports, {"kind": "error", "detail": str(failure)}
    _collect_python(tree, symbols, imports, inside_class=False)
    return symbols, imports, None


def _collect_python(node, symbols, imports, inside_class):
    """Walk the tree carrying whether we are inside a class body.

    ast gives no parent links, so whether a function is a method is decided by
    the walk rather than looked up afterwards. Doing it here is what makes
    "method" an honest label rather than a guess.
    """
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            symbols.append(
                {
                    "name": child.name,
                    "kind": "class",
                    "line": getattr(child, "lineno", None),
                    "params": [],
                }
            )
            # Recursed into here and not again at the bottom of the loop. Doing
            # both recorded every method a second time as a free function, which
            # is how a symbol table ends up claiming a class has two copies of
            # its own methods.
            _collect_python(child, symbols, imports, inside_class=True)
            continue
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "name": child.name,
                    "kind": "method" if inside_class else "function",
                    "line": getattr(child, "lineno", None),
                    "params": [a.arg for a in child.args.args],
                }
            )
        elif isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    symbols.append(
                        {
                            "name": target.id,
                            "kind": "constant",
                            "line": getattr(child, "lineno", None),
                            "params": [],
                        }
                    )
        elif isinstance(child, ast.Import):
            imports.extend(alias.name for alias in child.names)
        elif isinstance(child, ast.ImportFrom):
            if child.module:
                imports.append(child.module)
        _collect_python(child, symbols, imports, inside_class)


def parse_generic(text, language):
    """Symbols and imports for a language with no parser available.

    Narrow on purpose. These patterns recognise declarations, not behaviour, and
    a false positive here becomes a fabricated symbol.
    """
    symbols = []
    imports = []
    patterns = {
        "javascript": (
            (r"^\s*(?:export\s+)?function\s+(\w+)\s*\(([^)]*)\)", "function"),
            (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s*)?\(", "function"),
            (r"^\s*class\s+(\w+)", "class"),
        ),
        "java": (
            (
                r"(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(([^)]*)\)",
                "method",
            ),
            (r"(?:public\s+)?class\s+(\w+)", "class"),
        ),
        "c": ((r"^[\w\s\*]+\s+\*?(\w+)\s*\([^;{]*\)\s*\{", "function"),),
        "cpp": ((r"^[\w\s\*:<>,]+\s+\*?(\w+)\s*\([^;{]*\)\s*\{", "function"),),
        "go": ((r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\(([^)]*)\)", "function"),),
        "rust": ((r"^\s*(?:pub\s+)?fn\s+(\w+)\s*\(([^)]*)\)", "function"),),
    }
    for pattern, kind in patterns.get(language, ()):
        for match in re.finditer(pattern, text, re.MULTILINE):
            symbols.append(
                {
                    "name": match.group(1),
                    "kind": kind,
                    "line": text.count("\n", 0, match.start()) + 1,
                    "params": _params(
                        match.group(2)
                        if match.lastindex and match.lastindex >= 2
                        else ""
                    ),
                }
            )
    for name in re.findall(r"^\s*import\s+[\"']([^\"']+)", text, re.MULTILINE):
        imports.append(name)
    for name in re.findall(r"require\(['\"]([^'\"]+)['\"]\)", text):
        imports.append(name)
    return symbols, imports


def _params(text):
    return [p.strip() for p in str(text).split(",") if p.strip()]


def read_diagnostics(text):
    """Errors, warnings and hints a document mentions.

    Recognising the shapes they arrive in is the part that can be done without
    a compiler. This does not diagnose anything; it says what kind of thing is
    being talked about, which is what a reader needs in order to know whether
    the surrounding material is about failure.
    """
    found = []
    for kind, pattern in DIAGNOSTIC_PATTERNS:
        for match in pattern.finditer(text):
            found.append(
                {
                    "kind": kind,
                    "text": match.group(0),
                    "line": text.count("\n", 0, match.start()) + 1,
                }
            )
            if len(found) >= 200:
                return found
    return found


def parse(text):
    """Everything one document states about itself as code."""
    language = detect_language(text)
    if language == "python":
        symbols, imports, problem = parse_python(text)
        if problem is not None:
            symbols, imports = [], []
    else:
        symbols, imports = parse_generic(text, language)
        problem = None
    return {
        "language": language,
        "symbols": symbols,
        "imports": sorted(set(imports)),
        "diagnostics": read_diagnostics(text),
        "parse_problem": problem,
        "read_with": "ast" if language == "python" else "patterns",
    }


def record(store, source, text, cycle=None, now=None):
    """Read a document and write what it found into the knowledge store.

    Returns the parsed record so the caller can report it without reading the
    store back.
    """
    parsed = parse(text)
    parsed["source"] = str(source)
    parsed["cycle"] = cycle
    parsed["recorded"] = time.time() if now is None else now
    store.put(SYMBOLS, str(source), parsed)

    for symbol in parsed["symbols"]:
        if symbol["kind"] not in DEFINITION_KINDS:
            # The declared tuple was a list of intentions that nothing checked
            # against, so a kind spelled differently here would be written to the
            # store and would never appear in any count of what the corpus
            # contains. Checking here is what makes adding a kind to
            # DEFINITION_KINDS mean something.
            raise ValueError(
                f"symbol {symbol['name']!r} has kind {symbol['kind']!r}, which "
                f"is not one of the declared definition kinds: "
                f"{', '.join(DEFINITION_KINDS)}"
            )
        key = f"{symbol['kind']}:{symbol['name']}"
        existing = store.collection(SYMBOLS).get(key)
        if existing is None:
            store.put(
                SYMBOLS,
                key,
                {
                    "name": symbol["name"],
                    "kind": symbol["kind"],
                    "params": symbol.get("params", []),
                    "languages": [parsed["language"]] if parsed["language"] else [],
                    "sources": [str(source)],
                },
            )
        else:
            languages = existing.setdefault("languages", [])
            if parsed["language"] and parsed["language"] not in languages:
                languages.append(parsed["language"])
            sources = existing.setdefault("sources", [])
            if str(source) not in sources:
                sources.append(str(source))
            store.put(SYMBOLS, key, existing)

    for name in parsed["imports"]:
        key = str(name)
        existing = store.collection(DEPENDENCIES).get(key)
        if existing is None:
            store.put(DEPENDENCIES, key, {"module": key, "sources": [str(source)]})
        else:
            sources = existing.setdefault("sources", [])
            if str(source) not in sources:
                sources.append(str(source))
            store.put(DEPENDENCIES, key, existing)

    if parsed["language"]:
        record_language(store, parsed["language"], source, now=now)

    # A source that could not be read is the most important thing to know about
    # it, so both kinds of diagnostic are recorded rather than dropped. A file
    # that does not parse produced an empty symbol list and, before this, no
    # record that anything was wrong: the document looked learned and
    # contributed nothing.
    problem = parsed.get("parse_problem")
    if problem:
        store.put(
            DIAGNOSTICS,
            f"parse:{source}",
            {
                "source": str(source),
                "language": parsed["language"],
                "kind": "parse_error",
                "detail": str(problem.get("detail", problem)),
                "cycle": cycle,
                "recorded": time.time() if now is None else now,
            },
        )
    for entry in parsed.get("diagnostics") or []:
        key = f"lint:{source}:{entry.get('line', '?')}"
        store.put(
            DIAGNOSTICS,
            key,
            {
                "source": str(source),
                "language": parsed["language"],
                "kind": str(entry.get("kind", "diagnostic")),
                "line": entry.get("line"),
                "detail": str(entry.get("detail", "")),
                "cycle": cycle,
                "recorded": time.time() if now is None else now,
            },
        )
    return parsed


def record_language(store, language, source, now=None):
    """Which languages the corpus contains, and where."""
    counts = store.collection(LANGUAGES)
    entry = counts.get(language)
    if entry is None:
        counts[language] = {
            "language": language,
            "documents": 1,
            "sources": [str(source)],
        }
    else:
        entry["documents"] = int(entry.get("documents", 0)) + 1
        sources = entry.setdefault("sources", [])
        if str(source) not in sources:
            sources.append(str(source))
    store.put(LANGUAGES, language, counts[language])
    return counts[language]


def cross_language(store):
    """Symbols of one name found in more than one language.

    Cross-language reasoning is listed, and the groundwork for it is
    this: the same concept reached in two languages becomes two records that
    share a name, and finding that is a lookup rather than an inference. What it
    gives is the pairing, not the reasoning about it.
    """
    pairs = []
    for key, entry in store.collection(SYMBOLS).items():
        if not isinstance(entry, dict) or not key.startswith(("function:", "method:")):
            continue
        languages = [l for l in entry.get("languages", []) if l]
        if len(languages) > 1:
            pairs.append(
                {
                    "name": entry.get("name"),
                    "kind": entry.get("kind"),
                    "languages": languages,
                    "sources": entry.get("sources", []),
                }
            )
    pairs.sort(key=lambda row: row["name"])
    return pairs


def census(store):
    """What the programming knowledge layer currently holds."""
    symbols = store.collection(SYMBOLS)
    by_kind = {}
    for entry in symbols.values():
        if isinstance(entry, dict) and entry.get("kind"):
            by_kind[entry["kind"]] = by_kind.get(entry["kind"], 0) + 1
    diagnostics = store.collection(DIAGNOSTICS)
    unreadable = sorted(
        entry.get("source", key)
        for key, entry in diagnostics.items()
        if isinstance(entry, dict) and entry.get("kind") == "parse_error"
    )
    return {
        "documents": len(
            [e for e in symbols.values() if isinstance(e, dict) and e.get("source")]
        ),
        "symbols_by_kind": by_kind,
        "dependencies": len(store.collection(DEPENDENCIES)),
        "languages": sorted(store.collection(LANGUAGES).keys()),
        "cross_language": len(cross_language(store)),
        # Reported because a document that could not be read looks identical to
        # one that said nothing worth recording, and that is the difference
        # between a corpus full of code and a corpus full of files with a .py
        # extension.
        "diagnostics": len(diagnostics),
        "unreadable_sources": unreadable,
        "note": (
            "Programming knowledge held as structure rather than as memorized "
            "text. Python is read with the standard "
            "library's own parser; other languages are read with declared "
            "heuristic patterns. Nothing here executes code or proves a "
            "programme correct."
        ),
    }
