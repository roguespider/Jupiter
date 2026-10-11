"""Human control over Jupiter changing itself (Architecture 38).

Architecture 38 asks for explicit human control over six things: architecture
changes, source-code changes, model expansion, external tool access, persistent
memory changes and destructive operations. None of them had a control, and a
flag that gates nothing is not a control. The previous changelog said that
control would be added when expansion existed; this is the part that does not
depend on expansion, built now so that anything autonomous added afterwards is
already standing on a gate rather than having one retrofitted.

The rule is deny by default. An action is permitted only when a human has said
so, by name, for that domain, in this configuration. There is no permissive
default, no wildcard, and no way for the system to grant itself permission: the
grant is configuration, which is the one thing the system does not write.

The six domains, and what each one actually wraps:

    architecture      train writes the architecture record describing itself
    source            any write into the Python sources
    expansion         any change to the model's shape
    external          any process Jupiter starts outside itself
    memory            any knowledge write not made by the governed cycle
    destructive       any destruction of information irrecoverably

The first, second, fourth and sixth exist today. Expansion and unsanctioned
memory change are defined here so that the code which grows them inherits a gate
rather than needing one written afterwards.

Every decision is recorded, including the grants, because a control nobody can
audit is indistinguishable from a control that is simply off.
"""

import json
import os
import time
from pathlib import Path

# The six domains Architecture 38 names. A name outside this list is refused
# rather than ignored, so a typo cannot quietly open a hole.
DOMAINS = (
    "architecture",
    "source",
    "expansion",
    "external",
    "memory",
    "destructive",
)

# Environment variable a human sets to authorize domains. Comma separated.
GRANT_VARIABLE = "JUPITER_CONTROL_GRANTS"

# Where decisions are recorded. Separate from the state the system learns,
# because a control log that the system could rewrite would be worth nothing.
DECISIONS_VARIABLE = "JUPITER_CONTROL_LOG"
DEFAULT_DECISIONS = "out/control_decisions.jsonl"

# Roots Jupiter may write without a gate. Everything else is a control question.
DATA_ROOTS = ("data", "out")


class ControlDenied(PermissionError):
    """A gated action was attempted without a human authorization.

    A PermissionError rather than a bare Exception because that is what it is:
    the process was refused permission to do something, and the distinction
    matters when reading a log of what went wrong.
    """


def granted_domains(environ=None):
    """The domains a human has authorized, from configuration only.

    The environment used is the one passed in, or os.environ when it is not.
    Tests pass a dictionary rather than mutating the real environment, because a
    control test that reads a developer's own shell is not testing the control:
    with JUPITER_CONTROL_GRANTS set in the ambient environment, a test asserting
    that a gate refuses by default would find it already granted and fail for a
    reason that has nothing to do with the code under test.
    """
    environ = os.environ if environ is None else environ
    raw = environ.get(GRANT_VARIABLE, "")
    granted = set()
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        if name not in DOMAINS:
            raise ValueError(
                f"{GRANT_VARIABLE} names {name!r}, which is not one of the "
                f"Architecture 38 domains: {', '.join(DOMAINS)}"
            )
        granted.add(name)
    return granted


def _log_path(c=None):
    environ = os.environ
    if c is not None and getattr(c, "control_log", None):
        return Path(c.control_log)
    return Path(environ.get(DECISIONS_VARIABLE, DEFAULT_DECISIONS))


def _record(path, entry):
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    except OSError:
        # A control that cannot log is still a control. Refusing to proceed
        # because the audit write failed would turn an observability problem
        # into an outage, which is the opposite of what this is for.
        return None


def check(domain, action, detail=None, c=None, allowed=None):
    """Decide whether a gated action may proceed, and record the decision.

    Returns True when permitted. Raises ControlDenied otherwise, because a
    caller that forgets to test a returned False would otherwise proceed
    silently, which is the failure a control exists to prevent.
    """
    if domain not in DOMAINS:
        raise ValueError(
            f"unknown control domain {domain!r}; Architecture 38 names "
            f"{', '.join(DOMAINS)}"
        )
    permitted = granted_domains() if allowed is None else set(allowed)
    granted = domain in permitted
    entry = {
        "time": time.time(),
        "domain": domain,
        "action": action,
        "detail": detail,
        "granted": granted,
    }
    _record(_log_path(c), entry)
    if not granted:
        raise ControlDenied(
            f"Architecture 38 requires human control over {domain} changes, and "
            f"{action!r} was not authorized."
            + (f" What differs: {detail}." if detail else "")
            + f" To permit it, a human sets {GRANT_VARIABLE} to include "
            f"{domain!r}. Nothing is changed until they do. This decision has "
            f"been recorded."
        )
    return True


def allowed_roots(c=None):
    """The directories Jupiter is allowed to write without a source grant.

    Taken from the configuration rather than hardcoded. A project whose raw
    directory, curated directory or output directory has been moved is still
    doing exactly what it was configured to do, and refusing its writes because
    they are not literally under ./data and ./out would make the control wrong
    rather than strict.
    """
    names = (
        "raw_dir",
        "curated_dir",
        "out_dir",
        "evaluation_report",
        "curation_report",
        "synthesis_dir",
    )
    roots = set(DATA_ROOTS)
    if c is not None:
        for name in names:
            value = getattr(c, name, None)
            if value:
                roots.add(Path(value).name)
    return frozenset(roots)


def is_data_path(path, project=None, c=None):
    """Whether a path is data Jupiter owns, rather than source or anything else.

    Deliberately conservative: the question is not "is this inside the project"
    but "is this inside a directory Jupiter is meant to write", because a file
    under the project root is just as much source as one under src/.

    A relative path is taken as relative to the project, not to whatever
    directory the process happens to be in. Resolving it against the working
    directory instead would classify the same file differently depending on
    where the program was started from, which is a control that can be walked
    past by running from the right place.
    """
    root = Path(project or ".").resolve()
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    try:
        relative = target.resolve().relative_to(root)
    except ValueError:
        # Outside the project entirely. The configured roots still apply, so a
        # corpus that lives elsewhere is recognised as data rather than refused.
        return _matches_configured(target, c)
    return bool(relative.parts) and relative.parts[0] in allowed_roots(c)


def _matches_configured(target, c):
    """Whether a path outside the project sits under a configured data root."""
    if c is None:
        return False
    for name in ("raw_dir", "curated_dir", "out_dir", "synthesis_dir"):
        value = getattr(c, name, None)
        if not value:
            continue
        configured = Path(value)
        if not configured.is_absolute():
            configured = Path(".").resolve() / configured
        try:
            target.resolve().relative_to(configured.resolve())
            return True
        except (ValueError, OSError):
            continue
    return False


def check_write(c, path, allowed=None):
    """Gate a file write by classifying the path first.

    Writes under data/ and out/ are the system's own working state and are what
    the project has always done. Anything else is a source or system write and
    is refused without a human grant, which is the control Architecture 38 asks
    for over source-code changes.
    """
    if is_data_path(path, c=c):
        return True
    return check(
        "source",
        "write outside the data and out directories",
        detail=str(path),
        c=c,
        allowed=allowed,
    )


def check_destructive(c, path, allowed=None):
    """Gate an operation that destroys information irrecoverably.

    A move that preserves is not destructive, and conflating the two would put
    a control on the only safe path forward. Archiving a checkpoint whose shape
    no longer matches preserves it, is reversible, and happens because the
    architecture gate upstream already asked a human whether the shape may
    change. Deleting it is a different act entirely, and that is what this gate
    is for.
    """
    return check(
        "destructive",
        "destroy information irrecoverably",
        detail=str(path),
        c=c,
        allowed=allowed,
    )


def summary(c=None):
    """What has been gated, and on what authority.

    Read from the log rather than from memory, so the answer survives a restart
    and cannot be edited by the process being audited.
    """
    path = _log_path(c)
    if not path.exists():
        return {"log": str(path), "decisions": 0, "granted": 0, "denied": 0}
    granted = 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        total += 1
        if entry.get("granted"):
            granted += 1
    return {
        "log": str(path),
        "decisions": total,
        "granted": granted,
        "denied": total - granted,
        "domains": list(DOMAINS),
        "grant_variable": GRANT_VARIABLE,
    }
