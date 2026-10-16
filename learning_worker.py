"""Compatibility shim for tests that import ``learning_worker``.

The original single ``learning_worker.py`` was split into two algorithm-specific
workers.  This module re-exports the symbols that the test suite depends on so
that existing tests continue to import ``learning_worker`` without change.

The shim delegates to ``gradient_learning_worker``, which is the canonical
(legacy) implementation used by the original entry-point.

IMPORTANT: Do NOT import individual symbols from this module. Tests that need
to manipulate module-level state (e.g. ``_refused_domains``) must reference
``learning_worker.gradient`` to access the live module object.
"""

import gradient_learning_worker as gradient

# Expose the underlying module so tests that manipulate module-level state
# (e.g. ``_refused_domains``) can reach the live object.
_gradient = gradient

__all__ = [
    "run_worker",
    "discover_documents",
    "offer_grant",
    "prompt_for_growth",
    "_interactive",
    "_refused_domains",
    "sys",
    "record_cycle_knowledge",
    "gradient",
    "_gradient",
]


# --- re-exported symbols (for direct ``learning_worker.foo`` access) ---

run_worker = gradient.run_worker
discover_documents = gradient.discover_documents
offer_grant = gradient.offer_grant
prompt_for_growth = gradient.prompt_for_growth
_interactive = gradient._interactive
_refused_domains = gradient._refused_domains
sys = gradient.sys
record_cycle_knowledge = gradient.record_cycle_knowledge
