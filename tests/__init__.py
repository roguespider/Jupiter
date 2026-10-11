"""Jupiter test suite.

Covers the levels Architecture 41 asks for: unit tests for individual
components, integration tests for subsystem interaction, a learning test that
checks a micro-update produces the intended result, and regression tests for
the acceptance and forgetting logic.

Uses unittest from the standard library. Architecture 22 requires checking the
standard library before adding a dependency, and unittest covers every level
required here, so the project gains no new package.

Run from the Jupiter directory:

    python -m unittest discover -s tests -t .
"""
