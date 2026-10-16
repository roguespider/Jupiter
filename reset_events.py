#!/usr/bin/env python3
"""Reset the global event store — wired to the entry point's reset-events mode."""

from event_system import reset_store

if __name__ == "__main__":
    reset_store()
    print("Event store reset.")
