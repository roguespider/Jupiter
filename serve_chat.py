#!/usr/bin/env python3
"""Web chat server that connects chat.html to the actual Jupiter model.

This file now serves the single runtime API defined in jupiter_api.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import and serve the single runtime API
from jupiter_api import app

if __name__ == "__main__":
    import uvicorn

    print("Jupiter chat server running at http://127.0.0.1:8081/chat.html")
    print("Point browser to /chat.html. Press Ctrl+C to stop.")
    print(
        "API endpoints: /api/health /api/chat /api/status /api/metrics /api/events /ws/events"
    )
    uvicorn.run(app, host="0.0.0.0", port=8081)
