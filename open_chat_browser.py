#!/usr/bin/env python3
"""Open browser after a short delay so the server has time to start."""

import time
import webbrowser

time.sleep(2)
webbrowser.open("http://127.0.0.1:8081/chat.html")
