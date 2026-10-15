"""Check growth status and show % chance of growth."""

import sys

sys.path.insert(0, ".")
import expansion

from config import load_config

c = load_config()
state = expansion.load(c)
proposed = state.get("capacity_growth")

prob = 0.0
if isinstance(proposed, dict):
    if proposed.get("approved"):
        prob = 100.0
    elif proposed.get("proposed_at_cycle") is not None:
        prob = 75.0
    else:
        prob = 25.0
else:
    proposals = state.get("_proposals", [])
    if proposals:
        prob = 50.0
    else:
        prob = 5.0

print(f"Growth probability: {prob:.2f}%")
if isinstance(proposed, dict) and not proposed.get("approved"):
    print("PENDING: True")
    print("A capacity growth is proposed and will be approved automatically.")
    result = expansion.approve_growth(c)
    if result:
        print("Growth approved and applied.")
    else:
        print("No growth was waiting to be approved.")
    sys.exit(0)
else:
    print("PENDING: False")
    print("No pending growth or already approved.")
    sys.exit(1)
