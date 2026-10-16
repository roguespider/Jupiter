#!/usr/bin/env python3
"""Single runtime API server around Jupiter's existing runtime.

Exposes:
- GET  /api/health
- POST /api/chat
- GET  /api/status
- GET  /api/metrics
- GET  /api/events
- WS   /ws/events

Every important action produces a structured event via event_system.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Existing Jupiter runtime imports
import dialogue
import store as store_module
import torch
from agent_memory import AgentMemory

# Event system
from event_system import EventStore, get_store
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from model import Jupiter, ModelConfig
from safetensors.torch import load_model
from tokenizer import load_tokenizer

import retrieval
from config import load_config, read_architecture

# Load runtime once (same pattern as serve_chat.py)
SCRIPT_DIR = Path(__file__).resolve().parent
c = load_config()
out_dir_path = Path(c.out_dir)
if not out_dir_path.is_absolute():
    out_dir_path = SCRIPT_DIR / c.out_dir
checkpoint = out_dir_path / "Jupiter_latest.safetensors"
architecture = out_dir_path / "Jupiter.architecture.json"

stored, _ = read_architecture(str(architecture))
cfg = ModelConfig(**stored)
model = Jupiter(cfg).to(c.device)
load_model(model, str(checkpoint), strict=True)
model.eval()

knowledge_store_path = Path(c.knowledge_store)
if not knowledge_store_path.is_absolute():
    knowledge_store_path = SCRIPT_DIR / c.knowledge_store
knowledge = (
    store_module.Store(knowledge_store_path) if knowledge_store_path.exists() else None
)

vocab_path = (
    Path(c.tokenizer_vocab)
    if hasattr(c, "tokenizer_vocab") and c.tokenizer_vocab
    else None
)
if vocab_path and not vocab_path.is_absolute():
    vocab_path = SCRIPT_DIR / c.tokenizer_vocab
    original_vocab = c.tokenizer_vocab
    c.tokenizer_vocab = str(vocab_path)
    tokenizer = load_tokenizer(c)
    c.tokenizer_vocab = original_vocab
else:
    tokenizer = load_tokenizer(c)

memory = AgentMemory(str(out_dir_path / "agent_memory.sqlite3"))
history = torch.zeros((1, 0), dtype=torch.long, device=c.device)

# Initialize event store with a session ID
store = get_store()

app = FastAPI(title="Jupiter Runtime API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve chat.html and static assets
app.mount("/static", StaticFiles(directory=str(SCRIPT_DIR)), name="static")

# In-memory event stream for WS clients (backward-compat)
live_events = []


def broadcast_event(event_dict: dict):
    """Broadcast to both the event store and the live WS stream."""
    # Ensure event has an ID from the store
    event_obj = store.emit(
        event_dict.get("event_type", event_dict.get("type", "unknown")),
        event_dict.get("data", event_dict),
    )
    live_events.append(event_obj.to_dict())
    if len(live_events) > 100:
        live_events[:] = live_events[-100:]


def emit(event_type: str, data: dict = None):
    """Emit a structured event through the event system."""
    event_obj = store.emit(event_type, data or {})
    live_events.append(event_obj.to_dict())
    if len(live_events) > 100:
        live_events[:] = live_events[-100:]
    return event_obj


@app.get("/chat.html")
async def chat_html():
    html_path = SCRIPT_DIR / "chat.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/api/health")
async def health():
    emit("runtime.status", {"status": "healthy", "service": "jupiter"})
    return {
        "status": "running",
        "service": "jupiter",
        "timestamp": time.time(),
    }


@app.get("/api/events")
async def get_events(since: str = None, limit: int = 50):
    """Retrieve events. Pass ?since=<event_id> for reconnect-safe streaming."""
    if since:
        events = store.get_since(since)
    else:
        events = store.get_last_n(limit)
    return {"events": [e.to_dict() for e in events], "session_id": store.session_id}


@app.get("/api/events/{event_id}")
async def get_event_by_id(event_id: str):
    event = store.get_by_id(event_id)
    if event is None:
        return JSONResponse(
            {"error": "event not found", "event_id": event_id}, status_code=404
        )
    return event.to_dict()


@app.post("/api/chat")
async def chat_api(request: Request):
    global history
    body = await request.json()
    line = body.get("message", "").strip()

    # message.received
    emit("message.received", {"message": line, "length": len(line)})

    if not line:
        emit("runtime.error", {"reason": "empty_message", "message": line})
        return JSONResponse({"reply": "[empty message]"})

    # Commands
    if line.startswith("/remember "):
        parts = line.replace("/remember ", "").split(" ", 1)
        kind = parts[0].strip() if parts else "fact"
        content = parts[1].strip() if len(parts) > 1 else ""
        if content:
            memory.remember(kind, content, source="user", confidence=0.95)
            emit("memory.recalled", {"kind": kind, "content_preview": content[:80]})
            emit("learning.update", {"action": "remember", "kind": kind})
            return JSONResponse({"reply": "[memory saved]"})
        else:
            emit("runtime.error", {"reason": "invalid_remember_usage"})
            return JSONResponse({"reply": "[usage: /remember <kind> <text>]"})

    if line.startswith("/correct "):
        rest = line.replace("/correct ", "").strip()
        if "=>" in rest:
            old_text, corrected_text = rest.split("=>", 1)
            memory.correct(old_text.strip(), corrected_text.strip())
            emit(
                "memory.recalled",
                {
                    "action": "correct",
                    "old": old_text.strip(),
                    "new": corrected_text.strip(),
                },
            )
            emit("learning.update", {"action": "correct"})
            return JSONResponse({"reply": "[correction saved]"})
        else:
            emit("runtime.error", {"reason": "invalid_correct_usage"})
            return JSONResponse({"reply": "[usage: /correct <old> => <corrected>]"})

    if line.startswith("/feedback "):
        feedback_text = line.replace("/feedback ", "").strip()
        outcome = "helpful"
        details = ""
        if feedback_text.startswith("wrong"):
            outcome = "wrong"
            details = feedback_text.replace("wrong", "").strip()
        elif feedback_text.startswith("partial"):
            outcome = "partial"
            details = feedback_text.replace("partial", "").strip()
        elif feedback_text.startswith("uncertain"):
            outcome = "uncertain"
            details = feedback_text.replace("uncertain", "").strip()
        emit("feedback.received", {"outcome": outcome, "details": details})
        emit("learning.update", {"action": "feedback", "outcome": outcome})
        return JSONResponse({"reply": f"[feedback recorded: {outcome}]"})

    if line == "/memory":
        try:
            stats = memory.stats()
        except Exception as e:
            stats = {"error": "memory stats unavailable", "detail": str(e)}
            emit("runtime.error", {"reason": "memory_stats_failed", "detail": str(e)})
        emit("memory.recalled", {"action": "stats", "stats": stats})
        return JSONResponse(
            {"reply": "[memory] " + ", ".join(f"{k}:{v}" for k, v in stats.items())}
        )

    if line == "/exit":
        memory.close()
        emit("message.received", {"message": "/exit", "action": "session_end"})
        emit("learning.update", {"action": "session_ended"})
        return JSONResponse({"reply": "[session ended]"})

    # Normal generation path
    # message.received already emitted above

    # context.retrieved
    turn = torch.tensor(
        [tokenizer.encode(dialogue.format_turn(dialogue.HUMAN, line))],
        dtype=torch.long,
        device=c.device,
    )

    # memory.recalled
    corrections = memory.recall_corrections(line)
    correction_text = ""
    if corrections:
        correction_lines = []
        for c_item in corrections:
            meta = c_item.get("metadata", {})
            original = meta.get("corrects", "")
            if original:
                correction_lines.append(
                    f"[correction] Corrected: {original}\n         Answer: {c_item['content']}"
                )
            else:
                correction_lines.append(f"[correction] {c_item['content']}")
        correction_text = "\n\n".join(correction_lines)
    emit(
        "memory.recalled", {"corrections_found": len(corrections) if corrections else 0}
    )

    mem_items = memory.recall(line, limit=4)
    memory_context = "\n".join(
        f"[memory {m['kind']}] {m['content']}" for m in mem_items
    )
    emit(
        "memory.recalled",
        {"items_recalled": len(mem_items), "kinds": [m.get("kind") for m in mem_items]},
    )

    # context.retrieved (knowledge retrieval)
    from sample import recalled_context

    prefix, results = recalled_context(knowledge, line, max(32, cfg.block_size // 3))
    emit(
        "context.retrieved",
        {
            "knowledge_results": len(results),
            "prefix_length": len(prefix) if prefix else 0,
        },
    )

    # response.token (generation) - uses real model.generate()
    context_parts = []
    if correction_text:
        context_parts.append(correction_text)
    if memory_context:
        context_parts.append(memory_context)
    if prefix:
        context_parts.append(prefix)
    prefix_str = "\n\n".join(context_parts) if context_parts else ""

    if prefix_str:
        prefix_tokens = torch.tensor(
            [tokenizer.encode(prefix_str + "\n")],
            dtype=torch.long,
            device=c.device,
        )
    else:
        prefix_tokens = torch.zeros((1, 0), dtype=torch.long, device=c.device)

    # response.token (generation)
    from sample import generate, generation_context

    context = generation_context(prefix_tokens, history, turn, cfg.block_size)
    text = generate(model, cfg, c, tokenizer, context)

    # Emit token-level event for the response
    emit(
        "response.token",
        {"tokens_generated": len(text.split()), "response_preview": text[:120]},
    )

    interaction_id = memory.log_interaction(
        line, text, metadata={"recalled": len(results)}
    )

    history = torch.cat((history, turn), dim=1)[:, -cfg.block_size :]
    jupiter_turn = torch.tensor(
        [tokenizer.encode(dialogue.format_turn(dialogue.JUPITER, text))],
        dtype=torch.long,
        device=c.device,
    )
    history = torch.cat((history, jupiter_turn), dim=1)[:, -cfg.block_size :]

    emit(
        "message.received",
        {"message": text, "action": "response", "interaction_id": interaction_id},
    )
    emit(
        "context.retrieved",
        {"action": "generation_complete", "history_length": int(history.shape[1])},
    )

    # Broadcast via legacy stream too
    broadcast_event(
        {
            "type": "chat",
            "user": line,
            "reply": text,
            "interaction_id": interaction_id,
            "event_id": store.events[-1].event_id if store.events else None,
        }
    )

    return JSONResponse({"reply": text})


@app.get("/api/status")
async def status():
    emit("worker.status", {"component": "api", "status": "running"})
    mem_stats = {}
    try:
        mem_stats = memory.stats()
    except Exception as e:
        mem_stats = {
            "error": "memory stats unavailable in this thread",
            "detail": str(e),
        }
        emit("runtime.error", {"reason": "memory_stats_failed", "detail": str(e)})
    telemetry_path = SCRIPT_DIR / "telemetry.jsonl"
    telemetry_exists = telemetry_path.exists()
    telemetry_size = telemetry_path.stat().st_size if telemetry_exists else 0

    return {
        "model": {
            "name": "Jupiter",
            "config": cfg.to_dict(),
            "parameters": model.parameter_count(),
            "device": str(c.device),
        },
        "memory": {
            "stats": mem_stats,
            "history_length": history.shape[1],
        },
        "workers": {
            "telemetry_exists": telemetry_exists,
            "telemetry_size_bytes": telemetry_size,
        },
        "runtime": {
            "port": 8081,
            "script_dir": str(SCRIPT_DIR),
        },
        "session_id": store.session_id,
    }


@app.get("/api/metrics")
async def metrics():
    emit("worker.status", {"component": "metrics", "status": "query"})
    telemetry_path = SCRIPT_DIR / "telemetry.jsonl"
    recent_events = []
    if telemetry_path.exists():
        try:
            with open(telemetry_path, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
                for line in lines[-20:]:
                    line = line.strip()
                    if line:
                        try:
                            recent_events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception:
            pass
    return {
        "runtime_metrics": {
            "timestamp": time.time(),
            "model_loaded": model is not None,
            "history_tokens": int(history.shape[1]),
            "telemetry_events": len(recent_events),
        },
        "recent_telemetry": recent_events,
        "session_id": store.session_id,
    }


@app.websocket("/ws/events")
async def events(websocket: WebSocket):
    await websocket.accept()
    # Send existing events first (reconnect-safe with IDs)
    for event in live_events:
        await websocket.send_json(event)
    # Keep connection open and stream new events
    try:
        while True:
            await websocket.receive_text()
            # After any client message, send latest events
            for event in live_events[-10:]:
                await websocket.send_json(event)
    except Exception:
        pass


# Legacy /chat endpoint for backward compatibility
@app.post("/chat")
async def chat_legacy(request: Request):
    return await chat_api(request)


if __name__ == "__main__":
    import uvicorn

    print(f"Jupiter runtime API running at http://127.0.0.1:8081/")
    print(
        "Endpoints: /api/health /api/chat /api/status /api/metrics /api/events /ws/events"
    )
    uvicorn.run(app, host="0.0.0.0", port=8081)
