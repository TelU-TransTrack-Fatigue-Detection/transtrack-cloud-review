"""
Mock callback server — Action A.

Receives POST /callback from the worker, validates the payload against
the ReviewResult schema, logs every request, and returns a configurable
response code so you can test error handling.

Usage (local):
    python mock/callback_server.py

Config via env vars:
    MOCK_MODE   = 200 | 400 | 500   (default 200)
    MOCK_PORT   = 8001               (default 8001)
    LOG_DIR     = mock/logs          (default mock/logs)

Override per-request with header:
    X-Mock-Mode: 400
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, Response
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.schemas import ReviewResult

MOCK_MODE = int(os.getenv("MOCK_MODE", "200"))
MOCK_PORT = int(os.getenv("MOCK_PORT", "8001"))
LOG_DIR   = Path(os.getenv("LOG_DIR", "mock/logs"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "callback.log"),
    ],
)
logger = logging.getLogger("mock.callback")

app = FastAPI(title="Mock Callback Server")
_received: list[dict] = []


@app.post("/callback")
async def receive_callback(request: Request):
    raw   = await request.body()
    mode  = int(request.headers.get("X-Mock-Mode", MOCK_MODE))
    ts    = datetime.now(timezone.utc).isoformat()

    try:
        data   = json.loads(raw)
        result = ReviewResult(**data)
        valid  = True
        error  = None
        logger.info(
            "RECEIVED id=%s imei=%s review_result=%s confidence=%d label=%s",
            result.id, result.imei, result.review_result,
            result.confidence_level, result.other.get("label", "?"),
        )
    except (json.JSONDecodeError, ValidationError) as exc:
        valid = False
        error = str(exc)
        data  = raw.decode(errors="replace")
        logger.warning("SCHEMA VALIDATION FAILED: %s", error)
        mode  = 400

    entry = {"ts": ts, "valid": valid, "mode": mode, "data": data, "error": error}
    _received.append(entry)

    log_path = LOG_DIR / f"callback_{ts[:10]}.jsonl"
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")

    if mode == 500:
        return Response(content="Internal Server Error", status_code=500)
    if mode == 400:
        return Response(
            content=json.dumps({"error": error or "bad request"}),
            status_code=400,
            media_type="application/json",
        )
    return Response(
        content=json.dumps({"status": "ok", "id": data.get("id") if isinstance(data, dict) else None}),
        status_code=200,
        media_type="application/json",
    )


@app.get("/received")
def list_received():
    return {"count": len(_received), "items": _received}


@app.get("/received/latest")
def latest():
    return _received[-1] if _received else {}


@app.delete("/received")
def clear():
    _received.clear()
    return {"status": "cleared"}


@app.get("/health")
def health():
    return {"status": "ok", "mode": MOCK_MODE, "received": len(_received)}


if __name__ == "__main__":
    logger.info("Mock callback server starting — port=%d mode=%d", MOCK_PORT, MOCK_MODE)
    uvicorn.run(app, host="0.0.0.0", port=MOCK_PORT)
