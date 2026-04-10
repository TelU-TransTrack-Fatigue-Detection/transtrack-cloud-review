"""
Mock video server — Action B.

Serves .mp4 files over HTTP, replacing expired MDVR signed URLs in local testing.
Put any .mp4 in mock/videos/ and reference it as:
    http://localhost:8002/videos/<filename>.mp4

Optionally simulates slow downloads via MOCK_RATE_LIMIT_KBPS env var.

Usage (local):
    python mock/video_server.py

    # With rate limiting (simulates slow MDVR):
    MOCK_RATE_LIMIT_KBPS=500 python mock/video_server.py

Config via env vars:
    MOCK_VIDEO_PORT      = 8002      (default 8002)
    MOCK_RATE_LIMIT_KBPS = 0         (0 = unlimited)
    VIDEO_DIR            = mock/videos
"""

import logging
import os
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse

VIDEO_DIR      = Path(os.getenv("VIDEO_DIR", "mock/videos"))
MOCK_PORT      = int(os.getenv("MOCK_VIDEO_PORT", "8002"))
RATE_LIMIT_KBS = int(os.getenv("MOCK_RATE_LIMIT_KBPS", "0"))
CHUNK_SIZE     = 64 * 1024

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mock.video")

app = FastAPI(title="Mock Video Server")


def _stream(path: Path):
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_SIZE):
            yield chunk
            if RATE_LIMIT_KBS > 0:
                time.sleep(len(chunk) / (RATE_LIMIT_KBS * 1024))


@app.get("/videos/{filename}")
def serve_video(filename: str):
    path = VIDEO_DIR / filename
    if not path.exists() or path.suffix.lower() not in {".mp4", ".avi", ".mov"}:
        return JSONResponse({"error": f"not found: {filename}"}, status_code=404)

    size = path.stat().st_size
    logger.info("Serving %s (%d KB) rate_limit=%d kbps", filename, size // 1024, RATE_LIMIT_KBS)
    return StreamingResponse(
        _stream(path),
        media_type="video/mp4",
        headers={"Content-Length": str(size), "Content-Disposition": f"inline; filename={filename}"},
    )


@app.get("/videos")
def list_videos():
    files = [
        {"name": f.name, "size_kb": f.stat().st_size // 1024}
        for f in sorted(VIDEO_DIR.iterdir())
        if f.suffix.lower() in {".mp4", ".avi", ".mov"}
    ]
    if not files:
        return JSONResponse({
            "files": [],
            "hint": f"Drop .mp4 files into {VIDEO_DIR.resolve()} and they will appear here.",
        })
    return {"files": files}


@app.get("/health")
def health():
    videos = list(VIDEO_DIR.glob("*.mp4"))
    return {"status": "ok", "video_count": len(videos), "rate_limit_kbps": RATE_LIMIT_KBS}


if __name__ == "__main__":
    logger.info("Mock video server starting — port=%d video_dir=%s", MOCK_PORT, VIDEO_DIR.resolve())
    logger.info("Available videos: %s", [f.name for f in VIDEO_DIR.glob("*.mp4")])
    uvicorn.run(app, host="0.0.0.0", port=MOCK_PORT)
