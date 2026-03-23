import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import settings
from inference import run_inference
from notifier import post_result
from schemas import IncomingAlarm, ReviewResult
from storage import save_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_RECORD_TTL_SECONDS = 30 * 60
_bearer = HTTPBearer()


def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if credentials.credentials != settings.API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        records_dir = Path(settings.RECORDS_DIR)
        for f in records_dir.glob("*.json"):
            try:
                if now - f.stat().st_mtime > _RECORD_TTL_SECONDS:
                    f.unlink()
                    logger.info("Expired record deleted: %s", f.name)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    Path(settings.RECORDS_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.RECORDS_DIR, "tmp").mkdir(parents=True, exist_ok=True)
    logger.info("Records directory ready: %s", settings.RECORDS_DIR)
    task = asyncio.create_task(cleanup_loop())
    yield
    task.cancel()


app = FastAPI(title="TransTRACK Cloud Review Engine", lifespan=lifespan)


@app.post("/review", status_code=status.HTTP_202_ACCEPTED)
async def review(
    payload: IncomingAlarm,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_api_key),
):
    background_tasks.add_task(process_alarm, payload)
    return {"status": "queued", "id": payload.id}


async def process_alarm(payload: IncomingAlarm) -> None:
    logger.info("Processing alarm id=%s alarm=%s", payload.id, payload.alarm)
    start = time.monotonic()
    video_path: str | None = None

    try:
        video_path = await download_video(payload.dms_video_url, payload.id)
        inference = await run_inference(video_path, payload.alarm)

        process_duration = int((time.monotonic() - start) * 1000)
        result = ReviewResult(
            id=payload.id,
            imei=payload.imei,
            time=payload.time,
            alarm=payload.alarm,
            dms_video_url=payload.dms_video_url,
            dms_video_url_after_proccess=inference["video_url_after_process"],
            confidence_level=inference["confidence_level"],
            review_result=inference["review_result"],
            process_duration=process_duration,
            other=inference.get("other", {}),
        )

        record = {"original": payload.model_dump(), "result": result.model_dump()}
        await asyncio.gather(
            save_record(payload.id, payload.imei, record),
            post_result(result),
        )

        logger.info(
            "Done id=%s confidence=%d review_result=%s duration_ms=%d",
            payload.id, result.confidence_level, result.review_result, process_duration,
        )

    except Exception as exc:
        logger.error("Failed to process alarm id=%s: %s", payload.id, exc, exc_info=True)

    finally:
        if video_path and not settings.KEEP_TMP_VIDEOS:
            try:
                Path(video_path).unlink(missing_ok=True)
            except Exception:
                pass


async def download_video(url: str, alarm_id: str) -> str:
    tmp_dir = Path(settings.RECORDS_DIR) / "tmp"
    video_path = tmp_dir / f"{alarm_id}.mp4"

    async with httpx.AsyncClient(timeout=settings.VIDEO_DOWNLOAD_TIMEOUT) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(video_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 64):
                    f.write(chunk)

    logger.info("Downloaded video to %s", video_path)
    return str(video_path)
