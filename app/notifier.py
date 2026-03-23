import asyncio
import logging

import httpx

from .config import settings
from .schemas import ReviewResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_BACKOFF_BASE = 1


async def post_result(result: ReviewResult) -> None:
    payload = result.model_dump()
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=settings.HTTPX_TIMEOUT) as client:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = await client.post(settings.CALLBACK_URL, json=payload)
                response.raise_for_status()
                logger.info("Callback delivered id=%s (attempt %d)", result.id, attempt)
                return
            except httpx.HTTPError as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "Callback attempt %d/%d failed id=%s: %s — retry in %ds",
                    attempt, _MAX_RETRIES, result.id, exc, wait,
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(wait)

    raise RuntimeError(
        f"Callback failed for id={result.id} after {_MAX_RETRIES} attempts"
    ) from last_exc
