import json
import time
from pathlib import Path

import aiofiles

from config import settings


async def save_record(alarm_id: str, imei: str, data: dict) -> Path:
    records_dir = Path(settings.RECORDS_DIR)
    records_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{alarm_id}_{imei}_{int(time.time() * 1000)}.json"
    filepath = records_dir / filename

    async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, indent=2, ensure_ascii=False))

    return filepath
