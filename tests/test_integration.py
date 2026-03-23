import asyncio
import json
import os
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport

from app.config import settings
from app.main import app

INTEGRATION_RECORDS_DIR = "./records/integration"

REAL_PAYLOAD = {
    "id": "1770249606650000020867395078806052",
    "imei": "867395078806052",
    "time": "05/02/2026 00:00:06",
    "alarm": "eyes_closed",
    "dms_video_url": os.environ.get(
        "TEST_VIDEO_URL",
        "https://mdvr.transtrack.id:36301/fileSrv/fileDown.php"
        "?filePath=RDovU2VydmVyTm9kZTEvRXZpZGVuY2UvaHRkb2Nzl3Zzc0ZpbGVzL2hmdHAvUkVDLUFMQVJNLzIwMjYwMjA1Lzg2NzM5NTA3ODgwNjA1Mi8wMDAwMDZfNjUvMV82NF82NV8yXzE3NzAyNDk2MDYubXA0"
        "&token=a48249a4ab90ed7d72042da809789fc2"
        "&ipaddr=34.124.162.30"
        "&dn=867395078806052_ch2_20260205000001_20260205000011__.mp4",
    ),
}


@pytest.fixture(autouse=True)
def setup_integration_dirs(monkeypatch):
    Path(INTEGRATION_RECORDS_DIR).mkdir(parents=True, exist_ok=True)
    Path(INTEGRATION_RECORDS_DIR, "tmp").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "RECORDS_DIR", INTEGRATION_RECORDS_DIR)
    monkeypatch.setattr(settings, "KEEP_TMP_VIDEOS", True)


def _print_video_info(video_info: dict):
    print("\n--- Video Properties ---")
    if "error" in video_info:
        print(f"  [WARNING] cv2 could not read video: {video_info['error']}")
        print("  This usually means the download token has expired.")
        print("  Set TEST_VIDEO_URL env var to a fresh signed URL to test with real video data.")
    else:
        print(f"  Resolution   : {video_info['resolution']}")
        print(f"  FPS          : {video_info['fps']}")
        print(f"  Total frames : {video_info['total_frames']}")
        print(f"  Duration     : {video_info['duration_sec']}s")
        print(f"  First frame  : {video_info['first_frame_read']}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_download_and_save():
    with patch("app.main.post_result", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = None

        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/review",
                json=REAL_PAYLOAD,
                headers={"Authorization": f"Bearer {settings.API_KEY}"},
            )

        assert resp.status_code == 202
        await asyncio.sleep(15)

    records_path = Path(INTEGRATION_RECORDS_DIR)
    json_files = sorted(records_path.glob("*.json"), key=lambda f: f.stat().st_mtime)
    assert len(json_files) >= 1, "No record file was saved"

    record = json.loads(json_files[-1].read_text(encoding="utf-8"))

    print("\n--- Integration Record ---")
    print(json.dumps(record, indent=2))

    video_info = record["result"]["other"]["video_info"]
    _print_video_info(video_info)

    video_path = Path(record["result"]["dms_video_url_after_proccess"])
    assert video_path.exists(), f"Video file not kept at {video_path}"
    assert video_path.stat().st_size > 0, "Video file is empty"
    print(f"  File size    : {video_path.stat().st_size / 1024:.1f} KB")

    assert record["original"]["id"] == REAL_PAYLOAD["id"]
    assert record["original"]["alarm"] == "eyes_closed"
    assert "video_info" in record["result"]["other"]
    assert isinstance(record["result"]["confidence_level"], int)
    assert isinstance(record["result"]["review_result"], bool)
    assert record["result"]["process_duration"] > 0

    if "error" not in video_info:
        assert video_info["total_frames"] > 0, "Video has no frames"
        assert video_info["fps"] > 0, "Video FPS is zero"
        assert video_info["duration_sec"] > 0, "Video duration is zero"
        assert video_info["first_frame_read"] is True, "Could not read first frame"
        assert "x" in video_info["resolution"], "Bad resolution format"

    mock_post.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_download_callback_payload_shape():
    captured = {}

    async def capture_result(result):
        captured["result"] = result

    with patch("app.main.post_result", side_effect=capture_result):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post(
                "/review",
                json=REAL_PAYLOAD,
                headers={"Authorization": f"Bearer {settings.API_KEY}"},
            )
        await asyncio.sleep(15)

    assert "result" in captured, "Callback was never triggered"
    r = captured["result"]

    print("\n--- Callback Payload ---")
    print(json.dumps(r.model_dump(), indent=2))

    video_info = r.other.get("video_info", {})
    _print_video_info(video_info)

    video_path = Path(r.dms_video_url_after_proccess)
    assert video_path.exists(), f"Video file not kept at {video_path}"
    print(f"  File size    : {video_path.stat().st_size / 1024:.1f} KB")

    assert r.id == REAL_PAYLOAD["id"]
    assert r.imei == REAL_PAYLOAD["imei"]
    assert r.alarm == REAL_PAYLOAD["alarm"]
    assert r.dms_video_url == REAL_PAYLOAD["dms_video_url"]
    assert isinstance(r.confidence_level, int)
    assert isinstance(r.review_result, bool)
    assert r.process_duration > 0
    assert r.dms_video_url_after_proccess != ""

    if "error" not in video_info:
        assert video_info.get("total_frames", 0) > 0
        assert video_info.get("fps", 0) > 0
        assert video_info.get("duration_sec", 0) > 0
        assert video_info.get("first_frame_read") is True
