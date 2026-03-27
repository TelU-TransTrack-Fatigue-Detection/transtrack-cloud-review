import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path


def test_alarm_type_121_maps_to_eyes_closed():
    from runner import process_event
    event = {
        "identity": "abc123",
        "alarm_type": "121",
        "time": "2026-02-01 05:00:00",
        "device": {"imei": "867395078806052"},
    }

    captured = {}

    async def fake_download(url, path):
        Path(path).write_bytes(b"x" * 4096)
        return True

    async def fake_inference(path, alarm):
        captured["alarm"] = alarm
        return {"confidence_level": 80, "review_result": True}

    with (
        patch("runner.download_video", side_effect=fake_download),
        patch("runner.run_inference", side_effect=fake_inference),
        patch("runner.append_excel"),
        patch("runner.TMP_DIR", Path("records/runner_tmp")),
    ):
        import asyncio
        Path("records/runner_tmp").mkdir(parents=True, exist_ok=True)
        asyncio.get_event_loop().run_until_complete(
            process_event(event, "https://example.com/video.mp4")
        )

    assert captured["alarm"] == "eyes_closed"


def test_alarm_type_122_maps_to_yawning():
    from runner import process_event
    event = {
        "identity": "abc456",
        "alarm_type": "122",
        "time": "2026-02-01 05:00:00",
        "device": {"imei": "867395078806052"},
    }

    captured = {}

    async def fake_download(url, path):
        Path(path).write_bytes(b"x" * 4096)
        return True

    async def fake_inference(path, alarm):
        captured["alarm"] = alarm
        return {"confidence_level": 75, "review_result": True}

    with (
        patch("runner.download_video", side_effect=fake_download),
        patch("runner.run_inference", side_effect=fake_inference),
        patch("runner.append_excel"),
        patch("runner.TMP_DIR", Path("records/runner_tmp")),
    ):
        import asyncio
        Path("records/runner_tmp").mkdir(parents=True, exist_ok=True)
        asyncio.get_event_loop().run_until_complete(
            process_event(event, "https://example.com/video.mp4")
        )

    assert captured["alarm"] == "yawning"
