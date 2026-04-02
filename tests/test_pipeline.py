import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from app.main import process_alarm
from app.schemas import IncomingAlarm
from conftest import VALID_PAYLOAD, ALWAYS_TRUE_INFERENCE


def _payload():
    return IncomingAlarm(**VALID_PAYLOAD)


def _mocks(tmp_path=None):
    patches = [
        patch("app.main.download_video", new_callable=AsyncMock),
        patch("app.main.run_inference", new_callable=AsyncMock),
        patch("app.main.post_result", new_callable=AsyncMock),
    ]
    return patches


@pytest.mark.asyncio
async def test_pipeline_calls_inference_with_video_and_alarm():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE) as mock_infer,
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
    mock_infer.assert_called_once_with("/tmp/fake.mp4", VALID_PAYLOAD["alarm"])


@pytest.mark.asyncio
async def test_pipeline_downloads_video_from_payload_url():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4") as mock_dl,
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())
    mock_dl.assert_called_once_with(VALID_PAYLOAD["dms_video_url"], VALID_PAYLOAD["id"])


@pytest.mark.asyncio
async def test_pipeline_posts_callback_with_review_true():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    assert mock_post.called
    result = mock_post.call_args[0][0]
    assert result.review_result is True
    assert result.confidence_level == 100


@pytest.mark.asyncio
async def test_pipeline_callback_contains_original_fields():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.id    == VALID_PAYLOAD["id"]
    assert result.imei  == VALID_PAYLOAD["imei"]
    assert result.alarm == VALID_PAYLOAD["alarm"]
    assert result.time  == VALID_PAYLOAD["time"]
    assert result.dms_video_url == VALID_PAYLOAD["dms_video_url"]


@pytest.mark.asyncio
async def test_pipeline_saves_record_to_disk(tmp_path, monkeypatch):
    from app.config import settings
    records_path = tmp_path / "records"
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_path))
    records_path.mkdir()

    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock),
    ):
        await process_alarm(_payload())

    json_files = list(records_path.glob("*.json"))
    assert len(json_files) == 1
    record = json.loads(json_files[0].read_text())
    assert record["original"]["id"] == VALID_PAYLOAD["id"]
    assert record["result"]["review_result"] is True
    assert record["result"]["confidence_level"] == 100


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_download_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, side_effect=Exception("network error")),
        patch("app.main.run_inference", new_callable=AsyncMock) as mock_infer,
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    mock_infer.assert_not_called()
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_inference_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, side_effect=Exception("model error")),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_callback_failure():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock, side_effect=RuntimeError("callback failed")),
    ):
        await process_alarm(_payload())


@pytest.mark.asyncio
async def test_pipeline_process_duration_is_positive():
    with (
        patch("app.main.download_video", new_callable=AsyncMock, return_value="/tmp/fake.mp4"),
        patch("app.main.run_inference", new_callable=AsyncMock, return_value=ALWAYS_TRUE_INFERENCE),
        patch("app.main.post_result", new_callable=AsyncMock) as mock_post,
    ):
        await process_alarm(_payload())
    result = mock_post.call_args[0][0]
    assert result.process_duration >= 0
