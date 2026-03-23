import asyncio
import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch, call

from conftest import VALID_PAYLOAD, VALID_HEADERS, ALWAYS_TRUE_INFERENCE


@pytest.mark.asyncio
async def test_pipeline_calls_inference_with_video_and_alarm(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    mock_infer.assert_called_once_with("/tmp/fake.mp4", VALID_PAYLOAD["alarm"])


@pytest.mark.asyncio
async def test_pipeline_downloads_video_from_payload_url(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    mock_dl.assert_called_once_with(VALID_PAYLOAD["dms_video_url"], VALID_PAYLOAD["id"])


@pytest.mark.asyncio
async def test_pipeline_posts_callback_with_review_true(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    assert mock_post.called
    result = mock_post.call_args[0][0]
    assert result.review_result is True
    assert result.confidence_level == 100


@pytest.mark.asyncio
async def test_pipeline_callback_contains_original_fields(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    result = mock_post.call_args[0][0]
    assert result.id == VALID_PAYLOAD["id"]
    assert result.imei == VALID_PAYLOAD["imei"]
    assert result.alarm == VALID_PAYLOAD["alarm"]
    assert result.time == VALID_PAYLOAD["time"]
    assert result.dms_video_url == VALID_PAYLOAD["dms_video_url"]


@pytest.mark.asyncio
async def test_pipeline_saves_record_to_disk(tmp_path, monkeypatch):
    from config import settings
    records_path = tmp_path / "records"
    monkeypatch.setattr(settings, "RECORDS_DIR", str(records_path))

    with (
        patch("main.download_video", new_callable=AsyncMock) as mock_dl,
        patch("main.run_inference", new_callable=AsyncMock) as mock_infer,
        patch("main.post_result", new_callable=AsyncMock),
    ):
        mock_dl.return_value = "/tmp/fake.mp4"
        mock_infer.return_value = ALWAYS_TRUE_INFERENCE

        from httpx import AsyncClient, ASGITransport
        from main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            await ac.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
            await asyncio.sleep(0.3)

    json_files = list(records_path.glob("*.json"))
    assert len(json_files) == 1

    record = json.loads(json_files[0].read_text())
    assert record["original"]["id"] == VALID_PAYLOAD["id"]
    assert record["result"]["review_result"] is True
    assert record["result"]["confidence_level"] == 100


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_download_failure(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    mock_dl.side_effect = Exception("network error")

    resp = await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    assert resp.status_code == 202
    mock_infer.assert_not_called()
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_inference_failure(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    mock_infer.side_effect = Exception("model error")

    resp = await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    assert resp.status_code == 202
    mock_post.assert_not_called()


@pytest.mark.asyncio
async def test_pipeline_does_not_crash_on_callback_failure(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    mock_post.side_effect = RuntimeError("callback failed after retries")

    resp = await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    assert resp.status_code == 202


@pytest.mark.asyncio
async def test_pipeline_process_duration_is_positive(client_with_mocks):
    client, mock_dl, mock_infer, mock_post = client_with_mocks
    await client.post("/review", json=VALID_PAYLOAD, headers=VALID_HEADERS)
    await asyncio.sleep(0.2)

    result = mock_post.call_args[0][0]
    assert result.process_duration >= 0
