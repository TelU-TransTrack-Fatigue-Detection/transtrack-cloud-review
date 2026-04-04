import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.notifier import post_result
from app.schemas import ReviewResult

SAMPLE_RESULT = ReviewResult(
    id="abc123",
    imei="867395078806052",
    time="05/02/2026 00:00:06",
    alarm="eyes_closed",
    dms_video_url="https://example.com/video.mp4",
    dms_video_url_after_proccess="/tmp/abc123.mp4",
    confidence_level=100,
    review_result=True,
    process_duration=350,
    other={},
)


def _make_mock_client(side_effect=None, response=None):
    if response is None:
        response = MagicMock()
        response.raise_for_status = MagicMock()
 
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(
        side_effect=side_effect if side_effect else None,
        return_value=response,
    )
    return mock_client


@pytest.mark.asyncio
async def test_post_result_sends_to_callback_url():
    mock_client = _make_mock_client()
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        await post_result(SAMPLE_RESULT)
 
    mock_client.post.assert_called_once()
    call_url = mock_client.post.call_args[0][0]
    from app.config import settings
    assert call_url == settings.CALLBACK_URL


@pytest.mark.asyncio
async def test_post_result_payload_matches_schema():
    mock_client = _make_mock_client()
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        await post_result(SAMPLE_RESULT)
 
    sent_json = mock_client.post.call_args[1]["json"]
    assert sent_json["review_result"] is True
    assert sent_json["confidence_level"] == 100
    assert sent_json["id"] == "abc123"
    assert "dms_video_url_after_proccess" in sent_json


@pytest.mark.asyncio
async def test_post_result_retries_on_http_error():
    mock_client = _make_mock_client(side_effect=httpx.HTTPError("server error"))
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Callback failed"):
                await post_result(SAMPLE_RESULT)
    assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_post_result_succeeds_on_second_attempt():
    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    mock_client = _make_mock_client(
        side_effect=[httpx.HTTPError("fail"), ok_response]
    )
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock):
            await post_result(SAMPLE_RESULT)
    assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_post_result_raises_after_max_retries():
    mock_client = _make_mock_client(side_effect=httpx.HTTPError("fail"))
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await post_result(SAMPLE_RESULT)


@pytest.mark.asyncio
async def test_post_result_sends_only_one_request_on_success():
    mock_client = _make_mock_client()
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        await post_result(SAMPLE_RESULT)
    assert mock_client.post.call_count == 1

@pytest.mark.asyncio
async def test_post_result_payload_has_all_schema_fields():
    mock_client = _make_mock_client()
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        await post_result(SAMPLE_RESULT)
 
    sent_json = mock_client.post.call_args[1]["json"]
    required_fields = {
        "id", "imei", "time", "alarm", "dms_video_url",
        "dms_video_url_after_proccess", "confidence_level",
        "review_result", "process_duration", "other"
    }
    assert required_fields.issubset(sent_json.keys())


@pytest.mark.asyncio
async def test_post_result_sends_json_content_type():
    mock_client = _make_mock_client()
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        await post_result(SAMPLE_RESULT)
    # Verify json= kwarg was used (not data=)
    assert "json" in mock_client.post.call_args[1]


@pytest.mark.asyncio
async def test_post_result_succeeds_on_third_attempt():
    ok_response = MagicMock()
    ok_response.raise_for_status = MagicMock()
    mock_client = _make_mock_client(
        side_effect=[httpx.HTTPError("fail"), httpx.HTTPError("fail"), ok_response]
    )
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock):
            await post_result(SAMPLE_RESULT)
    assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_post_result_error_message_includes_id():
    mock_client = _make_mock_client(side_effect=httpx.HTTPError("fail"))
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError) as exc_info:
                await post_result(SAMPLE_RESULT)
    assert "abc123" in str(exc_info.value)


@pytest.mark.asyncio
async def test_post_result_sleeps_between_retries():
    mock_client = _make_mock_client(side_effect=httpx.HTTPError("fail"))
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        with patch("app.notifier.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(RuntimeError):
                await post_result(SAMPLE_RESULT)
    # Should sleep between retries (2 sleeps for 3 attempts)
    assert mock_sleep.call_count == 2


@pytest.mark.asyncio
async def test_post_result_backoff_increases():
    mock_client = _make_mock_client(side_effect=httpx.HTTPError("fail"))
    sleep_calls = []
    with patch("app.notifier.httpx.AsyncClient", return_value=mock_client):
        async def capture_sleep(t):
            sleep_calls.append(t)
        with patch("app.notifier.asyncio.sleep", side_effect=capture_sleep):
            with pytest.raises(RuntimeError):
                await post_result(SAMPLE_RESULT)
    assert len(sleep_calls) == 2
    assert sleep_calls[1] >= sleep_calls[0]