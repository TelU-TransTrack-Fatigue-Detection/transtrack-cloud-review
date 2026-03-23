import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from notifier import post_result
from schemas import ReviewResult

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


@pytest.mark.asyncio
async def test_post_result_sends_to_callback_url():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("notifier.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await post_result(SAMPLE_RESULT)

        mock_client.post.assert_called_once()
        call_url = mock_client.post.call_args[0][0]
        from config import settings
        assert call_url == settings.CALLBACK_URL


@pytest.mark.asyncio
async def test_post_result_payload_matches_schema():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("notifier.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        await post_result(SAMPLE_RESULT)

        sent_json = mock_client.post.call_args[1]["json"]
        assert sent_json["review_result"] is True
        assert sent_json["confidence_level"] == 100
        assert sent_json["id"] == "abc123"
        assert "dms_video_url_after_proccess" in sent_json


@pytest.mark.asyncio
async def test_post_result_retries_on_http_error():
    with patch("notifier.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=httpx.HTTPError("server error")
        )
        mock_client_cls.return_value = mock_client

        with patch("notifier.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError, match="Callback failed"):
                await post_result(SAMPLE_RESULT)

        assert mock_client.post.call_count == 3


@pytest.mark.asyncio
async def test_post_result_succeeds_on_second_attempt():
    mock_ok = MagicMock()
    mock_ok.raise_for_status = MagicMock()

    with patch("notifier.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(
            side_effect=[httpx.HTTPError("fail"), mock_ok]
        )
        mock_client_cls.return_value = mock_client

        with patch("notifier.asyncio.sleep", new_callable=AsyncMock):
            await post_result(SAMPLE_RESULT)

        assert mock_client.post.call_count == 2


@pytest.mark.asyncio
async def test_post_result_raises_after_max_retries():
    with patch("notifier.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.post = AsyncMock(side_effect=httpx.HTTPError("fail"))
        mock_client_cls.return_value = mock_client

        with patch("notifier.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(RuntimeError):
                await post_result(SAMPLE_RESULT)
