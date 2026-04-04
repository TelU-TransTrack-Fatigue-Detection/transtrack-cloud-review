"""Tests for app.main.download_video — async streaming video download."""
import pytest
import httpx
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import download_video
from app.config import settings


# ─── Helper: buat mock response yang benar ───────────────────────────────────
#
# Masalah umum: aiter_bytes() dipanggil dengan `async for chunk in response.aiter_bytes(...)`.
# Ini artinya aiter_bytes adalah fungsi BIASA (bukan async) yang RETURN async generator.
# Jika di-mock dengan AsyncMock, ia menjadi coroutine → `async for` gagal dengan:
#   TypeError: 'async for' requires an object with __aiter__ method, got coroutine
#
# Solusi: gunakan MagicMock (bukan AsyncMock) untuk aiter_bytes,
# dengan side_effect yang return async generator langsung.

async def _async_gen(items):
    """Async generator untuk mensimulasikan streaming chunks."""
    for item in items:
        yield item


def _make_mock_response(chunks: list, raise_status_error=None):
    """
    Buat mock httpx response yang kompatibel dengan `async for chunk in response.aiter_bytes(...)`.
    - aiter_bytes harus MagicMock (bukan AsyncMock) yang return async generator.
    - raise_for_status bisa di-set untuk mensimulasikan HTTP error.
    """
    mock_response = MagicMock()

    if raise_status_error:
        mock_response.raise_for_status = MagicMock(side_effect=raise_status_error)
    else:
        mock_response.raise_for_status = MagicMock()

    # aiter_bytes dipanggil sebagai fungsi biasa → return async generator
    mock_response.aiter_bytes = MagicMock(return_value=_async_gen(chunks))

    return mock_response


def _make_mock_client(mock_response):
    """Buat mock httpx.AsyncClient dengan stream context manager."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.stream = MagicMock(return_value=_async_context(mock_response))
    return mock_client


class _async_context:
    """Context manager async untuk membungkus mock response."""
    def __init__(self, obj):
        self._obj = obj
    async def __aenter__(self):
        return self._obj
    async def __aexit__(self, *args):
        return False


# ─── Tests ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_download_video_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    mock_response = _make_mock_response([b"chunk1", b"chunk2"])
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client):
        path = await download_video("https://example.com/video.mp4", "test-001")

    assert Path(path).exists()


@pytest.mark.asyncio
async def test_download_video_returns_correct_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    mock_response = _make_mock_response([b"data"])
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client):
        path = await download_video("https://example.com/video.mp4", "alarm-xyz")

    assert "alarm-xyz" in path
    assert path.endswith(".mp4")


@pytest.mark.asyncio
async def test_download_video_writes_content(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    content       = b"fake mp4 binary content"
    mock_response = _make_mock_response([content])
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client):
        path = await download_video("https://example.com/video.mp4", "alarm-001")

    assert Path(path).read_bytes() == content


@pytest.mark.asyncio
async def test_download_video_writes_multiple_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    chunks        = [b"part1", b"part2", b"part3"]
    mock_response = _make_mock_response(chunks)
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client):
        path = await download_video("https://example.com/video.mp4", "alarm-002")

    assert Path(path).read_bytes() == b"part1part2part3"


@pytest.mark.asyncio
async def test_download_video_uses_configured_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    mock_response = _make_mock_response([b""])
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client) as mock_cls:
        await download_video("https://example.com/video.mp4", "t1")

    kwargs = mock_cls.call_args[1]
    assert kwargs.get("timeout") == settings.VIDEO_DOWNLOAD_TIMEOUT


@pytest.mark.asyncio
async def test_download_video_raises_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "RECORDS_DIR", str(tmp_path))
    (tmp_path / "tmp").mkdir()

    error         = httpx.HTTPStatusError("404", request=MagicMock(), response=MagicMock())
    mock_response = _make_mock_response([], raise_status_error=error)
    mock_client   = _make_mock_client(mock_response)

    with patch("app.main.httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(httpx.HTTPStatusError):
            await download_video("https://example.com/missing.mp4", "alarm-001")