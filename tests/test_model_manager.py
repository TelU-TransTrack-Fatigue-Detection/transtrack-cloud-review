"""Tests for app.model_manager — model file download and availability checks."""
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.model_manager import ensure_mediapipe, ensure_classifier, ensure_all_models


# ─── ensure_mediapipe ────────────────────────────────────────────────────────

def test_ensure_mediapipe_skips_download_if_exists(tmp_path):
    fake_task = tmp_path / "face_landmarker.task"
    fake_task.write_bytes(b"fake model")

    with patch("app.model_manager._MEDIAPIPE_PATH", fake_task):
        with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
            ensure_mediapipe()
    mock_dl.assert_not_called()


def test_ensure_mediapipe_downloads_if_missing(tmp_path):
    missing = tmp_path / "face_landmarker.task"

    with patch("app.model_manager._MEDIAPIPE_PATH", missing):
        with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
            ensure_mediapipe()
    mock_dl.assert_called_once()


def test_ensure_mediapipe_download_uses_correct_url(tmp_path):
    missing = tmp_path / "face_landmarker.task"

    with patch("app.model_manager._MEDIAPIPE_PATH", missing):
        with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
            ensure_mediapipe()
    url_called = mock_dl.call_args[0][0]
    assert "mediapipe" in url_called
    assert "face_landmarker" in url_called


def test_ensure_mediapipe_creates_parent_directory(tmp_path):
    deep_path = tmp_path / "nested" / "dir" / "face_landmarker.task"

    with patch("app.model_manager._MEDIAPIPE_PATH", deep_path):
        with patch("app.model_manager.urllib.request.urlretrieve"):
            ensure_mediapipe()
    assert deep_path.parent.exists()


# ─── ensure_classifier ───────────────────────────────────────────────────────

def test_ensure_classifier_skips_download_if_exists(tmp_path):
    existing = tmp_path / "best_val_f1.pth"
    existing.write_bytes(b"fake weights")

    with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
        ensure_classifier(existing, download_url="https://example.com/model.pth")
    mock_dl.assert_not_called()


def test_ensure_classifier_raises_if_missing_and_no_url(tmp_path):
    missing = tmp_path / "best_val_f1.pth"
    with pytest.raises(FileNotFoundError, match="MODEL_DOWNLOAD_URL"):
        ensure_classifier(missing, download_url="")


def test_ensure_classifier_downloads_if_missing_with_url(tmp_path):
    missing = tmp_path / "best_val_f1.pth"
    with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
        ensure_classifier(missing, download_url="https://example.com/model.pth")
    mock_dl.assert_called_once()


def test_ensure_classifier_download_saves_to_correct_path(tmp_path):
    missing = tmp_path / "best_val_f1.pth"
    url = "https://example.com/model.pth"
    with patch("app.model_manager.urllib.request.urlretrieve") as mock_dl:
        ensure_classifier(missing, download_url=url)
    assert mock_dl.call_args[0][0] == url
    assert mock_dl.call_args[0][1] == missing


def test_ensure_classifier_creates_parent_directory(tmp_path):
    deep = tmp_path / "models" / "classifier" / "best_val_f1.pth"
    with patch("app.model_manager.urllib.request.urlretrieve"):
        ensure_classifier(deep, download_url="https://example.com/model.pth")
    assert deep.parent.exists()


def test_ensure_classifier_raises_with_path_hint_when_no_url(tmp_path):
    missing = tmp_path / "best_val_f1.pth"
    with pytest.raises(FileNotFoundError) as exc_info:
        ensure_classifier(missing, download_url="")
    assert str(missing) in str(exc_info.value)


# ─── ensure_all_models ───────────────────────────────────────────────────────

def test_ensure_all_models_calls_both(tmp_path):
    model_path = tmp_path / "best_val_f1.pth"
    model_path.write_bytes(b"fake")

    with patch("app.model_manager.ensure_mediapipe") as mock_mp:
        with patch("app.model_manager.ensure_classifier") as mock_cls:
            ensure_all_models(model_path, download_url="")

    mock_mp.assert_called_once()
    mock_cls.assert_called_once_with(model_path, "")


def test_ensure_all_models_passes_download_url(tmp_path):
    model_path = tmp_path / "best_val_f1.pth"
    url = "https://example.com/model.pth"

    with patch("app.model_manager.ensure_mediapipe"):
        with patch("app.model_manager.ensure_classifier") as mock_cls:
            ensure_all_models(model_path, download_url=url)

    mock_cls.assert_called_once_with(model_path, url)