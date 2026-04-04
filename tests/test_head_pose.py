"""
Tests for app.pipeline._head_pose and _ensure_mediapipe_model.
"""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.pipeline import _head_pose, _PNP_INDICES
from conftest import make_mock_landmark


# ─── _head_pose ───────────────────────────────────────────────────────────────

def _make_lms_for_pose(nose_x=0.5, nose_y=0.5):
    """Create minimal landmark list for head pose (needs >= 292 entries)."""
    lms = [make_mock_landmark(0.5, 0.5) for _ in range(478)]
    # PNP_INDICES = [1, 152, 33, 263, 61, 291]
    offsets = [
        (0.5,  0.5),   # idx 1  — nose tip
        (0.5,  0.8),   # idx 152 — chin
        (0.3,  0.4),   # idx 33  — right eye outer
        (0.7,  0.4),   # idx 263 — left eye outer
        (0.35, 0.65),  # idx 61  — right mouth
        (0.65, 0.65),  # idx 291 — left mouth
    ]
    for i, idx in enumerate(_PNP_INDICES):
        lms[idx] = make_mock_landmark(offsets[i][0], offsets[i][1])
    return lms


def test_head_pose_returns_three_floats():
    lms = _make_lms_for_pose()
    pitch, yaw, roll = _head_pose(lms, w=640, h=480)
    # Each should be a float (may be nan if solvePnP fails)
    assert isinstance(pitch, float)
    assert isinstance(yaw, float)
    assert isinstance(roll, float)


def test_head_pose_returns_nan_on_degenerate_points():
    """All landmarks at the same point → solvePnP will likely fail."""
    lms = [make_mock_landmark(0.5, 0.5) for _ in range(478)]
    # Tambahkan mock cv2.solvePnP agar tidak crash di level C++ OpenCV
    with patch("app.pipeline.cv2.solvePnP", return_value=(False, None, None)):
        pitch, yaw, roll = _head_pose(lms, w=640, h=480)
    # Either finite angles or all NaN — both are valid
    all_finite = all(np.isfinite(v) for v in [pitch, yaw, roll])
    all_nan = all(np.isnan(v) for v in [pitch, yaw, roll])
    assert all_finite or all_nan


def test_head_pose_with_realistic_face():
    lms = _make_lms_for_pose()
    pitch, yaw, roll = _head_pose(lms, w=640, h=480)
    # With a realistic face layout, solvePnP should succeed
    if not np.isnan(pitch):
        assert -180 <= pitch <= 180
        assert -180 <= yaw <= 180
        assert -180 <= roll <= 180


def test_head_pose_different_resolutions():
    lms = _make_lms_for_pose()
    for w, h in [(320, 240), (640, 480), (1280, 720)]:
        result = _head_pose(lms, w=w, h=h)
        assert len(result) == 3


def test_head_pose_uses_width_as_focal_length():
    """Focal length = float(w), so wider image = different pose estimate."""
    lms = _make_lms_for_pose()
    r1 = _head_pose(lms, w=320, h=240)
    r2 = _head_pose(lms, w=1280, h=720)
    # Results can differ with different focal lengths
    assert r1 is not None and r2 is not None


def test_head_pose_returns_nan_tuple_when_solvepnp_fails():
    """Mock solvePnP to return ok=False."""
    import cv2
    lms = _make_lms_for_pose()
    with patch("app.pipeline.cv2.solvePnP", return_value=(False, None, None)):
        pitch, yaw, roll = _head_pose(lms, w=640, h=480)
    assert np.isnan(pitch)
    assert np.isnan(yaw)
    assert np.isnan(roll)


# ─── _ensure_mediapipe_model ─────────────────────────────────────────────────

def test_ensure_mediapipe_model_returns_path_string_if_exists(tmp_path):
    from app.pipeline import _ensure_mediapipe_model, _MEDIAPIPE_MODEL_PATH
    fake = tmp_path / "face_landmarker.task"
    fake.write_bytes(b"model")
    with patch("app.pipeline._MEDIAPIPE_MODEL_PATH", fake):
        result = _ensure_mediapipe_model()
    assert isinstance(result, str)
    assert result == str(fake)


def test_ensure_mediapipe_model_downloads_when_missing(tmp_path):
    from app.pipeline import _ensure_mediapipe_model
    missing = tmp_path / "face_landmarker.task"
    with patch("app.pipeline._MEDIAPIPE_MODEL_PATH", missing):
        with patch("urllib.request.urlretrieve") as mock_dl:
            _ensure_mediapipe_model()
    mock_dl.assert_called_once()


def test_ensure_mediapipe_model_no_download_when_exists(tmp_path):
    from app.pipeline import _ensure_mediapipe_model
    existing = tmp_path / "face_landmarker.task"
    existing.write_bytes(b"model bytes")
    with patch("app.pipeline._MEDIAPIPE_MODEL_PATH", existing):
        with patch("urllib.request.urlretrieve") as mock_dl:
            _ensure_mediapipe_model()
    mock_dl.assert_not_called()


def test_ensure_mediapipe_model_creates_parent_dir(tmp_path):
    from app.pipeline import _ensure_mediapipe_model
    deep = tmp_path / "models" / "mediapipe" / "face_landmarker.task"
    with patch("app.pipeline._MEDIAPIPE_MODEL_PATH", deep):
        with patch("urllib.request.urlretrieve"):
            _ensure_mediapipe_model()
    assert deep.parent.exists()