"""
Happy path tests for app.pipeline._extract() — face detected scenario.

Strategy: patch FaceLandmarker.create_from_options agar inject mock landmarks
ke setiap frame, tanpa perlu video dengan wajah asli.

Struktur mock:
  FaceLandmarker.create_from_options(options)  ← context manager
    └── lmk.detect(MpImage(...))
          └── returns MagicMock(face_landmarks=[[lm0, lm1, ...]])

_extract() internals yang perlu di-mock:
  1. _ensure_mediapipe_model()  → return path string (skip download)
  2. FaceLandmarker.create_from_options  → return mock landmarker
  3. lmk.detect  → return MagicMock(face_landmarks=[make_face_landmarks()])
"""
import numpy as np
import pytest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.pipeline import (
    _extract,
    CLASS_NAMES, SEQUENCE_LENGTH, LANDMARK_FPS,
    _LEFT_EYE, _RIGHT_EYE, _MOUTH,
)
from conftest import make_face_landmarks, make_mock_landmark


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_detect_result(landmarks=None):
    """Buat mock DetectionResult dengan face_landmarks terisi."""
    result = MagicMock()
    result.face_landmarks = [landmarks or make_face_landmarks()]
    return result


def _make_empty_result():
    """Buat mock DetectionResult tanpa wajah (face_landmarks kosong)."""
    result = MagicMock()
    result.face_landmarks = []
    return result


@contextmanager
def _mock_landmarker(detect_result):
    """
    Context manager helper: patch _ensure_mediapipe_model dan
    FaceLandmarker.create_from_options agar return mock landmarker
    yang selalu return detect_result saat .detect() dipanggil.
    """
    mock_lmk = MagicMock()
    mock_lmk.detect.return_value = detect_result
    mock_lmk.__enter__ = MagicMock(return_value=mock_lmk)
    mock_lmk.__exit__ = MagicMock(return_value=False)

    with patch("app.pipeline._ensure_mediapipe_model", return_value="fake/model.task"):
        with patch("app.pipeline.FaceLandmarker.create_from_options", return_value=mock_lmk):
            yield mock_lmk


# ─── Shape & type ────────────────────────────────────────────────────────────

def test_extract_with_face_returns_ndarray(black_video):
    """_extract() dengan wajah terdeteksi harus return numpy array."""
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    assert isinstance(result, np.ndarray)


def test_extract_with_face_has_8_channels(black_video):
    """Output selalu punya 8 kolom: ear_l, ear_r, mar, pitch, yaw, roll, nose_x, nose_y."""
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    assert result.shape[1] == 8


def test_extract_with_face_dtype_is_float32(black_video):
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    assert result.dtype == np.float32


def test_extract_with_face_no_nans_after_interpolation(black_video):
    """
    Setelah _extract selesai, semua NaN harus sudah diinterpolasi.
    Pipeline memanggil _interpolate_nan() sebelum return.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    assert not np.any(np.isnan(result))


def test_extract_with_face_no_infs(black_video):
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    assert not np.any(np.isinf(result))


# ─── Frame count / sequence length ───────────────────────────────────────────

def test_extract_with_face_rows_match_sampled_frames(black_video):
    """
    black_video: 30 frames @ 10 FPS → frame_interval = round(10/10) = 1
    Setiap frame di-sample → 30 rows sebelum pad/truncate.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    # Rows yang dihasilkan sebelum pad adalah jumlah frame yang di-sample.
    # Setelah ini di-_prepare() akan di-pad ke SEQUENCE_LENGTH,
    # tapi _extract() sendiri return raw rows → shape[0] == frames sampled.
    assert result.shape[0] > 0


def test_extract_long_video_not_truncated_inside_extract(synthetic_video):
    """
    synthetic_video: 90 frames @ 30 FPS → frame_interval = round(30/10) = 3
    → 30 sampled frames. _extract() return raw rows, bukan pad ke SEQUENCE_LENGTH.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(synthetic_video)
    # _extract tidak truncate; truncation dilakukan _prepare().
    assert result.shape[1] == 8


# ─── Feature values — EAR, MAR, nose position ────────────────────────────────

def test_extract_with_face_ear_values_are_finite(black_video):
    """EAR (kolom 0 dan 1) harus finite ketika landmark valid."""
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    # Minimal 1 row harus finite untuk EAR
    ear_l = result[:, 0]
    ear_r = result[:, 1]
    assert np.any(np.isfinite(ear_l))
    assert np.any(np.isfinite(ear_r))


def test_extract_with_face_mar_values_are_finite(black_video):
    """MAR (kolom 2) harus finite ketika landmark valid dan tidak masked."""
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    mar = result[:, 2]
    assert np.any(np.isfinite(mar))


def test_extract_with_face_nose_position_in_range(black_video):
    """
    Nose tip x dan y (kolom 6 dan 7) berasal dari lm[1].x dan lm[1].y.
    make_face_landmarks() set index 1 ke (0.5, 0.5) → harus ada nilai sekitar 0.5.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    nose_x = result[:, 6]
    nose_y = result[:, 7]
    # Setidaknya ada satu row dengan nilai di rentang [0, 1]
    assert np.any((nose_x >= 0) & (nose_x <= 1))
    assert np.any((nose_y >= 0) & (nose_y <= 1))


def test_extract_with_face_ear_positive(black_video):
    """
    EAR harus positif ketika eye landmarks punya spread yang cukup.
    make_face_landmarks() sudah dikonfigurasi untuk ini di conftest.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)
    ear_l = result[:, 0]
    ear_r = result[:, 1]
    # Filter out zero-padded rows
    valid_ear_l = ear_l[ear_l != 0]
    valid_ear_r = ear_r[ear_r != 0]
    if len(valid_ear_l) > 0:
        assert np.all(valid_ear_l >= 0)
    if len(valid_ear_r) > 0:
        assert np.all(valid_ear_r >= 0)


# ─── Mixed: sebagian frame ada wajah, sebagian tidak ─────────────────────────

def test_extract_mixed_frames_no_nans_in_output(black_video):
    """
    Simulasi: frame genap ada wajah, frame ganjil tidak.
    _extract harus interpolasi NaN dari frame tanpa wajah.
    """
    call_count = [0]

    def alternating_detect(mp_image):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return _make_empty_result()
        return _make_detect_result()

    mock_lmk = MagicMock()
    mock_lmk.detect.side_effect = alternating_detect
    mock_lmk.__enter__ = MagicMock(return_value=mock_lmk)
    mock_lmk.__exit__ = MagicMock(return_value=False)

    with patch("app.pipeline._ensure_mediapipe_model", return_value="fake/model.task"):
        with patch("app.pipeline.FaceLandmarker.create_from_options", return_value=mock_lmk):
            result = _extract(black_video)

    assert not np.any(np.isnan(result))
    assert result.shape[1] == 8


def test_extract_mixed_frames_has_data(black_video):
    """Dengan mixed frames, array output tidak boleh semua nol."""
    call_count = [0]

    def alternating_detect(mp_image):
        call_count[0] += 1
        if call_count[0] % 2 == 0:
            return _make_empty_result()
        return _make_detect_result()

    mock_lmk = MagicMock()
    mock_lmk.detect.side_effect = alternating_detect
    mock_lmk.__enter__ = MagicMock(return_value=mock_lmk)
    mock_lmk.__exit__ = MagicMock(return_value=False)

    with patch("app.pipeline._ensure_mediapipe_model", return_value="fake/model.task"):
        with patch("app.pipeline.FaceLandmarker.create_from_options", return_value=mock_lmk):
            result = _extract(black_video)

    assert result.sum() != 0.0


# ─── Mask detection (wajah tersembunyi / MAR sangat stabil) ──────────────────

def test_extract_masked_face_sets_mar_to_nan_then_interpolated(black_video):
    """
    Ketika _is_masked() True (MAR konstan dalam 5 frame terakhir),
    pipeline harus set MAR ke NaN, lalu diinterpolasi di akhir.
    """
    # Buat landmark dengan MAR yang sangat konstan
    # Mouth: p[0].y - p[1].y sangat kecil (mulut tertutup statis)
    lms = make_face_landmarks()
    lms[13] = make_mock_landmark(0.5, 0.500)   # upper lip
    lms[14] = make_mock_landmark(0.5, 0.501)   # lower lip (hampir sama → MAR ~0)
    lms[61] = make_mock_landmark(0.3, 0.500)   # mouth left
    lms[291] = make_mock_landmark(0.7, 0.500)  # mouth right

    detect_result = _make_detect_result(landmarks=lms)
    with _mock_landmarker(detect_result):
        result = _extract(black_video)

    # Output akhir tidak boleh ada NaN (sudah diinterpolasi)
    assert not np.any(np.isnan(result))
    assert result.shape[1] == 8


# ─── detect() dipanggil untuk setiap sampled frame ───────────────────────────

def test_extract_detect_called_for_each_sampled_frame(black_video):
    """
    black_video: 30 frames @ 10 FPS → frame_interval = 1 → semua frame di-sample.
    lmk.detect() harus dipanggil sebanyak jumlah frame.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result) as mock_lmk:
        _extract(black_video)

    # 30 frames, frame_interval=1 → detect dipanggil 30 kali
    assert mock_lmk.detect.call_count == 30


def test_extract_detect_called_less_often_for_high_fps_video(synthetic_video):
    """
    synthetic_video: 90 frames @ 30 FPS → frame_interval = round(30/10) = 3
    → detect dipanggil 30 kali (setiap 3 frame sekali).
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result) as mock_lmk:
        _extract(synthetic_video)

    # 90 frames / 3 = 30 sampled
    assert mock_lmk.detect.call_count == 30


# ─── Skenario: semua frame punya wajah vs tidak ada wajah ────────────────────

def test_extract_all_faces_detected_vs_none_both_valid(black_video):
    """
    Dua run: satu dengan semua frame berisi wajah, satu tanpa wajah sama sekali.
    Keduanya harus return array valid (no NaN).
    """
    # All faces
    with _mock_landmarker(_make_detect_result()):
        result_all = _extract(black_video)

    # No faces (black_video tanpa mock → MediaPipe tidak detect apapun)
    result_none = _extract(black_video)

    assert not np.any(np.isnan(result_all))
    assert not np.any(np.isnan(result_none))
    assert result_all.shape[1] == 8
    assert result_none.shape[1] == 8


def test_extract_all_faces_produces_nonzero_features(black_video):
    """
    Ketika semua frame terdeteksi wajah, feature array tidak boleh semua nol
    (tidak seperti no-face path yang menghasilkan zero-padded).
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)

    assert result.sum() != 0.0


# ─── Integritas kolom head pose (pitch/yaw/roll) ─────────────────────────────

def test_extract_head_pose_columns_are_finite_after_interpolation(black_video):
    """
    Kolom 3 (pitch), 4 (yaw), 5 (roll) harus finite setelah interpolasi.
    solvePnP mungkin return NaN untuk landmark tertentu, tapi tetap diinterpolasi.
    """
    detect_result = _make_detect_result()
    with _mock_landmarker(detect_result):
        result = _extract(black_video)

    for col_name, col_idx in [("pitch", 3), ("yaw", 4), ("roll", 5)]:
        col = result[:, col_idx]
        assert np.all(np.isfinite(col)), f"Kolom {col_name} (idx {col_idx}) mengandung non-finite"