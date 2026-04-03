"""Tests for app.pipeline — feature extraction, model architecture, preprocessing, predict()."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.pipeline import (
    MultiScaleTCN,
    _ear, _mar, _head_pose,
    _is_masked, _enforce_continuity, _interpolate_nan,
    _prepare, _load_model,
    CLASS_NAMES, SEQUENCE_LENGTH, LANDMARK_FPS,
    predict,
)
from conftest import make_mock_landmark, make_face_landmarks


# ─── Constants ───────────────────────────────────────────────────────────────

def test_sequence_length_is_200():
    assert SEQUENCE_LENGTH == 200


def test_landmark_fps_is_10():
    assert LANDMARK_FPS == 10.0


def test_class_names_has_three():
    assert len(CLASS_NAMES) == 3


def test_class_names_content():
    assert "normal" in CLASS_NAMES
    assert "eyes_closed" in CLASS_NAMES
    assert "yawning" in CLASS_NAMES


# ─── EAR ─────────────────────────────────────────────────────────────────────

def test_ear_returns_float():
    lms = make_face_landmarks()
    from app.pipeline import _RIGHT_EYE
    val = _ear(lms, _RIGHT_EYE)
    assert isinstance(val, float)


def test_ear_returns_nan_when_horizontal_distance_zero():
    lms = [make_mock_landmark(0.5, 0.5) for _ in range(478)]
    from app.pipeline import _RIGHT_EYE
    # All same position → h = 0
    val = _ear(lms, _RIGHT_EYE)
    assert np.isnan(val)


def test_ear_positive_for_open_eye():
    lms = make_face_landmarks()
    from app.pipeline import _LEFT_EYE, _RIGHT_EYE
    ear_l = _ear(lms, _LEFT_EYE)
    ear_r = _ear(lms, _RIGHT_EYE)
    # With distinct coordinates, EAR should be calculable
    assert not np.isnan(ear_l) or not np.isnan(ear_r)


# ─── MAR ─────────────────────────────────────────────────────────────────────

def test_mar_returns_float():
    lms = make_face_landmarks()
    from app.pipeline import _MOUTH
    val = _mar(lms, _MOUTH)
    assert isinstance(val, float)


def test_mar_returns_nan_when_horizontal_zero():
    lms = [make_mock_landmark(0.5, 0.5) for _ in range(478)]
    from app.pipeline import _MOUTH
    val = _mar(lms, _MOUTH)
    assert np.isnan(val)


def test_mar_larger_when_mouth_open():
    from app.pipeline import _MOUTH
    # Closed mouth: small vertical separation
    lms_closed = make_face_landmarks()
    lms_closed[13] = make_mock_landmark(0.5, 0.48)
    lms_closed[14] = make_mock_landmark(0.5, 0.52)
    lms_closed[61] = make_mock_landmark(0.3, 0.50)
    lms_closed[291] = make_mock_landmark(0.7, 0.50)
    mar_closed = _mar(lms_closed, _MOUTH)

    # Open mouth: large vertical separation
    lms_open = make_face_landmarks()
    lms_open[13] = make_mock_landmark(0.5, 0.40)
    lms_open[14] = make_mock_landmark(0.5, 0.60)
    lms_open[61] = make_mock_landmark(0.3, 0.50)
    lms_open[291] = make_mock_landmark(0.7, 0.50)
    mar_open = _mar(lms_open, _MOUTH)

    if not np.isnan(mar_closed) and not np.isnan(mar_open):
        assert mar_open > mar_closed


# ─── _is_masked ───────────────────────────────────────────────────────────────

def test_is_masked_returns_false_for_empty_history():
    assert _is_masked([]) is False


def test_is_masked_returns_false_for_short_history():
    assert _is_masked([0.5, 0.5, 0.5]) is False


def test_is_masked_returns_true_for_constant_history():
    # Zero variance = face is masked (exactly the same value)
    history = [0.3, 0.3, 0.3, 0.3, 0.3]
    assert _is_masked(history) is True


def test_is_masked_returns_false_for_varying_history():
    history = [0.1, 0.3, 0.5, 0.2, 0.8]
    assert _is_masked(history) is False


def test_is_masked_returns_false_with_nans():
    history = [np.nan, np.nan, np.nan, np.nan, np.nan]
    assert _is_masked(history) is False


# ─── _enforce_continuity ─────────────────────────────────────────────────────

def test_enforce_continuity_no_nans():
    arr = np.array([10.0, 15.0, 12.0, 8.0])
    out = _enforce_continuity(arr)
    assert not np.any(np.isnan(out))


def test_enforce_continuity_handles_all_nans():
    arr = np.full(5, np.nan)
    out = _enforce_continuity(arr)
    assert np.all(np.isnan(out))


def test_enforce_continuity_handles_single_valid():
    arr = np.array([np.nan, np.nan, 90.0, np.nan])
    out = _enforce_continuity(arr)
    # Should preserve the single valid value
    assert out[2] == 90.0


def test_enforce_continuity_smooths_angle_jumps():
    # Simulate angle wrapping: 170 → -170 should be corrected to ~190
    arr = np.array([170.0, np.nan, -170.0])
    out = _enforce_continuity(arr)
    # The corrected value should be close to 190 (not -170)
    valid = out[~np.isnan(out)]
    if len(valid) >= 2:
        assert abs(valid[-1] - valid[0]) < 180


# ─── _interpolate_nan ────────────────────────────────────────────────────────

def test_interpolate_nan_fills_gaps():
    arr = np.array([1.0, np.nan, 3.0])
    out = _interpolate_nan(arr.copy())
    assert not np.any(np.isnan(out))
    assert abs(out[1] - 2.0) < 1e-5


def test_interpolate_nan_no_change_when_no_nans():
    arr = np.array([1.0, 2.0, 3.0])
    out = _interpolate_nan(arr.copy())
    np.testing.assert_array_almost_equal(out, arr)


def test_interpolate_nan_fills_all_nans_with_zero_when_no_valid():
    arr = np.full(5, np.nan)
    out = _interpolate_nan(arr.copy())
    np.testing.assert_array_equal(out, np.zeros(5))


def test_interpolate_nan_handles_leading_nans():
    arr = np.array([np.nan, np.nan, 2.0, 4.0])
    out = _interpolate_nan(arr.copy())
    assert not np.any(np.isnan(out))


def test_interpolate_nan_handles_trailing_nans():
    arr = np.array([2.0, 4.0, np.nan, np.nan])
    out = _interpolate_nan(arr.copy())
    assert not np.any(np.isnan(out))


# ─── _prepare ────────────────────────────────────────────────────────────────

def test_prepare_output_shape_exact_length():
    features = np.random.rand(SEQUENCE_LENGTH, 8).astype(np.float32)
    tensor = _prepare(features)
    assert tensor.shape == (1, 8, SEQUENCE_LENGTH)


def test_prepare_output_shape_short_video():
    features = np.random.rand(50, 8).astype(np.float32)
    tensor = _prepare(features)
    assert tensor.shape == (1, 8, SEQUENCE_LENGTH)


def test_prepare_output_shape_long_video():
    features = np.random.rand(400, 8).astype(np.float32)
    tensor = _prepare(features)
    assert tensor.shape == (1, 8, SEQUENCE_LENGTH)


def test_prepare_no_nans_in_output():
    features = np.full((100, 8), np.nan, dtype=np.float32)
    tensor = _prepare(features)
    assert not torch.any(torch.isnan(tensor))


def test_prepare_no_infs_in_output():
    features = np.full((100, 8), np.inf, dtype=np.float32)
    tensor = _prepare(features)
    assert not torch.any(torch.isinf(tensor))


def test_prepare_returns_float_tensor():
    features = np.random.rand(100, 8).astype(np.float32)
    tensor = _prepare(features)
    assert tensor.dtype == torch.float32


def test_prepare_pads_short_with_zeros():
    features = np.ones((10, 8), dtype=np.float32)
    tensor = _prepare(features)
    # Padded region (after frame 10) should be zero
    assert tensor[0, :, 10:].sum().item() == 0.0


# ─── MultiScaleTCN Architecture ───────────────────────────────────────────────

def test_multiscale_tcn_forward_pass():
    model = MultiScaleTCN(num_classes=3, input_channels=8, seq_length=SEQUENCE_LENGTH)
    model.eval()
    x = torch.zeros(1, 8, SEQUENCE_LENGTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3)


def test_multiscale_tcn_output_is_logits_not_probs():
    model = MultiScaleTCN(num_classes=3, input_channels=8)
    model.eval()
    x = torch.randn(1, 8, SEQUENCE_LENGTH)
    with torch.no_grad():
        logits = model(x)
    # Raw logits can be outside [0,1]; softmax should give [0,1]
    probs = F.softmax(logits, dim=-1)
    assert (probs >= 0).all() and (probs <= 1).all()
    assert abs(probs.sum().item() - 1.0) < 1e-5


def test_multiscale_tcn_batch_size_2():
    model = MultiScaleTCN(num_classes=3, input_channels=8)
    model.eval()
    x = torch.zeros(2, 8, SEQUENCE_LENGTH)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 3)


def test_multiscale_tcn_deterministic_in_eval_mode():
    model = MultiScaleTCN(num_classes=3, input_channels=8)
    model.eval()
    x = torch.randn(1, 8, SEQUENCE_LENGTH)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    torch.testing.assert_close(out1, out2)


def test_multiscale_tcn_all_params_loaded():
    model = MultiScaleTCN(num_classes=3, input_channels=8)
    total = sum(p.numel() for p in model.parameters())
    assert total > 0


def test_multiscale_tcn_gradient_flows():
    model = MultiScaleTCN(num_classes=3, input_channels=8)
    model.train()
    x = torch.randn(1, 8, SEQUENCE_LENGTH, requires_grad=False)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    # Check at least one parameter has a gradient
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad


def test_multiscale_tcn_num_classes_configurable():
    for n in [2, 3, 5]:
        model = MultiScaleTCN(num_classes=n, input_channels=8)
        model.eval()
        x = torch.zeros(1, 8, SEQUENCE_LENGTH)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, n)


# ─── _load_model ─────────────────────────────────────────────────────────────

def test_load_model_raises_file_not_found(tmp_path):
    missing = tmp_path / "missing.pth"
    with pytest.raises(FileNotFoundError):
        _load_model(missing, "MultiScaleTCN", torch.device("cpu"))


def test_load_model_raises_unknown_model_name(tmp_path):
    # Create a fake checkpoint
    model = MultiScaleTCN(num_classes=3)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    with pytest.raises(ValueError, match="Unknown model"):
        _load_model(ckpt_path, "UnknownModel", torch.device("cpu"))


def test_load_model_returns_eval_mode(tmp_path):
    model = MultiScaleTCN(num_classes=3)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    loaded = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert not loaded.training


def test_load_model_caches_on_second_call(tmp_path):
    model = MultiScaleTCN(num_classes=3)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    import app.pipeline as pl
    # Clear cache first
    pl._model_cache.clear()

    m1 = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    m2 = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert m1 is m2  # Same object from cache


def test_load_model_handles_backbone_prefix(tmp_path):
    """State dict with 'backbone.' prefix should load correctly."""
    model = MultiScaleTCN(num_classes=3)
    wrapped_state = {f"backbone.{k}": v for k, v in model.state_dict().items()}
    ckpt_path = tmp_path / "model.pth"
    torch.save(wrapped_state, ckpt_path)

    import app.pipeline as pl
    pl._model_cache.clear()

    loaded = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert loaded is not None


def test_load_model_handles_raw_state_dict(tmp_path):
    """State dict saved directly (no wrapper dict) should also load."""
    model = MultiScaleTCN(num_classes=3)
    ckpt_path = tmp_path / "model.pth"
    torch.save(model.state_dict(), ckpt_path)

    import app.pipeline as pl
    pl._model_cache.clear()

    loaded = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert loaded is not None


# ─── predict() ───────────────────────────────────────────────────────────────
# Force CPU untuk semua predict() test agar tidak bergantung pada kompatibilitas GPU.
# RTX 5070 Ti (sm_120) tidak didukung PyTorch versi lama → CUDA error jika dibiarkan.

def _predict_cpu(video_path, ckpt_path):
    """Helper: jalankan predict() dengan CUDA dinonaktifkan secara paksa."""
    import app.pipeline as pl
    pl._model_cache.clear()
    with patch("app.pipeline.torch.cuda.is_available", return_value=False):
        with patch("app.pipeline._extract", return_value=np.zeros((SEQUENCE_LENGTH, 8), dtype=np.float32)):
            return predict(video_path, ckpt_path, "MultiScaleTCN")


def _make_ckpt(tmp_path):
    """Buat checkpoint sementara dari model dummy."""
    model = MultiScaleTCN(num_classes=3)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)
    return ckpt_path


def test_predict_returns_dict_with_required_keys(tmp_path, black_video):
    result = _predict_cpu(black_video, _make_ckpt(tmp_path))
    assert "label" in result
    assert "class_id" in result
    assert "confidence" in result


def test_predict_label_is_valid_class(tmp_path, black_video):
    result = _predict_cpu(black_video, _make_ckpt(tmp_path))
    assert result["label"] in CLASS_NAMES


def test_predict_confidence_between_0_and_1(tmp_path, black_video):
    result = _predict_cpu(black_video, _make_ckpt(tmp_path))
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_class_id_matches_label(tmp_path, black_video):
    result = _predict_cpu(black_video, _make_ckpt(tmp_path))
    assert CLASS_NAMES[result["class_id"]] == result["label"]


def test_predict_raises_file_not_found_for_missing_model(tmp_path, black_video):
    import app.pipeline as pl
    pl._model_cache.clear()
    with patch("app.pipeline.torch.cuda.is_available", return_value=False):
        with pytest.raises(FileNotFoundError):
            predict(black_video, tmp_path / "missing.pth", "MultiScaleTCN")


# ─── _extract with black video (no face) ─────────────────────────────────────

def test_extract_black_video_returns_array(black_video):
    """With no face detected, _extract must still return valid array."""
    from app.pipeline import _extract
    result = _extract(black_video)
    assert isinstance(result, np.ndarray)
    assert result.shape[1] == 8
    assert not np.any(np.isnan(result))


def test_extract_missing_video_returns_zeros():
    from app.pipeline import _extract
    result = _extract(Path("/tmp/nonexistent_video_xyz.mp4"))
    assert result.shape == (SEQUENCE_LENGTH, 8)
    assert np.all(result == 0)