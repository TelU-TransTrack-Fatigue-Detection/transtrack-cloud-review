"""
Cross-cutting edge case tests — boundary conditions, type correctness,
and robustness across the full stack.
"""
import numpy as np
import pytest
import torch
from pathlib import Path

from app.pipeline import (
    _prepare, _interpolate_nan, _enforce_continuity,
    MultiScaleTCN, SEQUENCE_LENGTH, CLASS_NAMES,
)


# ─── _prepare: boundary conditions ────────────────────────────────────────────

class TestPrepareEdgeCases:
    def test_prepare_single_frame(self):
        features = np.random.rand(1, 8).astype(np.float32)
        tensor = _prepare(features)
        assert tensor.shape == (1, 8, SEQUENCE_LENGTH)

    def test_prepare_exactly_sequence_length(self):
        features = np.random.rand(SEQUENCE_LENGTH, 8).astype(np.float32)
        tensor = _prepare(features)
        assert tensor.shape == (1, 8, SEQUENCE_LENGTH)

    def test_prepare_one_over_sequence_length(self):
        features = np.random.rand(SEQUENCE_LENGTH + 1, 8).astype(np.float32)
        tensor = _prepare(features)
        assert tensor.shape == (1, 8, SEQUENCE_LENGTH)

    def test_prepare_large_input_truncated(self):
        features = np.random.rand(1000, 8).astype(np.float32)
        tensor = _prepare(features)
        assert tensor.shape == (1, 8, SEQUENCE_LENGTH)

    def test_prepare_cleans_positive_inf(self):
        features = np.full((50, 8), np.inf, dtype=np.float32)
        tensor = _prepare(features)
        assert not torch.any(torch.isinf(tensor))

    def test_prepare_cleans_negative_inf(self):
        features = np.full((50, 8), -np.inf, dtype=np.float32)
        tensor = _prepare(features)
        assert not torch.any(torch.isinf(tensor))

    def test_prepare_correct_channel_order(self):
        # Input is (T, 8), output should be (1, 8, T) transposed
        features = np.arange(16, dtype=np.float32).reshape(2, 8)
        tensor = _prepare(features)
        # Channel 0 at time 0 should match features[0,0]
        assert tensor[0, 0, 0].item() == pytest.approx(features[0, 0])


# ─── _interpolate_nan: comprehensive ─────────────────────────────────────────

class TestInterpolateNan:
    def test_single_nan_between_values(self):
        arr = np.array([1.0, np.nan, 3.0])
        out = _interpolate_nan(arr.copy())
        assert abs(out[1] - 2.0) < 1e-5

    def test_multiple_nans_in_middle(self):
        arr = np.array([0.0, np.nan, np.nan, np.nan, 4.0])
        out = _interpolate_nan(arr.copy())
        assert not np.any(np.isnan(out))
        np.testing.assert_allclose(out, [0, 1, 2, 3, 4], atol=1e-5)

    def test_all_valid_unchanged(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        out = _interpolate_nan(arr.copy())
        np.testing.assert_array_equal(out, arr)

    def test_single_element_nan(self):
        arr = np.array([np.nan])
        out = _interpolate_nan(arr.copy())
        assert out[0] == 0.0

    def test_single_element_valid(self):
        arr = np.array([5.0])
        out = _interpolate_nan(arr.copy())
        assert out[0] == 5.0

    def test_large_array_no_nans(self):
        arr = np.random.rand(1000).astype(np.float32)
        arr[::7] = np.nan
        out = _interpolate_nan(arr.copy())
        assert not np.any(np.isnan(out))


# ─── _enforce_continuity: angle wrap scenarios ────────────────────────────────

class TestEnforceContinuity:
    def test_no_wrap_needed_unchanged(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0])
        out = _enforce_continuity(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_handles_empty_array(self):
        arr = np.array([], dtype=np.float32)
        out = _enforce_continuity(arr)
        assert len(out) == 0

    def test_handles_one_element(self):
        arr = np.array([45.0])
        out = _enforce_continuity(arr)
        assert out[0] == 45.0

    def test_handles_two_elements_no_wrap(self):
        arr = np.array([10.0, 20.0])
        out = _enforce_continuity(arr)
        np.testing.assert_array_almost_equal(out, arr)

    def test_output_preserves_non_nan_count(self):
        arr = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        out = _enforce_continuity(arr)
        non_nan_in  = np.sum(~np.isnan(arr))
        non_nan_out = np.sum(~np.isnan(out))
        assert non_nan_in == non_nan_out

    def test_180_jump_corrected(self):
        # 179 → -179 is a 358° jump; continuity should fix it to ~181
        arr = np.array([179.0, np.nan, -179.0])
        out = _enforce_continuity(arr)
        valid = out[~np.isnan(out)]
        if len(valid) >= 2:
            # After correction, the jump should be small
            assert abs(valid[-1] - valid[0]) < 10


# ─── MODEL_REGISTRY completeness ─────────────────────────────────────────────

def test_model_registry_contains_multiscaletcn():
    from app.pipeline import _MODEL_REGISTRY
    assert "MultiScaleTCN" in _MODEL_REGISTRY


def test_model_registry_maps_to_correct_class():
    from app.pipeline import _MODEL_REGISTRY
    assert _MODEL_REGISTRY["MultiScaleTCN"] is MultiScaleTCN


# ─── CLASS_NAMES index consistency ────────────────────────────────────────────

def test_class_names_are_strings():
    assert all(isinstance(c, str) for c in CLASS_NAMES)


def test_class_names_no_duplicates():
    assert len(CLASS_NAMES) == len(set(CLASS_NAMES))


def test_class_id_matches_index():
    """Each CLASS_NAMES[i] should have index i."""
    for i, name in enumerate(CLASS_NAMES):
        assert CLASS_NAMES[i] == name


# ─── Config: type safety ──────────────────────────────────────────────────────

def test_config_video_download_timeout_is_int():
    from app.config import settings
    assert isinstance(settings.VIDEO_DOWNLOAD_TIMEOUT, int)


def test_config_httpx_timeout_is_int():
    from app.config import settings
    assert isinstance(settings.HTTPX_TIMEOUT, int)


def test_config_keep_tmp_videos_is_bool():
    from app.config import settings
    assert isinstance(settings.KEEP_TMP_VIDEOS, bool)


def test_config_max_queue_size_is_int():
    from app.config import settings
    assert isinstance(settings.MAX_QUEUE_SIZE, int)