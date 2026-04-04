"""
Tests untuk memverifikasi model weights asli yang ada di disk.
File: models/classifier/best_val_f1.pth, best_val_loss.pth, latest.pth

Tests ini akan di-skip otomatis jika file .pth tidak ditemukan,
sehingga CI/CD tetap bisa jalan tanpa model weights.
"""
import pytest
import torch
import torch.nn.functional as F
from pathlib import Path

from app.pipeline import (
    MultiScaleTCN, CLASS_NAMES, SEQUENCE_LENGTH,
    _load_model, predict,
)
from unittest.mock import patch

# ─── Path ke model asli ──────────────────────────────────────────────────────

CLASSIFIER_DIR = Path("models/classifier")
BEST_F1_PATH   = CLASSIFIER_DIR / "best_val_f1.pth"
BEST_LOSS_PATH = CLASSIFIER_DIR / "best_val_loss.pth"
LATEST_PATH    = CLASSIFIER_DIR / "latest.pth"

ALL_CHECKPOINTS = [
    pytest.param(BEST_F1_PATH,   id="best_val_f1"),
    pytest.param(BEST_LOSS_PATH, id="best_val_loss"),
    pytest.param(LATEST_PATH,    id="latest"),
]

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _skip_if_missing(path: Path):
    """Decorator: skip test jika file tidak ada di disk."""
    return pytest.mark.skipif(
        not path.exists(),
        reason=f"Model file tidak ditemukan: {path}"
    )


# ─── Test per checkpoint ─────────────────────────────────────────────────────

@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_file_exists(ckpt_path):
    """Verifikasi file .pth ada di disk."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")
    assert ckpt_path.stat().st_size > 0, f"{ckpt_path} ada tapi kosong (0 bytes)"


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_loads_without_error(ckpt_path):
    """File .pth bisa di-load oleh torch tanpa error."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt is not None


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_contains_state_dict(ckpt_path):
    """Checkpoint berisi 'model_state_dict' atau langsung berupa state dict."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Bisa berupa dict dengan key "model_state_dict", atau langsung state dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    assert isinstance(state, dict), "State dict harus berupa dict"
    assert len(state) > 0, "State dict kosong"


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_state_dict_loads_into_model(ckpt_path):
    """Weight dari .pth berhasil di-load ke arsitektur MultiScaleTCN."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    import app.pipeline as pl
    pl._model_cache.clear()

    # Tidak boleh raise exception
    model = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert model is not None


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_model_in_eval_mode(ckpt_path):
    """Model yang di-load dari .pth asli harus dalam eval mode."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    import app.pipeline as pl
    pl._model_cache.clear()

    model = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    assert not model.training, "Model harus dalam eval mode setelah load"


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_forward_pass_succeeds(ckpt_path):
    """Model asli bisa melakukan forward pass tanpa error."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    import app.pipeline as pl
    pl._model_cache.clear()

    model = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    dummy = torch.zeros(1, 8, SEQUENCE_LENGTH)

    with torch.no_grad():
        logits = model(dummy)

    assert logits.shape == (1, len(CLASS_NAMES))
    assert torch.isfinite(logits).all(), "Logits mengandung NaN atau Inf"


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_output_is_valid_probability(ckpt_path):
    """Softmax dari output model menghasilkan probabilitas yang valid (sum = 1)."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    import app.pipeline as pl
    pl._model_cache.clear()

    model = _load_model(ckpt_path, "MultiScaleTCN", torch.device("cpu"))
    dummy = torch.zeros(1, 8, SEQUENCE_LENGTH)

    with torch.no_grad():
        logits = model(dummy)
        probs  = F.softmax(logits, dim=-1)

    assert (probs >= 0).all(), "Ada probabilitas negatif"
    assert (probs <= 1).all(), "Ada probabilitas > 1"
    assert abs(probs.sum().item() - 1.0) < 1e-5, "Probabilitas tidak sum ke 1"


@pytest.mark.parametrize("ckpt_path", ALL_CHECKPOINTS)
def test_real_checkpoint_predicts_valid_class(ckpt_path):
    """predict() dengan model asli mengembalikan label yang valid."""
    if not ckpt_path.exists():
        pytest.skip(f"File tidak ditemukan: {ckpt_path}")

    import app.pipeline as pl
    pl._model_cache.clear()

    with patch("app.pipeline.torch.cuda.is_available", return_value=False):
        with patch("app.pipeline._extract",
                   return_value=torch.zeros(SEQUENCE_LENGTH, 8).numpy()):
            result = predict(
                Path("dummy_video.mp4"),
                ckpt_path,
                "MultiScaleTCN",
            )

    assert result["label"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    assert CLASS_NAMES[result["class_id"]] == result["label"]


# ─── Test khusus best_val_f1 (model utama) ───────────────────────────────────

@_skip_if_missing(BEST_F1_PATH)
def test_best_val_f1_metadata_epoch():
    """Checkpoint best_val_f1 menyimpan metadata epoch."""
    ckpt = torch.load(BEST_F1_PATH, map_location="cpu", weights_only=False)
    assert "epoch" in ckpt, "Checkpoint harus menyimpan key 'epoch'"
    assert isinstance(ckpt["epoch"], int)
    assert ckpt["epoch"] > 0


@_skip_if_missing(BEST_F1_PATH)
def test_best_val_f1_metadata_val_f1():
    """Checkpoint best_val_f1 menyimpan nilai val_f1."""
    ckpt = torch.load(BEST_F1_PATH, map_location="cpu", weights_only=False)
    assert "val_f1" in ckpt, "Checkpoint harus menyimpan key 'val_f1'"
    val_f1 = ckpt["val_f1"]
    assert 0.0 <= val_f1 <= 1.0, f"val_f1={val_f1} di luar range [0, 1]"


@_skip_if_missing(BEST_F1_PATH)
def test_best_val_f1_is_reasonable():
    """Val F1 dari model terbaik harus di atas threshold minimum yang wajar."""
    ckpt = torch.load(BEST_F1_PATH, map_location="cpu", weights_only=False)
    if "val_f1" in ckpt:
        assert ckpt["val_f1"] > 0.5, (
            f"Val F1 terlalu rendah: {ckpt['val_f1']:.4f}. "
            "Model mungkin belum cukup terlatih."
        )


@_skip_if_missing(BEST_LOSS_PATH)
def test_best_val_loss_metadata():
    """Checkpoint best_val_loss menyimpan nilai val_loss."""
    ckpt = torch.load(BEST_LOSS_PATH, map_location="cpu", weights_only=False)
    assert "val_loss" in ckpt, "Checkpoint harus menyimpan key 'val_loss'"
    assert ckpt["val_loss"] > 0.0, "val_loss harus positif"


# ─── Test konsistensi antar checkpoint ───────────────────────────────────────

@pytest.mark.skipif(
    not (BEST_F1_PATH.exists() and BEST_LOSS_PATH.exists()),
    reason="Butuh kedua file: best_val_f1.pth dan best_val_loss.pth"
)
def test_best_f1_and_best_loss_have_same_architecture():
    """Kedua checkpoint harus memiliki jumlah parameter yang sama (arsitektur identik)."""
    import app.pipeline as pl

    pl._model_cache.clear()
    m1 = _load_model(BEST_F1_PATH,   "MultiScaleTCN", torch.device("cpu"))
    pl._model_cache.clear()
    m2 = _load_model(BEST_LOSS_PATH, "MultiScaleTCN", torch.device("cpu"))

    p1 = sum(p.numel() for p in m1.parameters())
    p2 = sum(p.numel() for p in m2.parameters())
    assert p1 == p2, (
        f"Jumlah parameter berbeda: best_val_f1={p1:,}, best_val_loss={p2:,}. "
        "Kemungkinan dilatih dengan arsitektur berbeda."
    )


@pytest.mark.skipif(
    not (BEST_F1_PATH.exists() and BEST_LOSS_PATH.exists()),
    reason="Butuh kedua file: best_val_f1.pth dan best_val_loss.pth"
)
def test_best_f1_and_best_loss_produce_different_predictions():
    """Dua checkpoint yang berbeda seharusnya memiliki weights yang berbeda."""
    import app.pipeline as pl
    dummy = torch.zeros(1, 8, SEQUENCE_LENGTH)

    pl._model_cache.clear()
    m1 = _load_model(BEST_F1_PATH,   "MultiScaleTCN", torch.device("cpu"))
    pl._model_cache.clear()
    m2 = _load_model(BEST_LOSS_PATH, "MultiScaleTCN", torch.device("cpu"))

    with torch.no_grad():
        out1 = m1(dummy)
        out2 = m2(dummy)

    # Jika weights sama persis, kemungkinan ada bug di checkpointing
    assert not torch.allclose(out1, out2), (
        "Output kedua checkpoint identik — kemungkinan weights tidak tersimpan dengan benar."
    )