import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from app.pipeline import MultiScaleTCN, SEQUENCE_LENGTH, CLASS_NAMES

WEIGHTS = {
    "best_val_f1":   Path("models/classifier/best_val_f1.pth"),
    "best_val_loss": Path("models/classifier/best_val_loss.pth"),
    "latest":        Path("models/classifier/latest.pth"),
}

INPUT_CHANNELS = 8
NUM_CLASSES    = 3
DROPOUT        = 0.3


def check_weight(name: str, path: Path, device: torch.device):
    print(f"\n{'='*50}")
    print(f"Checking: {name} → {path}")

    if not path.exists():
        print(f"  [MISSING] File not found")
        return

    ckpt  = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

    model = MultiScaleTCN(
        num_classes    = NUM_CLASSES,
        input_channels = INPUT_CHANNELS,
        seq_length     = SEQUENCE_LENGTH,
        dropout        = DROPOUT,
    )

    try:
        model.load_state_dict(state)
        print(f"  [OK] Weights loaded successfully")
    except RuntimeError as e:
        print(f"  [FAIL] load_state_dict error:\n  {e}")
        return

    model.to(device).eval()

    dummy  = torch.zeros(1, INPUT_CHANNELS, SEQUENCE_LENGTH, device=device)
    with torch.no_grad():
        logits = model(dummy)
        probs  = F.softmax(logits, dim=-1)
        conf, cls = torch.max(probs, dim=-1)

    print(f"  [OK] Forward pass successful")
    print(f"       Input  : {tuple(dummy.shape)}")
    print(f"       Output : {tuple(logits.shape)}")
    print(f"       Dummy result → class={cls.item()} ({CLASS_NAMES[cls.item()]}) conf={conf.item():.4f}")

    if "epoch" in ckpt:
        print(f"       Epoch  : {ckpt['epoch']}")
    if "val_f1" in ckpt:
        print(f"       Val F1 : {ckpt['val_f1']:.4f}")
    if "val_loss" in ckpt:
        print(f"       Val Loss: {ckpt['val_loss']:.4f}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Classes: {CLASS_NAMES}")

    for name, path in WEIGHTS.items():
        check_weight(name, path, device)

    print(f"\n{'='*50}")
    print("Done.")
