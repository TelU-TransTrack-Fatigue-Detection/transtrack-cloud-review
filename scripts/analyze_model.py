import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from app.pipeline import MultiScaleTCN, CLASS_NAMES, SEQUENCE_LENGTH, LANDMARK_FPS

WEIGHTS = {
    "best_val_f1":   Path("models/classifier/best_val_f1.pth"),
    "best_val_loss": Path("models/classifier/best_val_loss.pth"),
    "latest":        Path("models/classifier/latest.pth"),
}

def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def analyze_ckpt(name, path):
    print(f"\n{'='*60}")
    print(f"Checkpoint : {name}")
    print(f"File       : {path}")
    ckpt  = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)

    print(f"\n--- Saved metadata ---")
    for k, v in ckpt.items():
        if k != "model_state_dict":
            print(f"  {k}: {v}")

    if any(k.startswith("backbone.") for k in state):
        state = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}

    print(f"\n--- Layer shapes ---")
    for k, v in state.items():
        print(f"  {k:<55} {str(tuple(v.shape))}")

    model = MultiScaleTCN(num_classes=len(CLASS_NAMES), input_channels=8)
    model.load_state_dict(state)
    total, trainable = count_params(model)

    print(f"\n--- Model summary ---")
    print(f"  Input shape      : (batch, 8, {SEQUENCE_LENGTH})")
    print(f"  Input channels   : 8  [ear_l, ear_r, mar, pitch, yaw, roll, nose_x, nose_y]")
    print(f"  Sequence length  : {SEQUENCE_LENGTH} frames @ {LANDMARK_FPS} FPS = {int(SEQUENCE_LENGTH/LANDMARK_FPS)}s window")
    print(f"  Output classes   : {len(CLASS_NAMES)}  {CLASS_NAMES}")
    print(f"  Total params     : {total:,}")
    print(f"  Trainable params : {trainable:,}")

    print(f"\n--- Per-layer param count ---")
    for name_l, module in model.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.BatchNorm1d)):
            p = sum(x.numel() for x in module.parameters())
            print(f"  {name_l:<50} {type(module).__name__:<15} {p:>10,} params")

    print(f"\n--- Output distribution on zero input ---")
    model.eval()
    with torch.no_grad():
        dummy  = torch.zeros(1, 8, SEQUENCE_LENGTH)
        logits = model(dummy)
        import torch.nn.functional as F
        probs  = F.softmax(logits, dim=-1).squeeze()
    for i, (cls, p) in enumerate(zip(CLASS_NAMES, probs)):
        bar = "█" * int(p.item() * 40)
        print(f"  {cls:<10} {p.item():.4f}  {bar}")

if __name__ == "__main__":
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA available  : {torch.cuda.is_available()}")

    for name, path in WEIGHTS.items():
        if path.exists():
            analyze_ckpt(name, path)
        else:
            print(f"\n[MISSING] {name}: {path}")
