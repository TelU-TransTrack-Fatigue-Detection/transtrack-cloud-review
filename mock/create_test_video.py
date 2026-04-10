"""
Generates a synthetic test.mp4 in mock/videos/ for local end-to-end testing.
No real footage needed. Black frames — worker will find no face, run inference
anyway, and complete the full pipeline (download → extract → infer → callback).

Run once:
    python mock/create_test_video.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "videos" / "test.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

FPS    = 10
FRAMES = 200  # 20 seconds @ 10fps — matches model sequence length
W, H   = 320, 240

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))

for _ in range(FRAMES):
    writer.write(np.zeros((H, W, 3), dtype=np.uint8))

writer.release()
print(f"Created: {OUT.resolve()}  ({FRAMES} frames @ {FPS}fps)")
