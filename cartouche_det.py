#!/usr/bin/env python3
"""
cartouche_det.py — diagnostic: is the V4 model actually seeing cartouches?

Runs the ONNX model over every  cartouche_*.{png,jpeg,jpg,webp}  in the repo
root and inspects the RAW cartouche-class score (class index 9) for every
anchor, BEFORE any conf threshold or NMS. This tells us whether the model is
weakly firing on the oval (score present but sub-threshold) or missing it
entirely (score near zero).

Preprocessing = letterbox @1024, identical to the production global pass,
so the raw tensor matches what pipeline.py feeds the net.

Usage:
    python cartouche_det.py                 # all cartouche_* in root
    python cartouche_det.py some_image.jpg  # one explicit image
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from vision.spatial_logic import letterbox, CONF_THRESHOLD

ROOT            = Path(__file__).parent
ONNX_PATH       = ROOT / 'artifacts' / 'best_model_v4.onnx'
CLASS_MAP       = ROOT / 'artifacts' / 'class_map50_v4.json'
CARTOUCHE_IDX   = 9                       # 'cartouche' in class_map50_v4.json
IMGSZ           = 1024
EXTS            = ('.png', '.jpeg', '.jpg', '.webp')

# CARTOUCHE_CONF in the pipeline is 0.25 (weathered ovals score ~0.28);
# CONF_THRESHOLD (global keep) is also 0.25. Report against that bar.
THRESH          = CONF_THRESHOLD


def load_session() -> tuple[ort.InferenceSession, list[str]]:
    if not ONNX_PATH.exists():
        sys.exit(f'ONNX weights not found: {ONNX_PATH}')
    sess = ort.InferenceSession(str(ONNX_PATH),
                                providers=['CPUExecutionProvider'])
    class_names = list(json.load(open(CLASS_MAP)).keys())
    assert class_names[CARTOUCHE_IDX] == 'cartouche', \
        f"index {CARTOUCHE_IDX} is {class_names[CARTOUCHE_IDX]!r}, not 'cartouche'"
    return sess, class_names


def raw_forward(sess: ort.InferenceSession, bgr: np.ndarray) -> np.ndarray:
    """Letterbox -> ONNX -> raw output [4+C, N] (batch stripped)."""
    canvas, _, _, _ = letterbox(bgr, IMGSZ)
    x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    raw = sess.run([out], {inp: x})[0]        # [1, 4+C, N]
    return raw[0]


def inspect(sess, class_names, img_path: Path) -> None:
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f'  !! could not decode {img_path.name}')
        return

    raw = raw_forward(sess, bgr)                          # [4+C, N]
    cart = raw[4 + CARTOUCHE_IDX, :]                      # [N] sigmoid probs
    all_scores = raw[4:, :]                               # [C, N]

    n_over   = int((cart >= THRESH).sum())
    best_anchor = int(cart.argmax())
    best_cart_score = float(cart[best_anchor])

    # What class actually wins at the anchor with the highest cartouche score?
    winner_idx   = int(all_scores[:, best_anchor].argmax())
    winner_name  = class_names[winner_idx]
    winner_score = float(all_scores[winner_idx, best_anchor])

    top5 = np.sort(cart)[-5:][::-1]

    verdict = ('DETECTED'      if best_cart_score >= THRESH else
               'SUB-THRESHOLD' if best_cart_score >= 0.10   else
               'MISSED')

    print(f'\n{img_path.name}  ({bgr.shape[1]}x{bgr.shape[0]})')
    print(f'  max cartouche score : {best_cart_score:.4f}   [thresh {THRESH:.2f}] -> {verdict}')
    print(f'  anchors >= thresh   : {n_over}')
    print(f'  top-5 cartouche     : {np.array2string(top5, precision=4, floatmode="fixed")}')
    print(f'  winner @ best anchor: {winner_name!r} ({winner_score:.4f})'
          + ('  <-- cartouche loses to another class here'
             if winner_idx != CARTOUCHE_IDX else '  <-- cartouche wins'))


def main() -> None:
    sess, class_names = load_session()

    if len(sys.argv) > 1:
        images = [Path(a) for a in sys.argv[1:]]
    else:
        images = sorted(
            p for p in ROOT.iterdir()
            if p.is_file()
            and p.stem.lower().startswith('cartouche_')
            and p.suffix.lower() in EXTS
        )

    if not images:
        sys.exit('no cartouche_*.{png,jpeg,jpg,webp} images found in repo root')

    print(f'Model : {ONNX_PATH.name}')
    print(f'Images: {len(images)}   cartouche class idx = {CARTOUCHE_IDX}')
    print('=' * 64)
    for img in images:
        inspect(sess, class_names, img)


if __name__ == '__main__':
    main()
