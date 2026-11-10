#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from vision import cartouche_matcher as CM
from vision import enhance_img
from vision import layout_detector
from vision import spatial_logic as SL
from vision import sphinx_corrector as SC
"""
pipeline.py — end-to-end glue (V4).

bytes/path/ndarray
[optional enhance_img.enhance() — OFF by default; A/B on grand_glyphs
    showed it deletes real signs and hallucinates fakes on clean images]
ONNX global pass (letterbox resize — matches Ultralytics V4 training)
postprocess_onnx                       (conf  NMS  top-3)
tag_cartouche_members
merge_duplicate_boxes
outer:    cluster_quadrats + assemble_reading_order + sphinx_corrector.correct()
cartouches: per-cartouche RO (single_line=True)
                + cartouche_matcher.match_cartouche + apply_panel_consensus

Smoke test:
    python -m vision.pipeline <image>
"""

ROOT = Path(__file__).parent.parent   # repo root, where artifacts/ lives

# Drop cartouches whose bbox sits within this fraction of the image border.
# Egyptian cartouches always have both bracket ends visible; a clipped one
# is a photo-crop artifact and its interior is incomplete by definition.
EDGE_FRAC = 0.01


def _draw_detections(bgr: np.ndarray, detections) -> np.ndarray:
    """
    YOLO-style annotated copy of the image: one box per detection with
    'class conf' label. Cartouches gold and thicker; signs green;
    unknowns grey. Label/line size scales with image resolution.
    """
    out = bgr.copy()
    
    H, W = out.shape[:2]
    
    thick = max(1, round(min(H, W) / 640))
    
    font_scale = min(H, W) / 1600
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    for d in detections:
        
        x1, y1, x2, y2 = (int(round(v)) for v in d.bbox)
        
        cls, score = d.top3[0]
        
        if d.is_cartouche():
            
            color, t = (0, 190, 255), thick * 2      # gold (BGR)
            
        elif cls.lower() == 'unknown':
            
            color, t = (160, 160, 160), thick        # grey
            
        else:
            
            color, t = (80, 200, 60), thick          # green
            
        cv2.rectangle(out, (x1, y1), (x2, y2), color, t)

        label = f'{cls} {min(score, 0.998):.2f}'
        
        (tw, th), base = cv2.getTextSize(label, font, font_scale, thick)
        
        ty = y1 - 4 if y1 - th - base - 4 >= 0 else y2 + th + base + 4
        
        cv2.rectangle(out, (x1, ty - th - base), 
                      (x1 + tw + 2, ty + base),
                      color, -1)
        cv2.putText(out, label, (x1 + 1, ty), 
                    font, font_scale,
                    (0, 0, 0), thick, 
                    cv2.LINE_AA)
        
    return out


def _filter_edge_cartouches(detections, img_w: int, img_h: int) -> list:
    """Drop cartouche-class detections whose bbox touches any image edge."""
    ex, ey = EDGE_FRAC * img_w, EDGE_FRAC * img_h #| EDGE_FRAC * img_h
    kept = []
    for d in detections:
        
        if d.is_cartouche():
            
            x1, y1, x2, y2 = d.bbox
            
            if (x1 <= ex or y1 <= ey or x2 >= img_w - ex or y2 >= img_h - ey):
                continue
            
        kept.append(d)
        
    return kept


class SphinxPipeline:
    """Single-shot pipeline. Loads heavy artifacts once at construction."""

    def __init__(
        self,
        onnx_path     : Path = ROOT / 'artifacts' / 'best_model_v9.onnx',
        class_map     : Path = ROOT / 'artifacts' / 'class_map50_v9.json',
        trie_pkl      : Path = ROOT / 'artifacts' / 'sphinx_trie_v4.pkl',
        bbaw_parquet  : Path = ROOT / 'artifacts' / 'bbaw_clean.parquet',
        confusion_csv : Path = ROOT / 'artifacts' / 'confusion_matrix_v9_normalized.csv',
        imgsz         : int  = 1024,
        providers     : Optional[list] = None,
    ):
        if not Path(onnx_path).exists():
            
            raise FileNotFoundError(f'ONNX weights not found: {onnx_path}')
        
        providers = providers or ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        
        self.class_names = list(json.load(open(class_map)).keys())
        
        assert len(self.class_names) == 150, \
            f'expected 150 classes, got {len(self.class_names)}'

        conf = Path(confusion_csv) if Path(confusion_csv).exists() else None
        
        self.trie, self.log_prob, self.unigrams, self.sub_cost = \
            SC.load_corrector(trie_pkl, bbaw_parquet, confusion_csv=conf)

        self.royal_names = CM.load_royal_names()
        
        self.imgsz = imgsz

        self.infer_global = SL.make_onnx_infer_fn(
            self.session, self.class_names,
            conf_thresh=SL.CONF_THRESHOLD, imgsz=imgsz, mode='letterbox',
        )


    @staticmethod
    def _load_image(image: Union[str, Path, bytes, bytearray, np.ndarray]) -> np.ndarray:
        # TODO: use PIL
        if isinstance(image, np.ndarray):
            
            return image
        
        if isinstance(image, (bytes, bytearray)):
            
            arr = np.frombuffer(image, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            
        else:
            bgr = cv2.imread(str(image))
            
        if bgr is None:
            
            raise ValueError(f'could not decode image: {image!r}')
        
        return bgr


    def run(
        self,
        image       : Union[str, Path, bytes, bytearray, np.ndarray],
        *,
        direction   : str  = 'rtl',
        layout      : Optional[str] = None,   # 'rows' | 'columns' | None=auto
        use_enhance : bool = False,
        preset      : str  = 'default',
        annotate    : bool = False,   # add 'annotated_bgr' ndarray to result
        use_corrector : bool = True,  # False: YOLO top-1 codes go out as they are (A/B, 2026-11-06)
    ) -> dict:
        bgr = self._load_image(image)
        if use_enhance:
            bgr = enhance_img.enhance(bgr, preset=preset)

        H, W = bgr.shape[:2]

        # 1) Global detection (YOLO-driven)
        dets = self.infer_global(bgr)

        # 1a) Drop edge-clipped cartouches (partial bracket -> incomplete
        #     interior). Must run before layout vote so the cartouche-aspect
        #     signal isn't poisoned by tiny edge slivers.
        dets = _filter_edge_cartouches(dets, W, H)

        # 1a') Collapse twin cartouche boxes over the same physical cartouche
        #      (a low CARTOUCHE_CONF admits weak duplicates that the 0.50 NMS
        #      leaves alone). Must run before tagging so interior signs bind
        #      to the surviving box, and before the layout vote so duplicate
        #      cartouches don't skew the cartouche-aspect signal.
        dets = SL.merge_duplicate_cartouches(dets)

        # 1b) Layout should be supplied by the caller — the geometric
        #     auto-detector is UNRELIABLE on real walls (misvotes rows vs
        #     columns). Kept only as a last-resort fallback for callers
        #     that genuinely cannot know; the API makes layout mandatory.
        if layout is None:

            layout = layout_detector.detect_layout_from_detections(dets, W, H)
            import logging
            logging.getLogger('sphinxeyes.pipeline').warning(
                f'layout not supplied — auto-detector guessed {layout!r}. '
                f'This heuristic is unreliable; pass layout explicitly.')

        if layout not in ('rows', 'columns'):
            
            layout = 'rows'

        # 2) Cartouche containment + duplicate merge
        cart_idxs = SL.tag_cartouche_members(dets, img_w=W, img_h=H)
        dets = SL.merge_duplicate_boxes(dets)
        
        cart_idxs = [i for i, d in enumerate(dets) if d.is_cartouche()]

        # 3) Outer text reading order + corrector
        outer = [d for d in dets
                 if not d.inside_cartouche and not d.is_cartouche()]
        
        outer_quads = SL.cluster_quadrats(outer, layout=layout, split_lines=True)
        
        outer_ro    = SL.assemble_reading_order(
            outer_quads, layout=layout, direction=direction,
            blockers=[dets[i].bbox for i in cart_idxs],
        )

        canon_outer = self._canon_slots(outer_ro.slots)
        
        if use_corrector:
            correction = SC.correct(
                canon_outer, self.trie, self.log_prob, self.unigrams,
                sub_cost_matrix=self.sub_cost,
                boundary_hints=outer_ro.boundary_hints,
            )
        else:
            # Same shape, no trie and no Viterbi: each slot's top-1 code, empty slots as Unknown.
            correction = SC.CorrectionResult(
                segmented_words    = [],
                unknowns_resolved  = [],
                flat_corrected_seq = [slot[0][0] if slot else 'Unknown' for slot in canon_outer],
                flat_translit      = '',
                flat_translation   = '',
                score              = 0.0,
                had_fallback       = False,
            )

        # 4) Cartouches (interior reading + royal-name match + consensus)
        cart_results  : list[Optional[CM.RoyalMatch]] = []
        cart_slots_all: list[list[list[tuple[str, float]]]] = []
        cart_meta     : list[dict] = []

        for ci in cart_idxs:
            members = [d for d in dets
                       if d.inside_cartouche and d.cartouche_id == ci]
            if not members:
                cart_results.append(None)
                cart_slots_all.append([])
                cart_meta.append({'bbox': dets[ci].bbox, 'n_members': 0})
                continue
            quads = SL.cluster_quadrats(members, layout=layout)
            ro = SL.assemble_reading_order(
                quads, layout=layout, direction=direction,
                single_line=True, extent=dets[ci].bbox,
            )
            slots = self._canon_slots(ro.slots)
            cart_slots_all.append(slots)
            cart_results.append(
                CM.match_cartouche(slots, self.royal_names, sub_cost=self.sub_cost)
            )
            cart_meta.append({'bbox': dets[ci].bbox, 'n_members': len(members)})

        cart_final = CM.apply_panel_consensus(
            cart_results, cart_slots_all, self.royal_names,
            sub_cost=self.sub_cost,
        )

        result = {
            'layout'        : layout,
            'direction'     : direction,
            'image_shape'   : (H, W),
            'n_detections'  : len(dets),
            'n_cartouches'  : len(cart_idxs),
            'outer': {
                'slots'          : canon_outer,
                'boundary_hints' : outer_ro.boundary_hints,
                'n_synthetic'    : outer_ro.n_synthetic,
                'correction'     : correction.to_dict(),
            },
            'cartouches': [
                {
                    'bbox'          : meta['bbox'],
                    'n_members'     : meta['n_members'],
                    'slots'         : slots,
                    'inferred'      : inferred,
                    'translit'      : m.translit       if m else None,
                    'english'       : m.english        if m else None,
                    'spelling'      : m.spelling       if m else None,
                    'score'         : m.score          if m else None,
                    'aligned_codes' : m.aligned_codes  if m else None,
                    'verified'      : m.verified       if m else None,
                }
                for (m, inferred), slots, meta in zip(
                    cart_final, cart_slots_all, cart_meta)
            ],
        }
        if annotate:
            # ndarray, not JSON-serializable — API layer encodes to JPEG.
            result['annotated_bgr'] = _draw_detections(bgr, dets)
        return result

    @staticmethod
    def _canon_slots(slots):
        return [[(CM.normalize_code(c), float(s)) for c, s in slot]
                for slot in slots]


# Thin CLI — prints a run summary; all tests live in tests/ (pytest)

def _summarize(result: dict) -> None:
    print(f"layout={result['layout']}  direction={result['direction']}  "
          f"shape={result['image_shape']}")
    print(f"detections={result['n_detections']}  "
          f"cartouches={result['n_cartouches']}")
    o = result['outer']
    corr = o['correction']
    print(f"\n--- OUTER TEXT ---")
    print(f"  slots={len(o['slots'])}  synthetic={o['n_synthetic']}  "
          f"boundary_hints={o['boundary_hints']}")
    print(f"  corrected: {' '.join(corr.get('flat_corrected_seq', []))}")
    print(f"  translit : {corr.get('flat_translit', '')}")
    print(f"  score    : {corr.get('score', 0):.2f}  "
          f"fallback={corr.get('had_fallback')}")
    print(f"\n--- CARTOUCHES ---")
    for k, c in enumerate(result['cartouches']):
        tag = (f"{c['translit']} ({'inferred' if c['inferred'] else 'direct'}, "
               f"score={c['score']:.2f})") if c['translit'] else "REFUSED"
        print(f"  #{k}  members={c['n_members']:<2}  -> {tag}")


def main():
    # No default image: test_image_vn5.png is CONTAMINATED (CLAUDE.md
    # fragile breakpoint #8) — always require an explicit path.
    if len(sys.argv) < 2:
        sys.exit('usage: python -m vision.pipeline <image> '
                 '(e.g. Unas1c.jpg, image_2_test.jpg)')
    img = sys.argv[1]
    if not Path(img).exists():
        sys.exit(f'image not found: {img}')
    print(f"Loading pipeline...")
    p = SphinxPipeline()
    print(f"Running on {img} ...")
    result = p.run(img)
    _summarize(result)


if __name__ == '__main__':
    main()
