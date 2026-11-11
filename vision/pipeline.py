# End-to-end pipeline: photo in, reading out.
#
#   image -> [enhance, off by default] -> ONNX pass (letterbox)
#     -> drop edge cartouches -> merge twin cartouches -> layout
#     -> tag cartouche members -> merge duplicate boxes
#     -> outer text:  reading order + corrector      (outer_text)
#     -> cartouches:  interior read + royal names    (cartouche_reading)
#
# CLI: python -m vision.pipeline <image> --layout columns

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import numpy as np
import onnxruntime as ort

from vision import cartouche_matcher as CM
from vision import enhance_img
from vision import layout_detector
from vision import spatial_logic as SL
from vision import sphinx_corrector as SC
from vision.annotate import draw_detections
from vision.cartouche_reading import cartouche_indices, read_cartouches
from vision.outer_text import Corrector, read_outer_text
from vision.run_summary import print_summary

logger = logging.getLogger('sphinxeyes.pipeline')

# Repo root, where artifacts/ lives.
ROOT = Path(__file__).parent.parent
ARTIFACTS = ROOT / 'artifacts'

# 148 Gardiner signs, cartouche and unknown. Fixed since v1.
N_CLASSES = 150
LAYOUTS = ('rows', 'columns')
DEFAULT_LAYOUT = 'rows'

ImageInput = Union[str, Path, bytes, bytearray, np.ndarray]


def _load_class_names(class_map: Path) -> list[str]:
    with open(class_map, encoding='utf-8') as f:
        names = list(json.load(f).keys())

    if len(names) != N_CLASSES:
        raise ValueError(f'expected {N_CLASSES} classes, got {len(names)}')

    return names


def _load_image(image: ImageInput) -> np.ndarray:
    """BGR array from an array, encoded bytes or a file path."""
    if isinstance(image, np.ndarray):
        return image

    if isinstance(image, (bytes, bytearray)):
        buffer = np.frombuffer(image, dtype=np.uint8)
        bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        source = f'{len(image)} bytes'
    else:
        bgr = cv2.imread(str(image))
        source = str(image)

    if bgr is None:
        raise ValueError(f'could not decode image: {source}')

    return bgr


class SphinxPipeline:
    """Single-shot pipeline. Loads the heavy artifacts once."""

    def __init__(
        self,
        onnx_path: Path = ARTIFACTS / 'best_model_v9.onnx',
        class_map: Path = ARTIFACTS / 'class_map50_v9.json',
        trie_pkl: Path = ARTIFACTS / 'sphinx_trie_v4.pkl',
        bbaw_parquet: Path = ARTIFACTS / 'bbaw_clean.parquet',
        confusion_csv: Path = ARTIFACTS / 'confusion_matrix_v9_normalized.csv',
        imgsz: int = 1024,
        providers: Optional[list[str]] = None,
    ) -> None:
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f'ONNX weights not found: {onnx_path}')

        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=providers or ['CPUExecutionProvider'],
        )
        self.class_names = _load_class_names(class_map)
        self.imgsz = imgsz

        # Without the confusion matrix the corrector uses coarse edit costs.
        confusion = Path(confusion_csv)
        self.corrector: Corrector = SC.load_corrector(
            trie_pkl,
            bbaw_parquet,
            confusion_csv=confusion if confusion.exists() else None,
        )
        self.trie, self.log_prob, self.unigrams, self.sub_cost = self.corrector
        self.royal_names = CM.load_royal_names()

        self.infer_global = SL.make_onnx_infer_fn(
            self.session,
            self.class_names,
            conf_thresh=SL.CONF_THRESHOLD,
            imgsz=imgsz,
            mode='letterbox',
        )

    def _detect(self, bgr: np.ndarray) -> list[SL.Detection]:
        """Global pass, then the two cartouche clean-ups that must come
        before the layout vote and the member tagging."""
        img_h, img_w = bgr.shape[:2]
        detections = self.infer_global(bgr)
        detections = SL.filter_edge_cartouches(detections, img_w, img_h)

        return SL.merge_duplicate_cartouches(detections)

    @staticmethod
    def _resolve_layout(
        layout: Optional[str],
        detections: list[SL.Detection],
        img_w: int,
        img_h: int,
    ) -> str:
        """The caller's layout; a logged guess only when none was given."""
        if layout is None:
            layout = layout_detector.detect_layout_from_detections(
                detections, img_w, img_h
            )
            # WARNING: the auto-detector misvotes on real walls.
            logger.warning(
                'layout not supplied, auto-detector guessed %r. '
                'This heuristic is unreliable; pass layout explicitly.',
                layout,
            )

        if layout not in LAYOUTS:
            return DEFAULT_LAYOUT

        return layout

    def run(
        self,
        image: ImageInput,
        *,
        direction: str = 'rtl',
        layout: Optional[str] = None,
        use_enhance: bool = False,
        preset: str = 'default',
        annotate: bool = False,
        use_corrector: bool = True,
    ) -> dict[str, Any]:
        """Read one photo. layout is rows or columns and should be given.
        annotate adds annotated_bgr, an array the API encodes to JPEG."""
        bgr = _load_image(image)
        if use_enhance:
            bgr = enhance_img.enhance(bgr, preset=preset)

        img_h, img_w = bgr.shape[:2]
        detections = self._detect(bgr)
        layout = self._resolve_layout(layout, detections, img_w, img_h)

        # Tagging marks the interior signs in place; the merge comes after.
        SL.tag_cartouche_members(detections, img_w=img_w, img_h=img_h)
        detections = SL.merge_duplicate_boxes(detections)

        outer = read_outer_text(
            detections,
            layout,
            direction,
            self.corrector if use_corrector else None,
        )
        cart_idxs = cartouche_indices(detections, outer.cartouches)
        cartouches = read_cartouches(
            detections,
            cart_idxs,
            layout,
            direction,
            self.royal_names,
            self.sub_cost,
        )

        result: dict[str, Any] = {
            'layout': layout,
            'direction': direction,
            'image_shape': (img_h, img_w),
            'n_detections': len(detections),
            'n_cartouches': len(cart_idxs),
            'outer': {
                'slots': outer.slots,
                'boundary_hints': outer.reading_order.boundary_hints,
                'n_synthetic': outer.reading_order.n_synthetic,
                'correction': outer.correction.to_dict(),
            },
            'cartouches': cartouches,
        }

        if annotate:
            result['annotated_bgr'] = draw_detections(bgr, detections)

        return result


def main() -> None:
    # No default image on purpose: test_image_vn5.png is contaminated.
    parser = argparse.ArgumentParser(
        description='Run the SphinxEyes pipeline on one image.'
    )
    parser.add_argument('image', type=Path)
    parser.add_argument('--layout', choices=LAYOUTS, default=None)
    parser.add_argument('--direction', choices=('rtl', 'ltr'), default='rtl')
    args = parser.parse_args()

    if not args.image.exists():
        sys.exit(f'image not found: {args.image}')

    print('Loading pipeline...')
    pipeline = SphinxPipeline()

    print(f'Running on {args.image} ...')
    print_summary(pipeline.run(
        args.image,
        layout=args.layout,
        direction=args.direction,
    ))


if __name__ == '__main__':
    main()
