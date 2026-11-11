#!/usr/bin/env python3

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from vision.spatial_logic import Detection


CARTOUCHE_CLASS = 'cartouche'
EDGE_FRAC       = 0.02   # touching this fraction of the image border -> drop
                         # the cartouche from the aspect vote (clipped)
MIN_SIGNAL_GAP  = 0.10   # column-vs-row vote margin below which we tie


def _cluster_1d_bands(values: np.ndarray, bandwidth: float) -> int:
    """1-D running-mean clustering. Returns the number of clusters.

    Mirrors spatial_logic._group_lines so the band count matches what
    the reading-order assembler would see.
    """
    if len(values) == 0:
        return 0
    
    v = np.sort(values)
    bands = [[v[0]]]
    
    for x in v[1:]:
        
        mean_b = float(np.mean(bands[-1]))
        
        if x - mean_b > bandwidth:
            
            bands.append([x])
            
        else:
            bands[-1].append(x)
            
    return len(bands)


def _vote_cartouche_aspect(
                detections : list[Detection],
                img_w: float, 
                img_h: float,
                ) -> tuple[str | None, float]:
    """Tall cartouches -> columns; wide -> rows.
    Cartouches whose bbox touches the image border are dropped (clipped)."""
    edge_x = EDGE_FRAC * img_w
    edge_y = EDGE_FRAC * img_h
    tall = wide = 0
    for d in detections:
        
        if not getattr(d, 'is_cartouche', lambda: False)():
            continue
        
        x1, y1, x2, y2 = d.bbox
        
        if x1 <= edge_x or y1 <= edge_y or x2 >= img_w - edge_x or y2 >= img_h - edge_y:
            continue
        
        w, h = x2 - x1, y2 - y1
        
        if h > 1.2 * w:
            
            tall += 1
            
        elif w > 1.2 * h:
            
            wide += 1
            
    total = tall + wide
    
    if total == 0:
        
        return None, 0.0
    
    if tall > wide:
        # tie -> historical default
        return 'columns', tall / total
    
    if wide > tall:
        # tie -> historical default
        return 'rows', wide / total
    
    return None, 0.0


def _vote_band_count(
                detections : list[Detection],
                img_w: float, 
                img_h: float,
                ) -> tuple[str | None, float]:
    """Cluster non-cartouche centroids on x and on y; reading axis is
    the one with MORE bands (signs stacked along it)."""
    sign_dets = [d for d in detections
                 if not getattr(d, 'is_cartouche', lambda: False)()]
    
    if len(sign_dets) < 4:
        
        return None, 0.0
    
    cx = np.array([d.centroid[0] for d in sign_dets])
    cy = np.array([d.centroid[1] for d in sign_dets])
    
    med_w = float(np.median([d.width  for d in sign_dets]))
    med_h = float(np.median([d.height for d in sign_dets]))
    # bands_x : n_x: how many distinct vertical bands of signs (potential columns)
    # bands_y : n_y: how many distinct horizontal bands of signs (potential rows)
    n_x = _cluster_1d_bands(cx, 0.6 * med_w)   # columns count
    n_y = _cluster_1d_bands(cy, 0.6 * med_h)   # rows count
    
    if n_x == 0 or n_y == 0:
        return None, 0.0
    
    ratio = abs(n_y - n_x) / max(n_x, n_y)
    
    if n_y > n_x:  # more rows than columns -> text in columns
        
        return 'columns', ratio
    
    if n_x > n_y:
        
        return 'rows', ratio
    
    return None, 0.0


def _vote_sign_aspect(
            detections : list[Detection], 
            img_w: float, 
            img_h: float
            ) -> tuple[str | None, float]:
    """Median sign aspect ratio. Mild tiebreaker."""
    aspects = []
    
    for d in detections:
        
        if getattr(d, 'is_cartouche', lambda: False)():
            continue
        
        if d.width <= 0 or d.height <= 0:
            continue
        
        aspects.append(d.height / d.width)
        
    if not aspects:
        
        return None, 0.0
    
    med = float(np.median(aspects))
    
    if med >= 1.15:
        
        return 'columns', min(0.3, (med - 1.0))
    
    if med <= 0.85:
        
        return 'rows', min(0.3, (1.0 - med))
    
    return None, 0.0

 
def detect_layout_from_detections(
            detections : list[Detection],
            img_w: float, 
            img_h: float, *, 
            verbose: bool = False,
            ) -> str:
    """
    Vote across three geometric signals and return 'rows' or 'columns'.
    Fragile-breakpoint fallback: 'rows' on no evidence (historical default).
    """
    if not detections:
        
        if verbose:
            
            print(f"WARNING: no detections")
        # fallback to historical default rows layout
        return 'rows'
    
    if img_w <= 0 or img_h <= 0:
        
        if verbose:
            
            print(f"WARNING: invalid image size: {img_w}x{img_h}")
        # fallback to historical default rows layout
            raise ValueError(f"invalid image size: {img_w}x{img_h}")
        
        return 'rows'
    
    votes: dict[str, float] = {'rows': 0.0, 'columns': 0.0}
    # More important signals have higher weights
    weights = {'cartouche_aspect': 1.5, 'band_count': 1.0, 'sign_aspect': 0.5}

    for name, fn in (
        ('cartouche_aspect', _vote_cartouche_aspect),
        ('band_count',       _vote_band_count),
        ('sign_aspect',      _vote_sign_aspect),
    ):
        if name == 'sign_aspect':
            
            label, conf = fn(detections, img_w, img_h)
            
        else:
            label, conf = fn(detections, img_w, img_h)
            
        if label is not None:
            
            votes[label] += weights[name] * conf
            
        if verbose:
            
            print(f"  layout vote [{name}]: {label} (conf={conf:.2f})")

    diff = votes['columns'] - votes['rows']
    
    if verbose:
        
        print(f"  totals: columns={votes['columns']:.2f}  rows={votes['rows']:.2f}")
        
    if abs(diff) < MIN_SIGNAL_GAP:
        
        return 'rows'    # tie -> historical default
    
    return 'columns' if diff > 0 else 'rows'



# Legacy pixel fallback (kept for callers that don't have detections yet).

# def detect_layout(img) -> str:
#     """Pixel-based fallback (morphological line detection).
#     Prefer detect_layout_from_detections() — this one fails on stone reliefs.
#     """
#     import cv2
#     gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(
#         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV, 15, 4,
#     )
#     h, w = binary.shape
#     h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
#     v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
#     h_score  = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)) / 255
#     v_score  = np.sum(cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)) / 255
#     MIN_SCORE = 500
#     if h_score > MIN_SCORE and h_score > v_score * 1.5:
#         return 'rows'
#     if v_score > MIN_SCORE and v_score > h_score * 1.5:
#         return 'columns'
#     return 'rows'
