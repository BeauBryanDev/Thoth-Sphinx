from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from app.services.name_glyphs import Sign

ASSETS = Path(__file__).resolve().parents[2] / 'assets'
GLYPH_FONT = ASSETS / 'fonts' / 'NotoSansEgyptianHieroglyphs.woff2'
LATIN_FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf')

LAPIS = np.array([42, 58, 104], dtype=np.float32)   # matched to the frame's wing inlay (51, 67, 92)
BROWN = np.array([74, 46, 24], dtype=np.float32)
SPELL = '\U00013383\U00013143'   # V17 sꜣ + G5 Ḥr: protection of Horus
ONE_COLUMN_MAX = 9
FLAT = 0.55     # glyph height / em below this: flat sign, two stack in one row
NARROW = 0.45   # glyph width / em below this: narrow sign, two stand side by side


@dataclass(frozen=True)
class Box:
    """A rectangle on the frame, in pixels."""
    x0 : int
    y0 : int
    x1 : int
    y1 : int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


FRAME = ASSETS / 'cartouche_frame.webp'   # assets/attempts/frame_2.webp, chosen 2026-11-05
FRAME_PANEL = Box(372, 500, 653, 1045)    # inside the cartouche rope, clear of its rounded ends
FRAME_BASE = Box(200, 1203, 820, 1320)    # the flat base under the tied bar


def glyph_mask(text: str, font: ImageFont.FreeTypeFont) -> np.ndarray:
    """Anti-aliased coverage (0..1) of the text, cropped to its ink."""
    left, top, right, bottom = font.getbbox(text)
    img = Image.new('L', (right - left + 8, bottom - top + 8), 0)
    ImageDraw.Draw(img).text((4 - left, 4 - top), text, font=font, fill=255)
    
    return np.asarray(img, dtype=np.float32) / 255.0


def silhouette(mask: np.ndarray) -> np.ndarray:
    """The outline drawing filled solid: Noto draws signs as line art, inlay needs the whole shape."""
    binary = (mask > 0.35).astype(np.uint8)
    k = max(3, round(min(mask.shape) * 0.06)) | 1   # close the small gaps of open strokes before filling
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    solid = np.zeros_like(binary)
    cv2.drawContours(solid, contours, -1, 1, thickness=cv2.FILLED)
    
    return np.maximum(blur(solid.astype(np.float32), 0.8), mask)


def sign_mask(glyph: str, 
              cell_w: float, 
              cell_h: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """(solid shape, inner line art) of one sign, scaled to fill its cell."""
    probe = glyph_mask(glyph, ImageFont.truetype(str(GLYPH_FONT), 200))
    scale = min(cell_w / probe.shape[1], cell_h / probe.shape[0])
    lines = glyph_mask(glyph, ImageFont.truetype(str(GLYPH_FONT), max(12, int(200 * scale))))
    
    return silhouette(lines), lines


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """The mask moved by (dx, dy), zero-filled."""
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        mask[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
        
    return out


def blur(mask: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur of a 0..1 mask."""
    img = Image.fromarray((mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius))
    
    return np.asarray(img, dtype=np.float32) / 255.0


# make the carved signs look like they are sunk into the stone, with a lip that casts a shadow and a far wall that catches light
#  the image generator model tends to halluciate  the inlay  sings !=  real  user names .
def carve(canvas: np.ndarray,
          mask: np.ndarray,
          x: int,
          y: int,
          rng: np.random.Generator,
          lines: np.ndarray | None = None) -> None:
    """Cut the mask into the stone at (x, y): sunk relief inlaid with lapis, lit from the upper left."""
    h, w = mask.shape
    region = canvas[y:y + h, x:x + w]
    depth = max(2, round(min(h, w) * 0.025))

    outline = np.clip(blur(mask, 1.2) * 2.2 - mask, 0, 1) * (1 - mask)
    region[:] = region * (1 - outline[..., None] * 0.85) + BROWN * outline[..., None] * 0.85

    veins = blur(rng.random((h, w)).astype(np.float32), 2.5)
    lapis = LAPIS + (veins[..., None] - 0.5) * 55 + rng.normal(0, 5, (h, w, 1))
    gleam = blur(shift(mask, -depth * 2, -depth * 2) * mask, depth)   # polish toward the light
    lapis += gleam[..., None] * 30
    region[:] = region * (1 - mask[..., None]) + lapis * mask[..., None]
    
    if lines is not None:   # the sign's inner details, engraved into the inlay
        detail = (lines * mask)[..., None] * 0.75
        region[:] = region * (1 - detail) + BROWN * 0.7 * detail

    shadow = blur(np.clip(mask - shift(mask, depth, depth), 0, 1), depth * 0.6)   # lip shades the cut
    light = blur(np.clip(mask - shift(mask, -depth, -depth), 0, 1), depth * 0.6)   # far wall catches light
    region[:] = region * (1 - shadow[..., None] * 0.7) + BROWN * 0.4 * shadow[..., None] * 0.7
    region[:] = region * (1 - light[..., None] * 0.45) + np.array([236, 212, 160]) * light[..., None] * 0.45


def place(canvas: np.ndarray,
          mask: np.ndarray,
          cx: float,
          cy: float,
          rng: np.random.Generator,
          lines: np.ndarray | None = None) -> None:
    """Carve the mask centred on (cx, cy)."""
    h, w = mask.shape
    carve(canvas, mask, round(cx - w / 2), round(cy - h / 2), rng, lines)


def place_sign(canvas: np.ndarray,
               glyph: str,
               cx: float,
               cy: float,
               cell_w: float,
               cell_h: float,
               rng: np.random.Generator) -> None:
    """Carve one sign, solid and scaled to its cell, centred on (cx, cy)."""
    solid, lines = sign_mask(glyph, cell_w, cell_h)
    place(canvas, solid, cx, cy, rng, lines)


def rows_of(signs: list[Sign], 
font: ImageFont.FreeTypeFont, em: int
) -> list[tuple[str, list[str]]]:
    """Group the signs into rows: alone, two narrow side by side, or two flat stacked."""
    glyphs = [s.glyph for s in signs]
    if len(glyphs) <= ONE_COLUMN_MAX:
        return [('one', [g]) for g in glyphs]

    def ratio(g: str) -> tuple[float, float]:
        left, top, right, bottom = font.getbbox(g)
        return (right - left) / em, (bottom - top) / em

    rows: list[tuple[str, list[str]]] = []
    i = 0
    while i < len(glyphs):
        if i + 1 < len(glyphs):
            (w1, h1), (w2, h2) = ratio(glyphs[i]), ratio(glyphs[i + 1])
            if h1 < FLAT and h2 < FLAT:
                rows.append(('stack', glyphs[i:i + 2]))
                i += 2
                continue

            if w1 < NARROW and w2 < NARROW:
                rows.append(('side', glyphs[i:i + 2]))
                i += 2
                continue

        rows.append(('one', [glyphs[i]]))
        i += 1

    return rows


def carve_name(frame: Image.Image,
               signs: list[Sign],
               name: str,
               panel: Box = FRAME_PANEL,
               base: Box = FRAME_BASE,
               seed: int = 0
               ) -> Image.Image:
    """The frame with the name's signs carved in the panel, and the spell and Latin name on the base."""
    rng = np.random.default_rng(seed)
    canvas = np.asarray(frame.convert('RGB'), dtype=np.float32).copy()

    probe = ImageFont.truetype(str(GLYPH_FONT), 100)
    rows = rows_of(signs, probe, 100)
    row_h = panel.h / len(rows)
    cell_w = min(panel.w * 0.8, row_h * 1.8)   # flat signs go wide, but stay in scale with the birds
    cx = panel.x0 + panel.w / 2
    for n, (kind, glyphs) in enumerate(rows):
        cy = panel.y0 + row_h * (n + 0.5)
        if kind == 'one':
            place_sign(canvas, glyphs[0], cx, cy, cell_w, row_h * 0.88, rng)
        elif kind == 'side':
            for k, g in enumerate(glyphs):
                place_sign(canvas, g, cx + (k - 0.5) * cell_w * 0.5, cy, cell_w * 0.45, row_h * 0.88, rng)
        else:
            for k, g in enumerate(glyphs):
                place_sign(canvas, g, cx, cy + (k - 0.5) * row_h * 0.5, cell_w, row_h * 0.42, rng)

    spell_h = base.h * 0.46
    for k, g in enumerate(SPELL):
        place_sign(canvas, g, base.x0 + base.w / 2 + (k - 0.5) * spell_h * 1.1, base.y0 + base.h * 0.27,
                   spell_h, spell_h, rng)
    latin_size = int(min(base.h * 0.34, base.w * 1.5 / max(len(name), 1)))
    latin_font = ImageFont.truetype(str(LATIN_FONT), latin_size)
    place(canvas, glyph_mask(name, latin_font), base.x0 + base.w / 2, base.y0 + base.h * 0.74, rng)

    return Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8))


def to_webp(image: Image.Image, 
            quality: int = 80
            ) -> bytes:
    """WebP bytes of the image."""
    out = io.BytesIO()
    image.save(out, 'WEBP', quality=quality, method=6)
    
    return out.getvalue()
