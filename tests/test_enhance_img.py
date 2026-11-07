"""enhance_img.py function-level tests (synthetic arrays, no files)."""
from __future__ import annotations

import numpy as np
import pytest

from vision.enhance_img import PRESETS, apply_unsharp_mask, enhance


@pytest.fixture()
def color_img():
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(96, 128, 3), dtype=np.uint8)


def test_enhance_preserves_shape_and_dtype(color_img):
    out = enhance(color_img, preset='default')
    assert out.shape == color_img.shape
    assert out.dtype == np.uint8


@pytest.mark.parametrize('preset', sorted(PRESETS))
def test_all_presets_run(color_img, preset):
    out = enhance(color_img, preset=preset)
    assert out.shape == color_img.shape


def test_enhance_rejects_unknown_preset(color_img):
    with pytest.raises(ValueError):
        enhance(color_img, preset='nope')


def test_unsharp_mask_sharpens(color_img):
    out = apply_unsharp_mask(color_img)
    assert out.shape == color_img.shape and out.dtype == np.uint8
    assert not np.array_equal(out, color_img)   # it actually did something
