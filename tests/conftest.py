"""Shared fixtures for the SphinxEyes test suite.

Heavy artifacts (trie, parquet, ONNX) load once per session and skip
cleanly when absent so the fast suite runs anywhere.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

ARTIFACTS = REPO_ROOT / 'artifacts'


# ---------------------------------------------------------------------------
# Lexicon / corrector / pipeline (heavy — session scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def royal_names():
    from vision.cartouche_matcher import load_royal_names
    names = load_royal_names()          # vision/royal_names.json
    assert 'wnjs' in names, 'royal_names.json must contain wnjs'
    return names


@pytest.fixture(scope='session')
def corrector():
    """(trie, log_prob, unigrams, sub_cost) from real artifacts."""
    trie_pkl = ARTIFACTS / 'sphinx_trie_v4.pkl'
    parquet = ARTIFACTS / 'bbaw_clean.parquet'
    confusion = ARTIFACTS / 'confusion_matrix_v9_normalized.csv'
    if not trie_pkl.exists() or not parquet.exists():
        pytest.skip('corrector artifacts missing')
    from vision.sphinx_corrector import load_corrector
    return load_corrector(
        trie_pkl, parquet,
        confusion_csv=confusion if confusion.exists() else None,
    )


@pytest.fixture(scope='session')
def sphinx_pipeline():
    """Real SphinxPipeline (ONNX). Slow tests only."""
    from app.core.config import settings
    if not settings.onnx_path.exists():
        pytest.skip('ONNX model missing')
    from vision.pipeline import SphinxPipeline
    return SphinxPipeline()


# ---------------------------------------------------------------------------
# Canned pipeline output + stub pipeline (fast API tests, no ONNX)
# ---------------------------------------------------------------------------

def make_raw_result() -> dict:
    """Minimal-but-complete SphinxPipeline.run() output shape."""
    return {
        'layout': 'columns',
        'direction': 'rtl',
        'image_shape': (480, 640),
        'n_detections': 3,
        'n_cartouches': 1,
        'outer': {
            'slots': [[('G17', 0.9), ('G18', 0.05)], [('N35', 0.8)]],
            'boundary_hints': [2],
            'n_synthetic': 0,
            'correction': {
                'segmented_words': [{
                    'codes': ['G17', 'N35'], 'translit': 'mn',
                    'translation': 'test', 'freq': 10, 'edit_dist': 0,
                    'confidence': 0.9, 'source': 'trie',
                }],
                'unknowns_resolved': [],
                'flat_corrected_seq': ['G17', 'N35'],
                'flat_translit': 'mn',
                'flat_translation': 'test',
                'score': -1.0,
                'had_fallback': False,
            },
        },
        'cartouches': [{
            'bbox': (10.0, 10.0, 100.0, 40.0), 'n_members': 4,
            'slots': [[('N5', 0.8)], [('L1', 0.86)], [('N35', 0.83)]],
            'inferred': False, 'translit': 'wnjs', 'english': 'Unas',
            'spelling': ['E34', 'N35', 'M17', 'S29'], 'score': 3.2,
            'aligned_codes': ['E34', 'N35', 'M17', 'S29'], 'verified': True,
        }],
        'annotated_bgr': np.zeros((32, 32, 3), dtype=np.uint8),
    }


class StubPipeline:
    """Duck-types the bits of SphinxPipeline the routers touch."""
    version = 'stub'
    class_names = [f'c{i}' for i in range(150)]

    class _Out:
        shape = [1, 154, 21504]

    class _Session:
        @staticmethod
        def get_outputs():
            return [StubPipeline._Out()]

    session = _Session()

    def run(self, img, **kwargs):
        self.last_kwargs = kwargs
        return make_raw_result()


@pytest.fixture()
def stub_pipeline():
    return StubPipeline()


@pytest.fixture()
def client(stub_pipeline):
    """TestClient with lifespan bypassed and the stub pipeline injected."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    app = create_app()

    @asynccontextmanager
    async def _noop_lifespan(app_):
        app_.state.pipeline = stub_pipeline
        yield

    app.router.lifespan_context = _noop_lifespan
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def no_api_key(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, 'anthropic_api_key', '')


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

@pytest.fixture(scope='session')
def unas_image() -> Path:
    p = REPO_ROOT / 'Unas1c.jpg'
    if not p.exists():
        pytest.skip('Unas1c.jpg fixture missing')
    return p


@pytest.fixture(scope='session')
def glyph_wall_image() -> Path:
    p = REPO_ROOT / 'glyph_wall.jpeg'
    if not p.exists():
        pytest.skip('glyph_wall.jpeg fixture missing')
    return p


@pytest.fixture()
def tiny_jpeg() -> bytes:
    import cv2
    ok, buf = cv2.imencode('.jpg', np.full((64, 64, 3), 128, dtype=np.uint8))
    assert ok
    return bytes(buf)


@pytest.fixture(autouse=True)
def fresh_reread_limit(monkeypatch):
    """Each test starts with an empty re-read cooldown, whatever ran before it."""
    from app.core.ratelimit import Cooldown
    from app.routers import transliterate
    monkeypatch.setattr(transliterate, 'REREAD_LIMIT', Cooldown(3600))


@pytest.fixture(autouse=True)
def fresh_guards(monkeypatch):
    """Each test starts with empty request windows, daily quotas and spend."""
    from app.core import ratelimit
    monkeypatch.setattr(ratelimit, 'REQUESTS', ratelimit.SlidingWindow())
    monkeypatch.setattr(ratelimit, 'TRANSLIT_DAILY', ratelimit.DailyQuota())
    monkeypatch.setattr(ratelimit, 'SPEND', ratelimit.SpendCap())
    monkeypatch.setattr(ratelimit, 'CARTOUCHE_DAILY', ratelimit.DailyQuota())
