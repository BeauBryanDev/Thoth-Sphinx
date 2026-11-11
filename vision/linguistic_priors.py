# Linguistic priors for the segmenter: determinatives, phrase-initial
# particles and the curated trie additions. All three come from JSON files
# in vision/data/.

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from vision.lexicon_trie import SourceRecord, SphinxTrie

# Soft bonuses added to the Viterbi segment score. Tuned conservatively so
# corpus frequency still dominates.
# Last code of the segment is a determinative.
DETERMINATIVE_BONUS = 0.5
# Segment exactly matches a phrase-initial particle.
PARTICLE_BONUS = 1.0
# Segment ends at an image-derived row or column break.
LAYOUT_BONUS = 1.5

# Filled on first use. Set to None to reload.
_DETERMINATIVES_CACHE: Optional[set] = None
_PARTICLES_CACHE: Optional[list] = None
_PRIORS_DIR = Path(__file__).parent / 'data'


def load_determinatives(path: Optional[Path] = None) -> set:
    """Gardiner codes flagged as determinatives; empty if the file is absent."""
    global _DETERMINATIVES_CACHE

    if _DETERMINATIVES_CACHE is not None:
        return _DETERMINATIVES_CACHE

    p = Path(path) if path else _PRIORS_DIR / 'determinatives.json'
    if not p.exists():
        _DETERMINATIVES_CACHE = set()
        return _DETERMINATIVES_CACHE

    with open(p, encoding='utf-8') as f:
        data = json.load(f)

    _DETERMINATIVES_CACHE = set(data.get('codes', {}).keys())

    return _DETERMINATIVES_CACHE


def _longest_spelling_first(item: tuple) -> int:
    return -len(item[0])


def load_initial_particles(path: Optional[Path] = None) -> list:
    """Particles as (spelling tuple, meta), longest spelling first."""
    global _PARTICLES_CACHE

    if _PARTICLES_CACHE is not None:
        return _PARTICLES_CACHE

    p = Path(path) if path else _PRIORS_DIR / 'initial_particles.json'
    if not p.exists():
        _PARTICLES_CACHE = []
        return _PARTICLES_CACHE

    with open(p, encoding='utf-8') as f:
        data = json.load(f)

    out: list = []
    for name, meta in data.get('particles', {}).items():
        spellings = [meta.get('gardiner', [])]
        spellings.extend(meta.get('alt_spellings', []))

        for spelling in spellings:
            if spelling:
                info = {'name': name, 'meaning': meta.get('meaning', '')}
                out.append((tuple(spelling), info))

    # Longer matches are tested first.
    out.sort(key=_longest_spelling_first)
    _PARTICLES_CACHE = out

    return _PARTICLES_CACHE


def apply_trie_additions(trie: SphinxTrie, path: Optional[Path] = None) -> int:
    """Insert trie_additions.json entries; returns how many were added."""
    p = Path(path) if path else _PRIORS_DIR / 'trie_additions.json'
    if not p.exists():
        return 0

    with open(p, encoding='utf-8') as f:
        data = json.load(f)

    added = 0
    for entry in data.get('entries', []):
        for codes in entry['spellings']:
            # Added even where the trie has the word: freq is the segmenter's
            # weight.
            existing = trie.search(codes) or []
            curated = False
            for r in existing:
                if r.source == 'curated' and r.translit == entry['translit']:
                    curated = True
                    break

            if curated:
                continue

            record = SourceRecord(
                source='curated',
                translit=entry['translit'],
                translation=entry['translation'],
                freq=int(entry.get('freq', 1)),
                gardiner_orig=' '.join(codes),
            )
            if trie.insert(codes, record):
                added += 1

    return added
