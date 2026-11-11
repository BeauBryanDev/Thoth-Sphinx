# Bigram language model (Markov chain, order 1) over the BBAW Gardiner
# sequences, stored as log-probabilities.

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path


def build_bigram_model(
    bbaw_source,
    gardiner_col: str = 'gardiner_seq',
    smoothing: float = 1e-6,
) -> tuple[dict, Counter]:
    """Build (log_prob, unigrams) from a BBAW parquet path or DataFrame."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas is required: pip install pandas")

    if isinstance(bbaw_source, (str, Path)):
        df = pd.read_parquet(bbaw_source)
    else:
        df = bbaw_source

    bigrams = defaultdict(Counter)
    unigrams = Counter()

    for seq_str in df[gardiner_col].dropna():
        codes = seq_str.strip().split()
        if not codes:
            continue

        for c in codes:
            unigrams[c] += 1

        for a, b in zip(codes, codes[1:]):
            bigrams[a][b] += 1

    # log_prob[A][B] = log P(B | A), with additive smoothing.
    vocab_size = len(unigrams)
    log_prob: dict[str, dict[str, float]] = {}

    for a, followers in bigrams.items():
        total = sum(followers.values()) + smoothing * vocab_size
        log_prob[a] = {
            b: math.log((cnt + smoothing) / total)
            for b, cnt in followers.items()
        }

    return log_prob, unigrams


def transition_log_prob(
    prev_code: str,
    next_code: str,
    log_prob: dict,
    unigrams: Counter,
    smoothing: float = 1e-6,
) -> float:
    """log P(next | prev), falling back to the smoothed unigram."""
    if prev_code in log_prob and next_code in log_prob[prev_code]:
        return log_prob[prev_code][next_code]

    vocab_size = len(unigrams)
    total = sum(unigrams.values()) + smoothing * vocab_size
    count = unigrams.get(next_code, 0) + smoothing

    return math.log(count / total)
