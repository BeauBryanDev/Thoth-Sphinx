# Console summary of a pipeline result, for the CLI.

from __future__ import annotations

from typing import Any


def _cartouche_line(index: int, cartouche: dict[str, Any]) -> str:
    verdict = 'REFUSED'

    if cartouche['translit']:
        how = 'inferred' if cartouche['inferred'] else 'direct'
        verdict = (
            f"{cartouche['translit']} ({how}, "
            f"score={cartouche['score']:.2f})"
        )

    return f"  #{index}  members={cartouche['n_members']:<2}  -> {verdict}"


def print_summary(result: dict[str, Any]) -> None:
    """Print a run summary for the CLI."""
    outer = result['outer']
    correction = outer['correction']

    print(
        f"layout={result['layout']}  direction={result['direction']}  "
        f"shape={result['image_shape']}"
    )
    print(
        f"detections={result['n_detections']}  "
        f"cartouches={result['n_cartouches']}"
    )
    print('\n--- OUTER TEXT ---')
    print(
        f"  slots={len(outer['slots'])}  synthetic={outer['n_synthetic']}  "
        f"boundary_hints={outer['boundary_hints']}"
    )
    print(f"  corrected: {' '.join(correction.get('flat_corrected_seq', []))}")
    print(f"  translit : {correction.get('flat_translit', '')}")
    print(
        f"  score    : {correction.get('score', 0):.2f}  "
        f"fallback={correction.get('had_fallback')}"
    )
    print('\n--- CARTOUCHES ---')

    for index, cartouche in enumerate(result['cartouches']):
        print(_cartouche_line(index, cartouche))
