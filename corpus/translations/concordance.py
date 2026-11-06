import argparse
import re

from corpus.translations.extract import ALLEN_PDF, pages

# Allen's concordance of Unis: his spell number -> Sethe's PT utterance number, the one Piankoff uses.

PAIR = re.compile(r'W ?(\d+)([ab]?) +PT (\d+)(?:–(\d+))?')


def unis_concordance() -> dict[int, list[int]]:
    """Allen's Unis spell -> PT utterances (a spell split a/b, or 'PT 273–274', maps to several)."""
    text = '\n'.join(pages(ALLEN_PDF))
    start = text.index('UNIS (W)')
    block = text[start:text.index('TETI (T)', start)]
    out: dict[int, list[int]] = {}
    for spell, _, first, last in PAIR.findall(block):
        utterances = range(int(first), int(last or first) + 1)
        out.setdefault(int(spell), []).extend(u for u in utterances if u not in out.get(int(spell), []))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Print Allen's Unis spell to PT utterance concordance.")
    parser.parse_args()

    pairs = unis_concordance()
    print(f'{len(pairs)} Unis spells mapped; e.g. Spell 152 -> PT {pairs.get(152)}, Spell 180 -> PT {pairs.get(180)}')


if __name__ == '__main__':
    main()
