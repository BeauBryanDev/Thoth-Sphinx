from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MIN_LETTERS = 2
MAX_LETTERS = 16
VOWELS = frozenset('AEIOUY')


@dataclass(frozen=True)
class Sign:
    """One uniliteral hieroglyph: Gardiner code, Unicode glyph and what it looks like."""
    code  : str
    glyph : str
    look  : str


VULTURE = Sign('G1', '\U0001313F', 'an Egyptian vulture standing')
FOOT    = Sign('D58', '\U000130C0', 'a human lower leg and foot')
BASKET  = Sign('V31', '\U000133A1', 'a small basket with a handle')
CLOTH   = Sign('S29', '\U000132F4', 'a folded cloth standing upright')
HAND    = Sign('D46', '\U000130A7', 'a hand with flat palm and fingers together')
REED    = Sign('M17', '\U000131CB', 'a single flowering reed')
REEDS   = Sign('M17A', '\U000131CC', 'two flowering reeds side by side')
VIPER   = Sign('I9', '\U00013191', 'a horned viper')
STAND   = Sign('W11', '\U000133BC', 'a ring-shaped jar stand')
SHELTER = Sign('O4', '\U00013254', 'a reed shelter')
COBRA   = Sign('I10', '\U00013193', 'a cobra rearing')
LION    = Sign('E23', '\U000130ED', 'a lion lying down')
OWL     = Sign('G17', '\U00013153', 'an owl seen from the side with its face turned to the front')
WATER   = Sign('N35', '\U00013216', 'a horizontal zigzag water ripple')
CHICK   = Sign('G43', '\U00013171', 'a quail chick')
STOOL   = Sign('Q3', '\U000132AA', 'a square stool')
HILL    = Sign('N29', '\U0001320E', 'a sandy hill slope')
MOUTH   = Sign('D21', '\U0001308B', 'an open mouth, a flat oval with pointed ends')
BREAD   = Sign('X1', '\U000133CF', 'a small half-round bread loaf')
BOLT    = Sign('O34', '\U00013283', 'a door bolt')
POOL    = Sign('N37', '\U00013219', 'a rectangular pool of water')
SIEVE   = Sign('Aa1', '\U0001340D', 'a round sieve with crossed stripes')
TETHER  = Sign('V13', '\U0001337F', 'a tethering rope with a loop')

# Digraphs are read before single letters, longest first.
DIGRAPHS: dict[str, tuple[Sign, ...]] = {
    'SH': (POOL,),
    'KH': (SIEVE,),
    'CH': (TETHER,),
    'TH': (BREAD,),
    'PH': (VIPER,),
    'CK': (BASKET,),
}

LETTERS: dict[str, tuple[Sign, ...]] = {
    'A': (VULTURE,),
    'B': (FOOT,),
    'D': (HAND,),
    'E': (REED,),
    'F': (VIPER,),
    'G': (STAND,),
    'H': (SHELTER,),
    'I': (REED,),
    'J': (COBRA,),
    'K': (BASKET,),
    'L': (LION,),
    'M': (OWL,),
    'N': (WATER,),
    'O': (CHICK,),
    'P': (STOOL,),
    'Q': (HILL,),
    'R': (MOUTH,),
    'S': (CLOTH,),
    'T': (BREAD,),
    'U': (CHICK,),
    'V': (VIPER,),
    'W': (CHICK,),
    'X': (BASKET, CLOTH),
    'Z': (BOLT,),
}


def clean_name(name: str) -> str:
    """Upper-case Latin letters of a single given name, accents stripped; ValueError otherwise."""
    folded = unicodedata.normalize('NFKD', name.strip())
    letters = ''.join(c for c in folded if not unicodedata.combining(c)).upper()
    if not re.fullmatch(r'[A-Z]+', letters):
        raise ValueError('a name must be one word of Latin letters (A to Z), without spaces, digits or symbols')

    if not MIN_LETTERS <= len(letters) <= MAX_LETTERS:
        raise ValueError(f'a name must have {MIN_LETTERS} to {MAX_LETTERS} letters')

    return letters


def letter_signs(name: str, i: int) -> tuple[int, tuple[Sign, ...]]:
    """Signs for the sound at position i, and how many letters it used."""
    pair = name[i:i + 2]
    if pair in DIGRAPHS:
        return 2, DIGRAPHS[pair]

    letter = name[i]
    nxt = name[i + 1] if i + 1 < len(name) else ''
    if letter == 'C':
        return 1, (CLOTH,) if nxt in 'EIY' and nxt else (BASKET,)

    if letter == 'Y':
        return 1, (REEDS,) if i == 0 and nxt in VOWELS and nxt else (REED,)

    return 1, LETTERS[letter]


def spell_name(name: str) -> tuple[str, list[Sign]]:
    """The cleaned name and its signs; a doubled letter is written once."""
    letters = clean_name(name)
    signs: list[Sign] = []
    i = 0
    while i < len(letters):
        if i > 0 and letters[i] == letters[i - 1]:
            i += 1
            continue

        used, found = letter_signs(letters, i)
        signs.extend(found)
        i += used

    return letters, signs
