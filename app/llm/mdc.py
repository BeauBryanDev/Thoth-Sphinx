import re
import unicodedata

from corpus.extract import normalize_translit

# Manuel de Codage letters -> the Unicode Leiden letters the AAEW corpus uses (case matters in MdC).
MDC_TO_UNICODE: dict[str, str] = {
    'A': 'ꜣ', 'a': 'ꜥ', 'i': 'j', 'H': 'ḥ', 'x': 'ḫ', 'X': 'ẖ',
    'S': 'š', 'T': 'ṯ', 'D': 'ḏ', 'q': 'q', 'Q': 'q',
}

# Any of these means the text is already Unicode transliteration and must not be converted:
# the Leiden letters, plus the combining breve and macron below (jri̯, ẖ) that only Unicode uses.
UNICODE_LETTERS = frozenset('ꜣꜢꜥꜤḥḤḫḪẖšŠṯṮḏḎḳḲ̯̱')

# Claude's apparatus that the index never contains: line separators, ellipses, gap markers.
SEPARATOR_RE = re.compile(r'[|/]+')
GAP_RE = re.compile(r'\[\s*\.{2,}\s*\]|\.{3,}|…')


def is_unicode_translit(text: str) -> bool:
    """True when the text already uses Unicode Leiden letters."""
    # Normalize the text to NFC form (canonical decomposition, followed by canonical composition)
    # and check if any of the characters are in the set of Unicode Leiden letters
    return any(ch in UNICODE_LETTERS for ch in unicodedata.normalize('NFC', text))


def mdc_to_unicode(text: str) -> str:
    """Convert MdC letters to Unicode Leiden; other ASCII letters are lowercased."""
    # Initialize an empty list to store the output characters
    out: list[str] = []
    for ch in text:
        # If the character is in the mapping, append the corresponding Unicode Leiden letter
        if ch in MDC_TO_UNICODE:
            # If the character is in the mapping, append the corresponding Unicode Leiden letter
            out.append(MDC_TO_UNICODE[ch])
        # If the character is an ASCII letter, convert it to lowercase and append the corresponding Unicode Leiden letter
        elif ch.isascii() and ch.isalpha(): 
            # Capitals outside MdC only mark proper names (Ra, Wsjr): fold, then map (I -> i -> j).
            # Convert the character to lowercase and append the corresponding Unicode Leiden letter
            low = ch.lower()
            out.append(MDC_TO_UNICODE.get(low, low))
        else:
            # If the character is not in the mapping, append the character itself
            out.append(ch)
    # Join the list of characters into a string and return it
    return ''.join(out)


def translit_query(text: str | None) -> str:
    """A transliteration in either convention, cleaned the way the index was built."""
    # If the text is None or empty, return an empty string
    if not text or not text.strip():
        # Return an empty string
        return ''
    # Replace the separators and gap markers with spaces
    text = SEPARATOR_RE.sub(' ', GAP_RE.sub(' ', text))
    # If the text is not already Unicode transliteration, convert it to Unicode Leiden
    if not is_unicode_translit(text):
        text = mdc_to_unicode(text)
    # Normalize the text to NFC form (canonical decomposition, followed by canonical composition)
    # and join the words with spaces
    return ' '.join(normalize_translit(text).split())


