# Pure scoring functions for the agent eval: no I/O, no LLM calls.
# The four headline checks are lenient on purpose; see the eval plan.

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any

# Thresholds set before the first run. Do not tune them to the results.
TRANSLIT_PASS = 0.70
FACTS_PASS = 0.50

# Unicode transliteration to Manuel de Codage, the golden file's notation.
UNICODE_TO_MDC = {
    'ꜣ': 'A', 'ꜥ': 'a', 'ḥ': 'H', 'ḫ': 'x', 'ẖ': 'X', 'š': 'S',
    'ṯ': 'T', 'ḏ': 'D', 'ḳ': 'q', 'ꞽ': 'j', 'ỉ': 'j', 'ı': 'j',
    'i': 'j', 'y': 'j', 'Y': 'j', 'z': 's', 'Z': 's',
    'Ꜣ': 'A', 'Ꜥ': 'a', 'Ḥ': 'H', 'Ḫ': 'x', 'H̱': 'X', 'Š': 'S',
    'Ṯ': 'T', 'Ḏ': 'D', 'Ḳ': 'q', 'I': 'j',
}
# Capitals that are letters of their own in Manuel de Codage.
MDC_CAPITALS = set('AHXSTD')
SEPARATORS = re.compile(r'[-.=:·,;/|]+')
EDITORIAL = re.compile(r'[\[\](){}<>⸢⸣〈〉?⸮!*"\'’…]')


def normalize_translit(text: str) -> str:
    """One notation for comparing: MdC letters, spaces between words."""
    for src, dst in UNICODE_TO_MDC.items():
        text = text.replace(src, dst)

    text = EDITORIAL.sub('', text)
    text = SEPARATORS.sub(' ', text)

    letters: list[str] = []
    for ch in text:
        if ch.isalpha() and ch not in MDC_CAPITALS:
            ch = ch.lower()

        # Combining marks and digits carry nothing to compare.
        if ch.isalpha() or ch.isspace():
            letters.append(ch)

    return ' '.join(''.join(letters).split())


def translit_similarity(truth: str, reading: str) -> float:
    """0 to 1. The better of a character match and a word-overlap score,
    so a different column order is not punished."""
    a = normalize_translit(truth)
    b = normalize_translit(reading)
    if not a or not b:
        return 0.0

    chars = SequenceMatcher(
        None, a.replace(' ', ''), b.replace(' ', ''), autojunk=False
    ).ratio()

    truth_words = Counter(a.split())
    read_words = Counter(b.split())
    shared = sum((truth_words & read_words).values())
    words = 2 * shared / (sum(truth_words.values()) + sum(read_words.values()))

    return max(chars, words)


def spellings(name: str) -> set[str]:
    """The name with its -re and -ra endings swapped: Usermaatre, -ra."""
    return {
        name,
        re.sub(r're\b', 'ra', name),
        re.sub(r'ra\b', 're', name),
    }


def translit_coverage(truth: str, reading: str) -> float:
    """0 to 1: how much of the truth the reading contains. Text the
    reading has beyond the truth is not counted against it."""
    a = normalize_translit(truth)
    b = normalize_translit(reading)
    if not a or not b:
        return 0.0

    truth_chars = a.replace(' ', '')
    matcher = SequenceMatcher(
        None, truth_chars, b.replace(' ', ''), autojunk=False
    )
    matched = sum(block.size for block in matcher.get_matching_blocks())

    truth_words = Counter(a.split())
    shared = sum((truth_words & Counter(b.split())).values())

    return max(matched / len(truth_chars), shared / sum(truth_words.values()))


def mentions(text: str, name: str, whole: bool = True) -> bool:
    """The name appears in the text. whole: Thutmose II is not in
    Thutmose III; otherwise only the start must be a word start."""
    for variant in spellings(name):
        pattern = r'(?<![\w])' + re.escape(variant)
        if whole:
            pattern += r'(?![\w])'

        if re.search(pattern, text, flags=re.IGNORECASE):
            return True

    return False


def kings_found(
    record: dict[str, Any],
    text: str,
    translit: str,
) -> dict[str, bool]:
    """Per true king: named in the English text or the transliteration."""
    plain = normalize_translit(translit).replace(' ', '')
    found: dict[str, bool] = {}

    for cartouche in record['cartouches']:
        king = cartouche['king']
        hit = found.get(king, False)

        for alias in cartouche['aliases']:
            if mentions(text, alias):
                hit = True

        name = normalize_translit(cartouche['translit']).replace(' ', '')
        if name and name in plain:
            hit = True

        found[king] = hit

    return found


def invented_kings(
    record: dict[str, Any],
    text: str,
    catalog: dict[str, set[str]],
) -> list[str]:
    """Kings of the catalog named in the text that are not on this wall."""
    true_names: set[str] = set()
    for cartouche in record['cartouches']:
        true_names.add(cartouche['king'].lower())
        true_names.update(a.lower() for a in cartouche['aliases'])

    invented: list[str] = []
    for king, names in catalog.items():
        if any(n.lower() in true_names for n in names):
            continue

        for name in names:
            # A name shared with a true king (Usermaatre is Ramesses II
            # and III) proves nothing.
            if any(name.lower() in true for true in true_names):
                continue

            if mentions(text, name):
                invented.append(king)
                break

    return invented


def facts_recall(record: dict[str, Any], text: str) -> tuple[float, list[str]]:
    """Share of key facts stated, and the ones that are missing."""
    facts = record.get('key_facts', [])
    missing: list[str] = []

    for fact in facts:
        stated = False
        for wording in fact['accept']:
            if mentions(text, wording, whole=False):
                stated = True

        if not stated:
            missing.append(fact['fact'])

    if not facts:
        return 1.0, missing

    return 1 - len(missing) / len(facts), missing


def verdict(
    kings: dict[str, bool],
    invented: list[str],
    similarity: float,
    recall: float,
    errored: bool,
) -> str:
    """pass, partial or fail from the four headline checks."""
    if errored or invented or not all(kings.values()):
        return 'fail'

    if similarity >= TRANSLIT_PASS and recall >= FACTS_PASS:
        return 'pass'

    return 'partial'


def edit_distance(a: list[str], b: list[str]) -> int:
    previous = list(range(len(b) + 1))

    for i, item_a in enumerate(a, 1):
        current = [i]
        for j, item_b in enumerate(b, 1):
            cost = 0 if item_a == item_b else 1
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            ))

        previous = current

    return previous[-1]


def truth_codes(record: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for line in record['lines']:
        codes.extend(line['codes'])

    return codes


def vision_diagnostics(
    record: dict[str, Any],
    predict: dict[str, Any],
) -> dict[str, Any]:
    """How close the detector's outer codes are to the true ones.
    Diagnostic only: order errors and out-of-class signs both count."""
    truth = truth_codes(record)
    detected = predict['outer']['correction']['flat_corrected_seq']
    oov = set(record.get('oov_signs', []))

    shared = sum((Counter(truth) & Counter(detected)).values())
    reachable = [c for c in truth if c not in oov]
    names_ok = 0
    for cartouche in predict['cartouches']:
        english = cartouche.get('english') or ''
        for true_cartouche in record['cartouches']:
            if mentions(english, true_cartouche['king']):
                names_ok += 1
                break

    return {
        'signs_true': len(truth),
        'signs_detected': len(detected),
        'signs_out_of_class': len(truth) - len(reachable),
        'sign_error_rate': round(
            edit_distance(truth, detected) / len(truth), 3
        ),
        'sign_recall_any_order': round(shared / len(truth), 3),
        'unknown_share': round(
            detected.count('Unknown') / max(len(detected), 1), 3
        ),
        'cartouches_true': len(record['cartouches']),
        'cartouches_found': predict.get('n_cartouches', 0),
        'cartouches_named_right': names_ok,
        'cartouches_refused': sum(
            1 for c in predict['cartouches'] if not c.get('translit')
        ),
    }


def process_diagnostics(reading: dict[str, Any]) -> dict[str, Any]:
    """What the agent did: tools, evidence, escalation, cost."""
    tool_calls = reading.get('tool_calls', [])
    evidence = reading.get('evidence', [])
    verdicts = Counter(e.get('verdict') for e in evidence)

    return {
        'model': reading.get('model'),
        'cost_usd': reading.get('cost_usd'),
        'escalated': reading.get('escalated', False),
        'escalation_reasons': reading.get('escalation_reasons', []),
        'tool_calls': len(tool_calls),
        'tool_calls_failed': sum(1 for t in tool_calls if not t.get('ok')),
        'tools_used': sorted({t.get('tool') for t in tool_calls}),
        'lines_with_evidence': len(evidence),
        'evidence_strong': verdicts.get('strong', 0),
        'evidence_possible': verdicts.get('possible', 0),
        'evidence_none': verdicts.get('none', 0),
        'chunk_confidence': dict(Counter(
            c.get('confidence') for c in reading.get('chunks', [])
        )),
    }
