from collections import Counter

from vision.sphinx_corrector import UNREAD_PENALTY, correct, fuzzy_distance, viterbi_segment
from vision.sphinx_trie import SourceRecord, SphinxTrie

UNIGRAMS = Counter({'D21': 5, 'G17': 5, 'N35': 5, 'Q9': 1, 'Q8': 1})


def tiny_trie() -> SphinxTrie:
    trie = SphinxTrie()
    for codes, translit in (('D21', 'r'), ('G17', 'm'), ('N35', 'n'), ('D21 N35', 'rn')):
        trie.insert(codes.split(), SourceRecord(source='test', translit=translit, translation=''))
    return trie


def segment(codes: list[str]) -> tuple[list[tuple[list[str], list, int]], bool]:
    # No fuzzy matching and no priors, so only exact words and unread steps compete.
    return viterbi_segment(codes, tiny_trie(), {}, UNIGRAMS, fuzzy_max_dist=0,
                           determinatives=set(), particles=[])


def test_unread_penalty_is_below_any_real_word() -> None:
    assert UNREAD_PENALTY < 0


def test_unknown_sign_no_longer_blocks_the_words_around_it() -> None:
    path, had_fallback = segment(['D21', 'Q9', 'G17'])
    assert [(codes, [r.translit for r in recs]) for codes, recs, _ in path] == [
        (['D21'], ['r']), (['Q9'], []), (['G17'], ['m'])]
    assert had_fallback is False


def test_unread_signs_keep_yolo_order() -> None:
    path, _ = segment(['Q8', 'G17', 'Q9', 'N35'])
    assert [code for codes, _, _ in path for code in codes] == ['Q8', 'G17', 'Q9', 'N35']


def test_real_word_beats_unread_steps() -> None:
    path, _ = segment(['D21', 'N35'])
    assert [(codes, [r.translit for r in recs]) for codes, recs, _ in path] == [
        (['D21', 'N35'], ['rn'])]


def test_fallback_means_nothing_was_read() -> None:
    path, had_fallback = segment(['Q9', 'Q8'])
    assert [codes for codes, _, _ in path] == [['Q9'], ['Q8']]
    assert had_fallback is True


def test_correct_passes_unread_signs_through_as_codes() -> None:
    slots = [[('D21', 0.9)], [('Q9', 0.9)], [('G17', 0.9)]]
    result = correct(slots, tiny_trie(), {}, UNIGRAMS, fuzzy_max_dist=0)
    assert result.flat_corrected_seq == ['D21', 'Q9', 'G17']
    assert result.flat_translit == 'r Q9 m'
    assert result.had_fallback is False


def test_fuzzy_edits_grow_with_segment_length() -> None:
    assert [fuzzy_distance(span, 2) for span in (1, 2, 3, 4, 5, 6, 9)] == [0, 0, 0, 1, 1, 2, 2]
    assert fuzzy_distance(9, 0) == 0


def test_short_segment_is_not_fuzzy_matched() -> None:
    # 'D21 Q9' is one edit from the word 'D21 N35', but 2-sign segments must match exactly.
    path, _ = viterbi_segment(['D21', 'Q9'], tiny_trie(), {}, UNIGRAMS,
                              determinatives=set(), particles=[])
    assert [(codes, [r.translit for r in recs], edit) for codes, recs, edit in path] == [
        (['D21'], ['r'], 0), (['Q9'], [], 0)]


def test_rejected_fuzzy_chunk_is_reread_for_real_words() -> None:
    trie = tiny_trie()
    trie.insert('D21 G17 N35 X1'.split(), SourceRecord(source='test', translit='long', translation=''))
    codes = ['D21', 'G17', 'N35', 'Q9']

    # The Viterbi first prefers the 1-edit fuzzy chunk...
    path, _ = viterbi_segment(codes, trie, {}, UNIGRAMS, determinatives=set(), particles=[])
    assert [(seg, edit) for seg, _, edit in path] == [(codes, 1)]

    # ...the gate rejects it (YOLO is confident), and the span is re-read exactly.
    result = correct([[(code, 0.95)] for code in codes], trie, {}, UNIGRAMS)
    assert result.flat_corrected_seq == codes
    assert [w.translit for w in result.segmented_words] == ['r', 'm', 'n', 'Q9']
