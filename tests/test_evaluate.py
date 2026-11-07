import random

import pytest

from corpus import evaluate as ev


def test_select_modes_adds_rerank_and_respects_an_explicit_list():
    assert 'hybrid_rerank_gloss' not in ev.select_modes()
    assert 'hybrid_rerank_gloss' in ev.select_modes(with_rerank=True)
    assert 'trigram_noisy' not in ev.select_modes(skip_trigram=True)
    assert ev.select_modes(with_rerank=True, only=['hybrid_noisy']) == ['hybrid_noisy']


def test_select_modes_adds_every_haiku_draft():
    modes = ev.select_modes(with_haiku=True)
    assert {'embedding_de', 'hybrid_en', 'hybrid_de_context'} <= set(modes)
    assert 'hybrid_de' not in ev.select_modes()


def test_select_modes_rejects_unknown_names():
    with pytest.raises(ValueError):
        ev.select_modes(only=['hybrid_typo'])


def test_longest_shared_run_requires_contiguous_items():
    assert ev.longest_shared_run(list('abcdx'), list('zabcd')) == 4
    assert ev.longest_shared_run(list('axbxc'), list('abc')) == 1
    assert ev.longest_shared_run([], list('abc')) == 0


def test_required_run_scales_with_length_above_the_minimum():
    assert ev.required_run(6) == 4
    assert ev.required_run(11) == 6


def test_score_reads_ranks_and_ignores_hits_after_ten():
    ranked = ['x', 'a', 'y', 'b'] + ['z'] * 6 + ['c']
    result = ev.score(ranked, {'a', 'b', 'c'})
    assert result['hit@1'] == 0.0
    assert result['hit@5'] == 1.0
    assert result['recall@10'] == 2 / 3
    assert result['mrr@10'] == 0.5


def test_score_handles_no_hits():
    assert ev.score(['x'], {'a'})['mrr@10'] == 0.0


def test_noise_is_reproducible_and_changes_the_input():
    tokens = [f's{i}' for i in range(200)]
    first = ev.noisy_signs(tokens, ['q'], random.Random(1))
    assert first == ev.noisy_signs(tokens, ['q'], random.Random(1))
    assert first != tokens
    assert len(first) < len(tokens)


def test_noisy_translit_handles_blank_input():
    assert ev.noisy_translit('', random.Random(1)) == ''


def test_find_parallels_excludes_own_text_copies_and_short_runs():
    sequences = {'q': list('abcdefgh'), 'other_text': list('abcdexyz'), 'same_text': list('abcdefgh'),
                 'copy': list('abcdefgh'), 'too_short': list('abcxyz')}
    sentences = {'q': {'text_id': 't1', 'norm': 'n'}, 'other_text': {'text_id': 't2', 'norm': 'm'},
                 'same_text': {'text_id': 't1', 'norm': 'k'}, 'copy': {'text_id': 't3', 'norm': 'n'},
                 'too_short': {'text_id': 't4', 'norm': 'p'}}
    index = {}
    for sentence_id, sequence in sequences.items():
        for i in range(len(sequence) - 3):
            index.setdefault(tuple(sequence[i:i + 4]), set()).add(sentence_id)
    assert ev.find_parallels('q', sequences, index, sentences) == {'other_text'}


def test_first_sense_takes_the_leading_gloss():
    assert ev.first_sense('vulture; bird, general') == 'vulture'
    assert ev.first_sense('to give, place') == 'to give'


def test_gloss_query_skips_lemmas_without_glosses():
    assert ev.gloss_query(['a', 'b'], {'a': 'give'}, random.Random(3)) in ('give', '')


def test_loose_parallels_match_in_any_order_and_exclude_own_text():
    sequences = {'q': list('abcdef'), 'shuffled': list('fedcba'), 'same_text': list('fedcba'),
                 'weak': list('abxyzw')}
    sentences = {'q': {'text_id': 't1', 'norm': 'n'}, 'shuffled': {'text_id': 't2', 'norm': 'm'},
                 'same_text': {'text_id': 't1', 'norm': 'k'}, 'weak': {'text_id': 't3', 'norm': 'p'}}
    index = ev.build_lemma_index(sequences)
    assert ev.find_loose_parallels('q', sequences, index, sentences) == {'shuffled'}
