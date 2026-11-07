import pytest

from app.llm.mdc import is_unicode_translit, mdc_to_unicode, translit_query


@pytest.mark.parametrize(('mdc', 'unicode'), [
    ('A', 'ꜣ'), ('a', 'ꜥ'), ('i', 'j'), ('H', 'ḥ'), ('x', 'ḫ'), ('X', 'ẖ'),
    ('S', 'š'), ('T', 'ṯ'), ('D', 'ḏ'), ('q', 'q'),
    ('h', 'h'), ('s', 's'), ('t', 't'), ('d', 'd'), ('k', 'k'), ('z', 'z'),
])
def test_each_mdc_letter(mdc: str, unicode: str) -> None:
    assert mdc_to_unicode(mdc) == unicode


def test_proper_name_capitals_fold_then_map() -> None:
    # R, W, I are not MdC letters, so they only mark names: Ra -> rꜥ, Imn -> jmn.
    assert mdc_to_unicode('Ra') == 'rꜥ'
    assert mdc_to_unicode('WAst') == 'wꜣst'
    assert mdc_to_unicode('Imn') == 'jmn'


@pytest.mark.parametrize(('claude', 'query'), [
    # Readings Claude produced on the fixtures.
    ('anx Hr kA-nxt xa-m-WAst', 'ꜥnḫ ḥr kꜣ-nḫt ḫꜥ-m-wꜣst'),
    ('nsw-bjtj mn-xpr-Ra | sA Ra', 'nsw-bjtj mn-ḫpr-rꜥ sꜣ rꜥ'),
    ('dj anx Dt', 'dj ꜥnḫ ḏt'),
    # Apparatus: restorations kept, gaps, doubt marks and line separators dropped.
    ('Htp-dj-nsw [...] (n)Ds(=f)? t', 'ḥtp-dj-nsw nḏs=f t'),
    ('nb ... tAwy', 'nb tꜣwy'),
])
def test_claude_readings(claude: str, query: str) -> None:
    assert translit_query(claude) == query


def test_unicode_input_is_left_alone() -> None:
    assert translit_query('ꜥnḫ ḥr kꜣ-nḫt') == 'ꜥnḫ ḥr kꜣ-nḫt'
    # The breve marks Unicode Leiden even without special letters; converting would turn i into j.
    assert is_unicode_translit('jri̯.n =j')
    assert translit_query('jri̯.n =j') == 'jri̯.n =j'


def test_unicode_input_still_loses_editorial_brackets() -> None:
    assert translit_query('p〈ꜣ〉 [nb]') == 'pꜣ nb'


@pytest.mark.parametrize('empty', [None, '', '   ', '[...]', '...', ' | '])
def test_empty_queries(empty: str | None) -> None:
    assert translit_query(empty) == ''
