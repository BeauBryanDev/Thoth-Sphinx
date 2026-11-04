import pytest

from evidence_store.books.headings import clean_title, plausible, split_page


@pytest.mark.parametrize(
    ('raw', 'expected'),
    [
        ('C O N T E N T S', 'Contents'),
        ('TEXTS TRANS LA TED', 'Texts Translated'),
        ('F UNERARY RITUALS (P HARAONIC P ERIOD)', 'Funerary Rituals (pharaonic Period)'),
        ("KHAFRA'S  TALE  ....  10", "Khafra's Tale"),
        ('The History of theMS.', 'The History of the MS'),
        ('X.TheMythsofEa', 'X. The Myths of Ea'),
        ('Chapter in. Thoth and the Enneads', 'Chapter III. Thoth and the Enneads'),
    ],
)
def test_clean_title_repairs_ocr_and_drops_junk(raw: str, expected: str) -> None:
    """Letter spacing, split words, small caps, glued words and page numbers come out as plain titles."""
    assert clean_title(raw) == expected
    assert split_page('Thoth as god of the dead i36') == ('Thoth as god of the dead', 136)
    assert not plausible('part of the design, covered by a stone block')
    assert not plausible('Printed by Gilbert and Rivington')
