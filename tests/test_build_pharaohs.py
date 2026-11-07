import pytest

bs4 = pytest.importorskip('bs4')

from knowledge import build_pharaohs as bp  # noqa: E402  (after the bs4 skip on purpose)

PAGE = '''<div class="mw-parser-output">
<h2>New Kingdom (c. 1550–1069 BC)</h2><h3>Eighteenth Dynasty</h3>
<table class="wikitable"><tr><th>#</th><th>Image</th><th>Personal name</th><th>Throne name</th>
<th>Notes</th><th>Reign<sup>[1]</sup></th></tr>
<tr><th>1</th><td></td><td><a href="/wiki/Ahmose_I"><b>Ahmose I</b></a><br>Amosis</td>
<td>Nebpehtire</td><td>Founder.<sup>[2]</sup></td><td rowspan="2">c. 1540–1515 BC (~25 years)</td></tr>
<tr><th>2</th><td></td><td><b>Amenhotep I</b></td><td>–</td><td>Son.</td></tr>
<tr><td colspan="6">The position of the following pharaohs is unknown:</td></tr>
<tr><th>3</th><td></td><td colspan="4">Name lost</td></tr>
<tr><td colspan="6">End of Amarna period</td></tr>
<tr><th>4</th><td></td><td>Ay</td><td></td><td></td><td>18 years</td></tr>
<tr><th>5</th><td></td><td></td><td><b>Userkaf</b></td><td></td><td></td></tr>
<tr><th>6</th><td></td><td>[...] <b>tere</b> [...]</td><td></td><td></td><td></td></tr></table>
<table class="wikitable"><tr><th>Pharaoh</th><th>Manetho</th></tr>
<tr><td>Ahmose I</td><td>Amosis</td></tr></table>
<table class="wikitable"><tr><th>Name</th></tr><tr><td>'Anat-Har</td></tr></table>
<h4>Native rebels</h4>
<table class="wikitable"><tr><th>#</th><th>Personal name</th><th>Reign</th></tr>
<tr><th>1</th><td>Inaros</td><td>460 BC</td></tr></table>
</div>'''


@pytest.fixture(scope='module')
def records():
    return bp.extract_records(bs4.BeautifulSoup(PAGE, 'html.parser'))


def by_name(records, name):
    return next(r for r in records if r['king_name'] == name)


def test_comparison_tables_are_skipped_and_banners_are_not_rulers(records):
    assert [r['king_name'] for r in records] == ['Ahmose I', 'Amenhotep I', 'Name lost', 'Ay',
                                                 'Userkaf', '[...] tere [...]', "'Anat-Har",
                                                 'Inaros']


def test_a_ruler_known_only_by_throne_name_is_kept(records):
    ruler = by_name(records, 'Userkaf')
    assert ruler['throne_name'] == 'Userkaf'
    assert ruler['personal_name'] is None


def test_a_damaged_name_keeps_the_whole_cell(records):
    ruler = by_name(records, '[...] tere [...]')
    assert ruler['name_status'] == 'partial'


def test_bold_name_wins_and_the_full_cell_is_kept(records):
    ruler = by_name(records, 'Ahmose I')
    assert ruler['name_details'] == 'Ahmose I Amosis'
    assert ruler['wikipedia_title'] == 'Ahmose I'
    assert ruler['notes'] == 'Founder.'


def test_a_merged_reign_cell_reaches_both_rulers(records):
    assert by_name(records, 'Amenhotep I')['start_year'] == -1540
    assert by_name(records, 'Amenhotep I')['throne_name'] is None


def test_a_merged_name_cell_does_not_fill_other_columns(records):
    ruler = by_name(records, 'Name lost')
    assert ruler['name_status'] == 'lost'
    assert ruler['notes'] is None and ruler['reign_period'] is None


def test_banners_mark_rows_until_an_end_banner(records):
    assert by_name(records, 'Name lost')['group_note'].startswith('The position')
    assert by_name(records, 'Ay')['group_note'] is None


def test_dynasty_period_and_role_come_from_headings(records):
    ruler = by_name(records, 'Ahmose I')
    assert ruler['dynasty'] == 'Eighteenth Dynasty'
    assert ruler['dynasty_numbers'] == [18]
    assert ruler['period_years'] == {'start': -1550, 'end': -1069}
    assert by_name(records, 'Inaros')['role'] == 'rebel'


def test_a_famous_throne_name_replaces_an_unlinked_horus_name():
    html = '''<h3>Fourth Dynasty</h3><table class="wikitable">
    <tr><th>#</th><th>Image</th><th>Horus name</th><th>Throne name</th></tr>
    <tr><th>1</th><td></td><td><b>Medjedu</b></td><td><a href="/wiki/Khufu"><b>Khufu</b></a></td></tr>
    <tr><th>2</th><td></td><td><a href="/wiki/Sanakht"><b>Sanakht</b></a></td><td>Nebka</td></tr>
    </table>'''
    first, second = bp.extract_records(bs4.BeautifulSoup(html, 'html.parser'))
    assert (first['king_name'], first['horus_name'], first['throne_name']) == ('Khufu', 'Medjedu',
                                                                               'Khufu')
    assert first['wikipedia_title'] == 'Khufu'
    assert second['king_name'] == 'Sanakht'


def test_ids_are_unique(records):
    assert len({r['id'] for r in records}) == len(records)


@pytest.mark.parametrize('text, start, end', [
    ('31 May 1279– 13 August 1213 BC (66 years, 74 days)', -1279, -1213),
    ('c. March 51– 12 August 30 BC (21 years)', -51, -30),
    ('c. 273/2–July 270/68 BC', -273, -270),
    ('September 107– c. May 88 BC', -107, -88),
    ('1213–1203 (9 years, x months)', -1213, -1203),
    ('30 BC – 14 AD', -30, 14),
    ('460 BC', -460, -460),
])
def test_parse_reign_reads_years(text, start, end):
    dates = bp.parse_reign(text)
    assert (dates['start_year'], dates['end_year']) == (start, end)


@pytest.mark.parametrize('text', ['18 years', 'c. 64 years', '8–10 years', '2 + 1⁄2 years',
                                  'a few months', 'lost', ''])
def test_reign_lengths_are_never_dates(text):
    assert bp.parse_reign(text)['start_year'] is None


def test_alternative_dates_are_flagged():
    assert bp.parse_reign('c. 750–745 BC or 720–715 BC')['alternate_dates'] is True


@pytest.mark.parametrize('heading, numbers', [
    ('Thirty-third Dynasty (Ptolemaic)', [33]), ('Seventh and Eighth Dynasties', [7, 8]),
    ('Twentieth Dynasty (Ramesside)', [20]), ('Abydos Dynasty', []),
])
def test_dynasty_numbers(heading, numbers):
    assert bp.dynasty_numbers(heading) == numbers
