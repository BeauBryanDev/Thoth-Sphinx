import json

from evidence_store import summarize as sm


def test_rows_flatten_each_test_and_keep_the_status(tmp_path):
    report = {'modes': {'hybrid': {'queries': 3, 'hit@5': 0.5}}, 'loose_modes': {'hybrid': {}}}
    (tmp_path / 'retrieval_haiku_unclean_drafts.json').write_text(json.dumps(report))
    table = list(sm.rows(sm.load_runs(tmp_path)))
    assert [row['test'] for row in table] == ['strict', 'loose']
    assert table[0]['status'] == 'invalid'
    assert table[0]['hit@5'] == 0.5
    assert table[1]['hit@5'] is None


def test_missing_reports_are_skipped(tmp_path):
    assert sm.load_runs(tmp_path) == []
