import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from corpus.books import book_inspector


def test_write_refuses_a_new_entry_with_an_existing_doc_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / 'manifest.json'
    original = json.dumps({'version': 1, 'books': [{'doc_id': 'egyptian-magic', 'path': 'egyptian_magic.pdf'}]}, indent=2)
    manifest.write_text(original)
    monkeypatch.setattr(
        book_inspector,
        'skeleton',
        lambda report: {'doc_id': 'egyptian-magic', 'path': report.path.name, 'sections': []},
    )
    report = SimpleNamespace(path=Path('Egyptian-Magic.pdf'))

    with pytest.raises(SystemExit, match='already in the manifest'):
        book_inspector.write_manifest(report, manifest)
    assert manifest.read_text() == original
