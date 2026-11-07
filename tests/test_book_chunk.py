from pathlib import Path

from corpus.books.chunk import (
    MAX_TOKENS,
    OVERLAP_TOKENS,
    chunk_book,
    count,
    read_book,
    split_sentences,
)


def test_chunks_stay_in_size_and_section(tmp_path: Path) -> None:
    """Chunks never pass 640 tokens, never mix sections, overlap at most 64 tokens, keep their pages."""
    sentence = 'Thoth weighed the heart of the scribe against the feather of Maat in the hall of judgement.'
    pages = '\n\n'.join(f'<!-- p. {n} -->\n\n' + ' '.join([sentence] * 12) for n in range(1, 9))
    book = tmp_path / 'test-book.md'
    book.write_text(
        '# Test Book\n\n<!-- doc_id: test-book | author: x -->\n\n'
        f'## Chapter I. The Hall\n\n{pages}\n\n## Chapter II. Short\n\n<!-- p. 9 -->\n\nA short closing chapter on the ibis of Hermopolis, sacred to Thoth, and on its many burials.\n'
    )
    chunks = chunk_book(read_book(book))
    assert all(count(c.text) <= MAX_TOKENS and c.overlap <= OVERLAP_TOKENS for c in chunks)
    assert {c.section for c in chunks} == {'Chapter I. The Hall', 'Chapter II. Short'}
    assert chunks[0].page_start == 'p. 1' and chunks[-1].page_start == 'p. 9'
    assert split_sentences('He reigned c. 1290 B.C.E. in Dyn. XIX, says E. A. Budge. Then he died.') == [
        'He reigned c. 1290 B.C.E. in Dyn. XIX, says E. A. Budge.',
        'Then he died.',
    ]
