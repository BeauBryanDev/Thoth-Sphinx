from corpus.books.to_markdown import page_markdown


def test_raw_page_keeps_citations_and_rebuilds_paragraphs() -> None:
    raw = (
        'birds also feature within these catacombs, but to a\n'
        'much lesser extent (Kessler and Nur el-Din 2005, pp.\n'
        '152–54; von den Driesch et al. 2005, pp. 205, 216–17).\n'
        'The two most famous sites with bird catacombs are\n'
        'Tuna el-Gebel in Middle Egypt, and the royal burial\n'
        'ground at Saqqara, close to Memphis.\n'
        'Near Saqqara the area bordering the Abusir Lake\n'
        'has been posited as a possible ibis habitat.\n'
    )
    book = {
        'title': 'Birds',
        'sections': [],
    }

    text = page_markdown(raw, book, set(), pdf_order=True)

    assert '152–54; von den Driesch et al. 2005, pp. 205, 216–17).' in text
    assert text.split('\n\n') == [
        'birds also feature within these catacombs, but to a much lesser extent '
        '(Kessler and Nur el-Din 2005, pp. 152–54; von den Driesch et al. 2005, pp. 205, 216–17). '
        'The two most famous sites with bird catacombs are Tuna el-Gebel in Middle Egypt, '
        'and the royal burial ground at Saqqara, close to Memphis.',
        'Near Saqqara the area bordering the Abusir Lake has been posited as a possible ibis habitat.',
    ]
