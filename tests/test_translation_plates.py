from corpus.translations.extract import pack, speakers, strip_note_letters


def test_plate_translations_keep_speakers_and_drop_note_marks() -> None:
    lines = [
        'Label: Strewing gypsum;a going around threeb times.',
        'By Amun: Words spoken by Amun-Re: “Your son belongs to you. It is he who',
        'issued from your body.a May you give to him all [he]alth, pre-eminent.”',
        'By the King: The lord of the Two Lands USERMAꜤATRE.',
    ]

    blocks = [strip_note_letters(' '.join(group)) for group in speakers(lines)]

    assert pack(blocks) == [
        'Label: Strewing gypsum; going around three times.\n'
        'By Amun: Words spoken by Amun-Re: “Your son belongs to you. It is he who issued from your body. '
        'May you give to him all [he]alth, pre-eminent.”\n'
        'By the King: The lord of the Two Lands USERMAꜤATRE.'
    ]
