from types import SimpleNamespace

from corpus import draft


class FakeClaude:
    def __init__(self, stop_reason='end_turn'):
        self.stop_reason, self.request = stop_reason, None
        self.messages = self

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            usage=SimpleNamespace(input_tokens=20, output_tokens=5),
            content=[SimpleNamespace(type='thinking'),
                     SimpleNamespace(type='text',
                                     text='{"translation": " Ein Opfer des Königs. "}')])


def test_translate_reads_the_text_block_and_reports_usage():
    api = FakeClaude()
    assert draft.translate('ḥtp-ḏi̯ nswt', api=api) == ('Ein Opfer des Königs.', 20, 5)
    assert api.request['model'] == 'claude-haiku-5-5'
    assert 'German' in api.request['system']
    assert api.request['output_config']['format']['type'] == 'json_schema'


def test_translate_uses_the_requested_language():
    api = FakeClaude()
    draft.translate('nṯr', language='English', api=api)
    assert 'English' in api.request['system']


def test_refusal_or_cut_off_gives_a_blank_draft():
    assert draft.translate('nṯr', api=FakeClaude('refusal'))[0] == ''
    assert draft.translate('nṯr', api=FakeClaude('max_tokens'))[0] == ''


def test_prompt_marks_neighbours_as_context_only():
    assert draft.prompt('nṯr') == 'Sentence: nṯr'
    text = draft.prompt('nṯr', before='ꜥnḫ')
    assert 'context only: ꜥnḫ' in text
    assert '(none)' in text
    assert text.endswith('Translate only this sentence: nṯr')
