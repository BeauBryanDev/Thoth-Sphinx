import copy

import numpy as np
import pytest

from app.core.guards import fence_user_text
from app.schemas.reading_chat import ReadingChatTurn, ReadingLine, ReadingSnapshot, SignCorrection
from app.schemas.transliterations import CorpusMatch, LineEvidence
from app.services import reading_chat
from app.services.reading_chat import ReadingChatService, snapshot_text, valid_corrections
from app.services.sphinx_chat import SPHINX_PERSONA
from tests.test_llm import StubMessages, _response, stub  # noqa: F401  (stub is a fixture)


def _reading() -> ReadingSnapshot:
    """A small reading in the shape the frontend sends."""
    return ReadingSnapshot(
        lines=[ReadingLine(line=1, codes=['S34', 'G5', 'E1', 'D40', 'X1'],
                           transliteration='anx Hr kA-nxt xa', gloss='Live the Horus', confidence='MEDIUM'),
               ReadingLine(line=2, codes=['M23', 'L2'], transliteration='nsw-bjtj', confidence='HIGH')],
        full_transliteration='anx Hr kA-nxt xa | nsw-bjtj',
        full_translation='Live the Horus, Strong Bull; the King of Upper and Lower Egypt.',
        cartouches=['mn-xpr-ra — Thutmose III (Dynasty XVIII)'],
        evidence=[LineEvidence(line=1, verdict='strong', searched_by='claude-sonnet-5-5',
                               best=CorpusMatch(sentence_id='IBUBd3O', rating='strong',
                                                transliteration='Ḥr,w Kꜣ-nḫt-ḫꜥ-m-Wꜣs,t',
                                                date='Sethos I.'))],
        image_notes=['Sunk relief with red paint.'], model='claude-sonnet-5-5', escalated=True,
        escalation_reasons=['haiku_requested_second_opinion'],
        second_opinion_reason='line 1 epithet unclear', direction='ltr', layout='rows')


def _reply(text: str = 'The Horus name is attested.', corrections: list | None = None) -> object:
    return _response(parsed=reading_chat._ChatReply(reply=text, corrections=corrections or []))


def test_snapshot_numbers_signs_and_carries_the_evidence() -> None:
    text = snapshot_text(_reading())
    assert 'Line 1 — signs: [1]S34 [2]G5 [3]E1 [4]D40 [5]X1' in text
    assert 'corpus: strong — closest: Ḥr,w Kꜣ-nḫt-ḫꜥ-m-Wꜣs,t (Sethos I., IBUBd3O)' in text
    assert ('second opinion: yes (haiku_requested_second_opinion); '
            'the first reader asked because: line 1 epithet unclear') in text
    assert 'Thutmose III' in text and 'Sunk relief' in text and 'context: none given' in text


def test_turn_sends_snapshot_first_and_keeps_the_prefix_stable(stub: StubMessages) -> None:
    calls: list[dict] = []

    def parse(**kwargs: object) -> object:
        calls.append(copy.deepcopy(kwargs))
        return _reply()

    stub.parse = parse  # type: ignore[method-assign]
    svc, reading = ReadingChatService(), _reading()
    photo = np.zeros((40, 60, 3), np.uint8)
    out1 = svc.reply('Why did you escalate?', [], reading, photo)
    history = [ReadingChatTurn(role='assistant', content='I read 2 lines.'),
               ReadingChatTurn(role='user', content='Why did you escalate?'),
               ReadingChatTurn(role='assistant', content=out1.reply)]
    svc.reply('Is line 1 in the corpus?', history, reading, photo)

    first, second = calls[0], calls[1]
    assert first['model'] == 'claude-haiku-5-5' and first['output_config'] == {'effort': 'low'}
    assert first['system'][0]['text'].startswith(SPHINX_PERSONA)       # persona kept word for word
    assert [t['name'] for t in first['tools']] == ['search_parallels', 'lookup_lemma', 'lookup_entity', 'search_books']
    image, snap = first['messages'][0]['content']
    assert image['type'] == 'image' and snap['cache_control'] == {'type': 'ephemeral'}
    # Turn 2 repeats the same cached prefix, then the conversation, then the new question.
    assert second['messages'][0] == first['messages'][0] and second['system'] == first['system']
    assert [m['role'] for m in second['messages']] == ['user', 'assistant', 'user', 'assistant', 'user']
    assert second['messages'][-1]['content'][0]['text'] == fence_user_text('Is line 1 in the corpus?')
    assert out1.reply == 'The Horus name is attested.' and out1.corrections == []


def test_history_is_capped(stub: StubMessages, monkeypatch: pytest.MonkeyPatch) -> None:
    from app.core.config import settings
    monkeypatch.setattr(settings, 'reading_chat_history', 2)
    stub.response = _reply()
    history = [ReadingChatTurn(role='user' if i % 2 else 'assistant', content=f'm{i}') for i in range(9)]
    ReadingChatService().reply('next', history, _reading())
    msgs = stub.calls[0][1]['messages']
    # m7 is the visitor's (fenced), m8 is ours (only cleaned).
    assert [m['content'] for m in msgs[1:-1]] == [fence_user_text('m7'), 'm8']
    assert msgs[-1]['content'][0]['text'] == fence_user_text('next')   # the question, cached


def test_corrections_are_validated_against_the_reading() -> None:
    reading = _reading()
    proposed = [
        SignCorrection(line=1, position=5, from_code='X1', to_code='n28', reason='hill-sun sign'),
        SignCorrection(line=1, position=9, from_code='X1', to_code='N28', reason='no such position'),
        SignCorrection(line=1, position=2, from_code='G17', to_code='G5', reason='wrong from_code'),
        SignCorrection(line=2, position=1, from_code='M23', to_code='not a code', reason='bad code'),
        SignCorrection(line=3, position=1, from_code='S34', to_code='S34', reason='no such line'),
        SignCorrection(line=2, position=2, from_code='L2', to_code='L2', reason='no change'),
    ]
    kept = valid_corrections(proposed, reading)
    assert [(c.line, c.position, c.from_code, c.to_code) for c in kept] == [(1, 5, 'X1', 'N28')]


def test_reply_returns_valid_corrections_only(stub: StubMessages) -> None:
    stub.response = _reply('You are right: that is N28, xa.', [
        SignCorrection(line=1, position=5, from_code='X1', to_code='N28', reason='hill with rising sun'),
        SignCorrection(line=1, position=1, from_code='G5', to_code='S34', reason='wrong from_code')])
    out = ReadingChatService().reply('Sign 5 on line 1 is the hill sign, not bread.', [], _reading())
    assert [c.to_code for c in out.corrections] == ['N28'] and out.cost_usd is not None


def test_endpoint_503_without_key(client, no_api_key) -> None:  # noqa: ANN001
    r = client.post('/chat/reading/', json={'prompt': 'hi', 'reading': _reading().model_dump()})
    assert r.status_code == 503


def test_endpoint_rejects_empty_reading_and_bad_image(client, stub: StubMessages) -> None:  # noqa: ANN001
    r = client.post('/chat/reading/', json={'prompt': 'hi', 'reading': {'lines': []}})
    assert r.status_code == 422
    r = client.post('/chat/reading/', json={'prompt': 'hi', 'reading': _reading().model_dump(),
                                            'image': 'data:image/png;base64,eA=='})
    assert r.status_code == 422 and 'image' in r.json()['detail']
    r = client.post('/chat/reading/', json={'prompt': '', 'reading': _reading().model_dump()})
    assert r.status_code == 422


def test_endpoint_happy_path_and_llm_failure(client, stub: StubMessages) -> None:  # noqa: ANN001
    stub.response = _reply('Hello, seeker.')
    r = client.post('/chat/reading/', json={'prompt': 'hi', 'reading': _reading().model_dump()})
    assert r.status_code == 200 and r.json()['reply'] == 'Hello, seeker.' and r.json()['message_id']
    stub.response = _response(stop_reason='refusal')
    r = client.post('/chat/reading/', json={'prompt': 'hi', 'reading': _reading().model_dump()})
    assert r.status_code == 502
