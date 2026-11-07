import pytest

from app.core.ratelimit import Cooldown, client_key


def test_one_action_per_window() -> None:
    limit = Cooldown(3600)
    assert limit.reserve('a', now=0) is None
    assert limit.reserve('a', now=10) == 3591        # seconds left, rounded up
    assert limit.reserve('b', now=10) is None         # another client is independent
    assert limit.reserve('a', now=3600) is None       # window passed


def test_release_gives_the_slot_back() -> None:
    limit = Cooldown(3600)
    limit.reserve('a', now=0)
    limit.release('a')
    assert limit.reserve('a', now=1) is None


def test_zero_window_disables_the_limit() -> None:
    limit = Cooldown(0)
    assert limit.reserve('a') is None and limit.reserve('a') is None


def test_client_key_hides_the_ip() -> None:
    key = client_key('203.0.113.7')
    assert key == client_key('203.0.113.7') and '203' not in key and len(key) == 16
    assert client_key(None) == client_key(None)


@pytest.fixture()
def fresh_limit(monkeypatch: pytest.MonkeyPatch):
    from app.core.config import settings
    from app.routers import transliterate
    monkeypatch.setattr(settings, 'reread_cooldown_s', 3600)
    monkeypatch.setattr(transliterate, 'REREAD_LIMIT', Cooldown(3600))
    return transliterate.REREAD_LIMIT


BODY = {'codes': ['S34', 'N28'], 'confidences': [0.9, 1.0],
        'human_corrections': [{'line': 1, 'position': 2, 'from_code': 'X1', 'to_code': 'N28'}]}


def test_second_reread_within_the_hour_is_429(client, fresh_limit, monkeypatch) -> None:  # noqa: ANN001
    from app.schemas.transliterations import TransliterationOut
    from app.services.transliteration_services import TransliterationService
    ok = TransliterationOut(chunks=[], full_transliteration='anx', full_translation='life', model='m', n_chunks=1)
    monkeypatch.setattr(TransliterationService, 'transliterate_sequence', lambda self, *a, **k: ok)
    assert client.post('/transliterate/', json=BODY).status_code == 200
    second = client.post('/transliterate/', json=BODY)
    assert second.status_code == 429 and int(second.headers['Retry-After']) > 3500
    assert 'One correction re-read per 60 minutes' in second.json()['detail']
    # A first decode (no corrections) is never limited.
    plain = {k: v for k, v in BODY.items() if k != 'human_corrections'}
    assert client.post('/transliterate/', json=plain).status_code == 200


def test_failed_reread_does_not_burn_the_hour(client, fresh_limit, monkeypatch) -> None:  # noqa: ANN001
    from app.schemas.transliterations import TransliterationOut
    from app.services.transliteration_services import TransliterationService
    failed = TransliterationOut(chunks=[], full_transliteration='', full_translation='', model='m',
                                n_chunks=0, error='LLM call failed')
    monkeypatch.setattr(TransliterationService, 'transliterate_sequence', lambda self, *a, **k: failed)
    assert client.post('/transliterate/', json=BODY).status_code == 200
    assert client.post('/transliterate/', json=BODY).status_code == 200   # slot was released
