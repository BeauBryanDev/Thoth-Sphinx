"""Router tests via TestClient with a stub pipeline (no ONNX, no network)."""
from __future__ import annotations


def test_health_live(client):
    r = client.get('/health/live')
    assert r.status_code == 200 and r.json()['status'] == 'ok'


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    body = r.json()
    assert body['classes'] == 150


def test_health_ready(client):
    assert client.get('/health/ready').status_code == 200


def test_health_ready_503_without_pipeline(client):
    client.app.state.pipeline = None
    assert client.get('/health/ready').status_code == 503


def test_predict_happy_path(client, tiny_jpeg):
    r = client.post('/predict/',
                    files={'file': ('t.jpg', tiny_jpeg, 'image/jpeg')},
                    data={'layout': 'columns', 'direction': 'rtl'})
    assert r.status_code == 200
    body = r.json()
    assert body['n_cartouches'] == 1
    assert body['cartouches'][0]['translit'] == 'wnjs'
    # raw interior codes exposed even for matched cartouches (and REFUSED
    # ones — the frontend forwards them to the LLM instead of dropping them)
    assert body['cartouches'][0]['interior_codes'] == ['N5', 'L1', 'N35']
    assert body['annotated_image'].startswith('data:image/jpeg;base64,')


def test_predict_requires_layout(client, tiny_jpeg):
    r = client.post('/predict/',
                    files={'file': ('t.jpg', tiny_jpeg, 'image/jpeg')})
    assert r.status_code == 422


def test_predict_rejects_bad_direction_layout_preset(client, tiny_jpeg):
    for data in ({'layout': 'diagonal', 'direction': 'rtl'},
                 {'layout': 'rows', 'direction': 'boustrophedon'},
                 {'layout': 'rows', 'direction': 'rtl', 'preset': 'ultra'}):
        r = client.post('/predict/',
                        files={'file': ('t.jpg', tiny_jpeg, 'image/jpeg')},
                        data=data)
        assert r.status_code == 422, data


def test_predict_rejects_bad_context_vocab(client, tiny_jpeg):
    r = client.post('/predict/',
                    files={'file': ('t.jpg', tiny_jpeg, 'image/jpeg')},
                    data={'layout': 'rows', 'direction': 'rtl',
                          'period': 'bronze_age'})
    assert r.status_code == 422


def test_transliterate_empty_codes_422(client):
    r = client.post('/transliterate/', json={'codes': []})
    assert r.status_code == 422


def test_transliterate_no_key_degrades(client, no_api_key):
    r = client.post('/transliterate/', json={
        'codes': ['G17', 'N35'],
        'confidences': [0.9],           # misaligned -> neutral 0.5
    })
    assert r.status_code == 200
    assert 'ANTHROPIC_API_KEY' in r.json()['error']


def test_chat_503_without_key(client, no_api_key):
    r = client.post('/chat/', json={'prompt': 'hello'})
    assert r.status_code == 503


def test_reverse_503_without_key(client, no_api_key):
    r = client.post('/reverse/', json={'text': 'life, prosperity, health'})
    assert r.status_code == 503
