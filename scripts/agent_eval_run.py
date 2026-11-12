# Run the agent on the chosen images against a running server and save every
# raw response. No scoring here: scripts/agent_eval_score.py reads the output.
# Two conditions per image: detected (YOLO's signs) and oracle (true signs).

import argparse
import base64
import json
import mimetypes
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

PREDICT_TIMEOUT_S, READ_TIMEOUT_S = 120, 240
CONDITIONS = ('detected', 'oracle')
# Wording the prompt builder keys on; kept in step with glyphApi.ts.
UNRESOLVED_WITH_SIGNS = (
    '[UNRESOLVED cartouche — raw signs detected but no confident '
    'royal-name match: {codes}]'
)
UNRESOLVED_EMPTY = '[UNRESOLVED cartouche — no signs detected]'


def data_url(path: Path) -> str:
    """The raw image as a data URL, the way the browser sends it."""
    mime = mimetypes.guess_type(path.name)[0] or 'image/jpeg'
    encoded = base64.b64encode(path.read_bytes()).decode()

    return f'data:{mime};base64,{encoded}'


def cartouche_label(cartouche: dict[str, Any]) -> str:
    """One cartouche_names entry from a /predict/ cartouche."""
    if cartouche.get('translit'):
        spelling = ' '.join(cartouche.get('spelling') or [])
        english = cartouche.get('english') or ''
        return f"{cartouche['translit']} — {english} (interior: {spelling})"

    codes = ' '.join(cartouche.get('interior_codes') or [])
    if codes:
        return UNRESOLVED_WITH_SIGNS.format(codes=codes)

    return UNRESOLVED_EMPTY


def detected_request(predict: dict[str, Any], image: str) -> dict[str, Any]:
    """The /transliterate/ body the frontend builds from a detection."""
    outer = predict['outer']
    confidences: list[float] = []

    for slot in outer['slots']:
        confidences.append(slot[0][1] if slot else 0.0)

    return {
        'codes': outer['correction']['flat_corrected_seq'],
        'confidences': confidences,
        'boundary_hints': outer['boundary_hints'],
        'cartouche_names': [cartouche_label(c) for c in predict['cartouches']],
        'direction': predict['direction'],
        'layout': predict['layout'],
        'image': image,
    }


def oracle_request(record: dict[str, Any], image: str) -> dict[str, Any]:
    """The same body from the ground truth: the agent with perfect vision."""
    codes: list[str] = []
    line_ends: list[int] = []

    for line in record['lines']:
        codes.extend(line['codes'])
        line_ends.append(len(codes))

    names: list[str] = []
    for cartouche in record['cartouches']:
        interior = ' '.join(cartouche['codes'])
        names.append(
            f"{cartouche['translit']} — {cartouche['king']} "
            f"(interior: {interior})"
        )

    return {
        'codes': codes,
        'confidences': [1.0] * len(codes),
        'boundary_hints': line_ends[:-1],
        'cartouche_names': names,
        'direction': record['direction'],
        'layout': record['layout'],
        'image': image,
    }


def call(
    method: str,
    url: str,
    timeout: int,
    **kwargs: Any,
) -> tuple[int, dict[str, Any], float]:
    """One HTTP call: status (0 = no response), decoded body, seconds."""
    started = time.monotonic()

    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as e:
        return 0, {'detail': f'no response: {e}'}, time.monotonic() - started

    elapsed = time.monotonic() - started
    try:
        body = response.json()
    except ValueError:
        body = {'detail': response.text[:500]}

    if response.status_code != 200:
        body['x_guard'] = response.headers.get('X-Guard')

    return response.status_code, body, elapsed


def run_image(
    base: str,
    record: dict[str, Any],
    path: Path,
    condition: str,
) -> dict[str, Any]:
    """One read of one image in one condition."""
    image = data_url(path)
    result: dict[str, Any] = {
        'file': record['file'],
        'condition': condition,
        'predict': None,
        'reading': None,
        'error': None,
    }

    if condition == 'detected':
        with open(path, 'rb') as f:
            status, predict, seconds = call(
                'POST',
                f'{base}/predict/',
                PREDICT_TIMEOUT_S,
                files={'file': (path.name, f)},
                data={
                    'layout': record['layout'],
                    'direction': record['direction'],
                    'translate': 'false',
                },
            )

        result['predict_s'] = round(seconds, 2)
        if status != 200:
            result['error'] = {'stage': 'predict', 'status': status, **predict}
            return result

        predict.pop('annotated_image', None)
        result['predict'] = predict
        body = detected_request(predict, image)
    else:
        body = oracle_request(record, image)

    result['request'] = {k: v for k, v in body.items() if k != 'image'}
    status, reading, seconds = call(
        'POST', f'{base}/transliterate/', READ_TIMEOUT_S, json=body
    )
    result['read_s'] = round(seconds, 2)

    if status != 200:
        result['error'] = {'stage': 'read', 'status': status, **reading}
    else:
        result['reading'] = reading

    return result


def git_sha() -> str:
    try:
        out = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'

    return out.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run the agent on the golden images and save the reads.'
    )
    parser.add_argument('--base', default='http://localhost:8002')
    parser.add_argument(
        '--golden', type=Path, default=Path('reports/agent_golden_chosen.json')
    )
    parser.add_argument('--images', type=Path, default=Path('chosen'))
    parser.add_argument('--out', type=Path, default=Path('reports/agent_eval'))
    parser.add_argument('--run-id', default=None)
    parser.add_argument(
        '--conditions', nargs='+', choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument('--only', nargs='+', default=None, metavar='FILE')
    args = parser.parse_args()

    base = args.base.rstrip('/')
    records = json.loads(args.golden.read_text(encoding='utf-8'))['images']
    if args.only:
        records = [r for r in records if r['file'] in args.only]

    now = datetime.now(timezone.utc)
    run_dir = args.out / (args.run_id or now.strftime('%Y%m%d_%H%M%S'))
    run_dir.mkdir(parents=True, exist_ok=True)

    spent = 0.0
    for record in records:
        for condition in args.conditions:
            result = run_image(
                base, record, args.images / record['file'], condition
            )
            name = f"{Path(record['file']).stem}.{condition}.json"
            (run_dir / name).write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            cost = (result['reading'] or {}).get('cost_usd') or 0.0
            spent += cost
            state = 'ok' if result['reading'] else f"ERROR {result['error']}"
            print(f"{record['file']:30s} {condition:9s} ${cost:.4f}  {state}")

    manifest = {
        'run_at': now.isoformat(timespec='seconds'),
        'git_sha': git_sha(),
        'base': base,
        'golden': str(args.golden),
        'conditions': args.conditions,
        'images': [r['file'] for r in records],
        'cost_usd': round(spent, 4),
    }
    (run_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8'
    )
    print(f'\nsaved to {run_dir}  total ${spent:.4f}')


if __name__ == '__main__':
    main()
