import logging
import re
import time
from typing import Any

from app.llm import tools
from app.schemas.transliterations import EntityCard, ToolCallOut

logger = logging.getLogger('sphinxeyes.entity_cards')

# Pre-fetch: a matched cartouche already names its king, so code looks him up before the model
# starts. That saves the model a tool round trip, and the UI shows the same card.


def cartouche_label(entry: str) -> str | None:
    """'wnjs — Unas (Dynasty V) (interior: E34 …)' -> 'Unas (Dynasty V)'; None if unresolved."""
    if entry.startswith('[UNRESOLVED') or ' — ' not in entry:
        return None
    label = entry.split(' — ', 1)[1].split(' (interior:', 1)[0].strip()
    return label or None


def king_card(label: str) -> EntityCard | None:
    """Follow the graph from the royal name to its pharaoh; None if either is missing."""
    royal = tools.lookup_entity(
        label,
        entity_type='royal_name',
        limit=1,
        examples=0,
    )['entities']
    link = next((ln for ln in (royal[0]['links'] if royal else []) if ln['relation'] == 'names'), None)
    if link is None:
        return None
    kings = tools.lookup_entity(
        link['name'],
        entity_type='pharaoh',
        limit=3,
        examples=0,
    )['entities']
    king = next((k for k in kings if k['id'] == link['id']), None)
    if king is None:
        return None

    attributes: dict[str, Any] = king.get('attributes') or {}
    links: list[dict[str, Any]] = king.get('links') or []
    return EntityCard(
        id=king['id'],
        name=king['name'],
        cartouche=label,
        dynasty=next((ln['name'] for ln in links if ln['relation'] == 'ruled_in'), None),
        period=attributes.get('period'),
        reign=attributes.get('reign_length') or attributes.get('reign_period'),
        notes=_first_sentences(attributes.get('notes') or '', 2),
        monuments=[
            ln['name'] for ln in links
            if ln['relation'] == 'made_by' and ln['direction'] == 'incoming'
        ][:5],
        wikipedia_url=attributes.get('wikipedia_url'),
        source=king.get('source') or '',
        reliability=king.get('reliability') or '',
    )


def prefetch(cartouche_names: list[str]) -> tuple[list[EntityCard], list[ToolCallOut]]:
    """One card per matched cartouche, and each lookup as a tool call made by the pipeline."""
    cards: list[EntityCard] = []
    calls: list[ToolCallOut] = []
    labels = dict.fromkeys(filter(None, map(cartouche_label, cartouche_names)))
    for label in labels:
        started = time.monotonic()
        try:
            card = king_card(label)
            summary = f'{card.name} ({card.dynasty or "dynasty unknown"})' if card else 'no king in the graph'
            ok = True
        except Exception as e:  # graph down: the reading still ships, without the card
            logger.warning('entity card failed for %s: %s', label, e)
            card, summary, ok = None, f'graph unavailable ({e})', False
        if card:
            cards.append(card)
        calls.append(ToolCallOut(
            model='pipeline',
            tool='lookup_entity',
            input={'name': label},
            summary=summary,
            ok=ok,
            latency_ms=int((time.monotonic() - started) * 1000),
        ))
    return cards, calls


def prompt_block(cards: list[EntityCard]) -> str:
    """The cards as a short prompt section; empty when there are none."""
    if not cards:
        return ''
    rows = []
    for c in cards:
        facts = [c.dynasty, c.period, f'reign {c.reign}' if c.reign else None]
        row = f'  - {c.name}: ' + ', '.join(f for f in facts if f)
        if c.monuments:
            row += f'; monuments: {", ".join(c.monuments)}'
        rows.append(row)
    return ('\n- KNOWLEDGE GRAPH, the kings of the matched cartouches (pre-fetched reference '
            'background, not evidence; the stone outranks it):\n' + '\n'.join(rows))


def _first_sentences(text: str, n: int) -> str:
    """The first n sentences of a note."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return ' '.join(parts[:n])
