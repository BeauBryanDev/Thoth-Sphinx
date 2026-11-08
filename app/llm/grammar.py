import json
from functools import cache
from pathlib import Path
from typing import Any

from app.llm.prompts import load

VISION_DIR = Path(__file__).resolve().parents[2] / 'vision'


def _read_json(name: str) -> dict[str, Any]:
    """A prior file from vision/, or {} when it is missing."""
    path = VISION_DIR / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def particles_block(particles: dict[str, dict[str, Any]]) -> str:
    """One line per particle: transliteration, signs, meaning, register."""
    lines = []
    for word, entry in particles.items():
        signs = ' '.join(entry['gardiner'])
        register = entry.get('register', 'all')
        lines.append(f'- {word.replace("_", " ")} ({signs}): {entry["meaning"]}; register: {register}')
    return '\n'.join(lines) or '(none)'


def determinatives_block(codes: dict[str, dict[str, Any]]) -> str:
    """One line per category, the commonest determinatives first."""
    groups: dict[str, list[tuple[int, str]]] = {}
    for code, entry in codes.items():
        category = entry.get('category') or 'other'
        groups.setdefault(category, []).append((entry.get('last_count', 0), code))
    ordered = sorted(
        groups.items(),
        key=lambda item: -len(item[1]),
    )
    lines = []
    for category, members in ordered:
        signs = ' '.join(code for _, code in sorted(members, reverse=True))
        lines.append(f'- {category.replace("_", " ")}: {signs}')
    return '\n'.join(lines) or '(none)'


@cache
def grammar_skill() -> str:
    """me_grammar.md with the particle and determinative lists filled in from vision/."""
    particles = _read_json('initial_particles.json').get('particles', {})
    codes = _read_json('determinatives.json').get('codes', {})
    return (
        load('me_grammar')
        .replace('<<PARTICLES>>', particles_block(particles))
        .replace('<<DETERMINATIVES>>', determinatives_block(codes))
    )
