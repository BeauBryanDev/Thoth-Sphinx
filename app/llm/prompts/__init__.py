from functools import cache
from pathlib import Path

# System prompts live as markdown next to this file; read, never imported, so no import cycles.
PROMPTS_DIR = Path(__file__).parent


@cache
def load(name: str) -> str:
    """The text of prompts/<name>.md, without surrounding blank lines."""
    return (PROMPTS_DIR / f'{name}.md').read_text(encoding='utf-8').strip()
