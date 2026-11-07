from __future__ import annotations

import base64
import logging
import os
from functools import cache
from typing import Any

from PIL import Image

from app.core import ratelimit
from app.core.config import settings
from app.llm.agent_tools import AgentTool
from app.llm.prompts import load
from app.services import cartouche_carve
from app.services.name_glyphs import Sign, spell_name
from corpus.play_riddles import RiddleGame

logger = logging.getLogger('sphinxeyes.cartouche')

PROMPT_TEMPLATE = load('cartouche_image')   # app/llm/prompts/cartouche_image.md
ONE_COLUMN_MAX = 9   # longer names are grouped into quadrats

ONE_COLUMN_RULE = 'They stand in a single vertical column, one sign per row.'
QUADRAT_RULE = ('They are grouped in quadrats down a single column, as on temple walls: small or flat '
                'signs sit two side by side in one row, tall signs stand alone in their row.')


def build_prompt(name: str, signs: list[Sign]) -> str:
    """The image prompt for one name: its signs listed by look, in reading order."""
    sign_lines = '\n'.join(
        f'{n}. {sign.look}'
        for n, sign in enumerate(signs, start=1)
    )
    return PROMPT_TEMPLATE.format(
        sign_count=len(signs),
        sign_lines=sign_lines,
        layout_rule=ONE_COLUMN_RULE if len(signs) <= ONE_COLUMN_MAX else QUADRAT_RULE,
        name=name,
    )


def openai_client() -> Any:
    """OpenAI client with the embeddings key from .env; imported lazily, only when a gift is made."""
    from dotenv import dotenv_values
    from openai import OpenAI

    from corpus.paths import ENV_FILE

    key = os.environ.get('OPENAI_API_KEY') or dotenv_values(ENV_FILE).get('OPENAI_API_KEY')
    if not key:
        raise RuntimeError('OPENAI_API_KEY not configured')

    return OpenAI(
        api_key=key,
        max_retries=1,
        timeout=settings.cartouche_timeout_s,
    )


def generate(prompt: str) -> str:
    """Generate the cartouche and return it as a WebP data URL."""
    response = openai_client().images.generate(
        model=settings.cartouche_model,
        prompt=prompt,
        size=settings.cartouche_size,
        quality=settings.cartouche_quality,
        output_format='webp',
        output_compression=settings.cartouche_compression,
        n=1,
    )
    data = response.data[0].b64_json
    logger.info('cartouche generated: %.0f KB', len(base64.b64decode(data)) / 1024)
    return f'data:image/webp;base64,{data}'


@cache
def frame() -> Image.Image:
    """The chosen cartouche frame, loaded once."""
    return Image.open(cartouche_carve.FRAME).convert('RGB')


def render(name: str, signs: list[Sign]) -> str:
    """Carve the name on the frame and return it as a WebP data URL; no API call."""
    image = cartouche_carve.carve_name(frame(), signs, name)
    data = cartouche_carve.to_webp(image, settings.cartouche_compression)
    return 'data:image/webp;base64,' + base64.b64encode(data).decode('ascii')


def gift_tool(game: RiddleGame,
              client_key: str,
              gifts: list[str]
              ) -> AgentTool:
    """The cartouche tool of one request: spends the game's pass, puts the image in `gifts`."""

    def run_create_royal_cartouche(name: str) -> dict[str, Any]:
        letters, signs = spell_name(name)
        if not game.consume_pass():
            raise ValueError('no gift has been earned: the seeker must first pass the riddle trial')

        wait = ratelimit.CARTOUCHE_DAILY.reserve(client_key, settings.cartouche_images_per_day)
        if wait is not None:
            game.refund_pass()
            raise ValueError(f'this seeker has received {settings.cartouche_images_per_day} cartouches '
                             'today; the earned gift is kept and can be carved after 00:00 UTC')

        try:
            image = render(letters, signs)
        except Exception as error:
            logger.warning('cartouche carving failed: %s', error)
            ratelimit.CARTOUCHE_DAILY.release(client_key)
            game.refund_pass()
            raise ValueError('the carving failed; the earned gift is kept, try again in a moment') from None

        gifts.append(image)
        return {
            'created': True,
            'name': letters,
            'signs': [{'code': s.code, 'glyph': s.glyph} for s in signs],
        }

    return AgentTool(
        name='create_royal_cartouche',
        description=('Carve the seeker\'s name in hieroglyphs inside a royal cartouche under the protection '
                     'of Horus: the gift for a seeker who passed the riddle trial. Pass the one given name '
                     'the seeker chose, in Latin letters, exactly as they wrote it. The image is shown to the '
                     'seeker beside your reply; you receive only the signs used. Call it once per gift.'),
        schema={'type': 'object', 'properties': {
            'name': {'type': 'string', 
                     'description': 'the seeker\'s chosen name, one word of Latin letters'
                     },
        }, 'required': ['name']},
        run=run_create_royal_cartouche,
    )
