import json
from functools import cache

import anthropic
from anthropic.types import JSONOutputFormatParam, MessageParam, OutputConfigParam
from dotenv import dotenv_values

from corpus.paths import ENV_FILE

MODEL = 'claude-haiku-5-5'
MAX_TOKENS = 2000
PRICE_PER_MILLION_INPUT = 0.10
PRICE_PER_MILLION_OUTPUT = 0.50
SYSTEM = ('You read Ancient Egyptian transliteration. The reading may contain errors or missing '
          'words. Translate the sentence into {language} as best you can, even if unsure, as one '
          'short plain sentence.')
# Without a schema, Haiku often reasons in its visible reply, and those notes would be embedded.
OUTPUT_FORMAT: JSONOutputFormatParam = {
    'type': 'json_schema',
    'schema': {'type': 'object',
               'properties': {'translation': {'type': 'string'}},
               'required': ['translation'],
               'additionalProperties': False}}
OUTPUT_CONFIG: OutputConfigParam = {'effort': 'low', 'format': OUTPUT_FORMAT}


@cache
def client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=dotenv_values(ENV_FILE)['ANTHROPIC_API_KEY'],
                               max_retries=5)


def prompt(transliteration: str, before: str | None = None, after: str | None = None) -> str:
    """Build the request; neighbours help with meaning but must not be translated."""
    if not (before or after):
        return f'Sentence: {transliteration}'
    return (f'Previous sentence, for context only: {before or "(none)"}\n'
            f'Next sentence, for context only: {after or "(none)"}\n'
            f'Translate only this sentence: {transliteration}')


def translate(transliteration: str,
              language: str = 'German',
              before: str | None = None,
              after: str | None = None,
              api: anthropic.Anthropic | None = None
              ) -> tuple[str, int, int]:
    """Return (draft, input tokens, output tokens). A refusal or cut-off gives a blank draft."""
    messages: list[MessageParam] = [
        {'role': 'user',
         'content': prompt(transliteration, before, after)}]
    
    message = (api or client()).messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS, 
        output_config=OUTPUT_CONFIG,
        system=SYSTEM.format(language=language),
        messages=messages)
    
    usage = message.usage.input_tokens, message.usage.output_tokens
    
    if message.stop_reason != 'end_turn':
        return '', *usage
    
    text = next(block.text for block in message.content if block.type == 'text')
    
    return json.loads(text)['translation'].strip(), *usage
