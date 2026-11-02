from collections.abc import Callable
from functools import wraps
from typing import Any

import psycopg
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from evidence_store import tools

INSTRUCTIONS = (
    'Evidence tools for reading Ancient Egyptian texts. Corpus sentences are evidence; entity '
    'profiles are background. Report a "none" verdict as no parallel found, because most '
    'sentences have none, and never cite weak hits as support.')
READ_ONLY = ToolAnnotations(read_only_hint=True, idempotent_hint=True)

server = MCPServer('sphinx-evidence', instructions=INSTRUCTIONS)


def readable_errors[**P, R](tool: Callable[P, R]) -> Callable[P, R]:
    """Let the agent read why a call failed. The SDK hides ordinary exception text."""
    @wraps(tool)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return tool(*args, **kwargs)
        except ValueError as error:
            raise ToolError(str(error)) from error
        except psycopg.OperationalError as error:
            raise ToolError('The evidence database is not available.') from error
    return wrapped


@server.tool(annotations=READ_ONLY)
@readable_errors
def search_parallels(signs: str | None = None, translit: str | None = None,
                     date: str | None = None, include_undated: bool = True, limit: int = 5,
                     use_rerank: bool = False) -> dict[str, Any]:
    """Find attested sentences that resemble a reading, to support or challenge it.

    Use this after you have a draft reading. Give Gardiner signs separated by spaces, such as
    "D21 N35 X1", a transliteration, such as "ḥtp-ḏi̯ nswt", or both. Give a transliteration
    whenever you can, because signs alone cannot be rated.

    The verdict is strong, possible, none, unrated or too_short. Only strong hits support a
    reading. Possible hits are leads, and none means no parallel was found, which is the usual
    result. A query under five words gets too_short, which says nothing either way.
    German translations in hits are the corpus's own, so quote them as evidence.

    date limits results to a period from list_periods, and includes its child periods.
    use_rerank reorders results through a paid service, and often changes little.
    """
    return tools.search_parallels(signs, translit, date, include_undated, limit, use_rerank)


@server.tool(annotations=READ_ONLY)
@readable_errors
def lookup_lemma(text: str, limit: int = 10) -> dict[str, Any]:
    """Look up a word in the Egyptian dictionary by transliteration or English meaning.

    Use this to get the meaning of a word, or to find a word from its English sense, such as
    "nṯr" or "vulture". Lemmas marked obsolete are listed last and should be avoided.
    """
    return tools.lookup_lemma(text, limit)


@server.tool(annotations=READ_ONLY)
@readable_errors
def get_context(sentence_id: str, window: int = 2) -> dict[str, Any]:
    """Show the sentences around a hit, because many sentences are short fragments.

    Use this before relying on a short hit. sentence_id comes from search_parallels or
    lookup_entity, and window is the number of sentences on each side, up to 5.
    """
    return tools.get_context(sentence_id, window)


@server.tool(annotations=READ_ONLY)
@readable_errors
def list_periods() -> dict[str, Any]:
    """List the valid date filters, such as "6. Dyn." or "Altes Reich", with sentence counts.

    Use this before filtering search_parallels by date. Names are German, so don't guess them.
    """
    return tools.list_periods()


@server.tool(annotations=READ_ONLY)
@readable_errors
def lookup_entity(name: str, entity_type: str | None = None, limit: int = 3,
                  examples: int = 3) -> dict[str, Any]:
    """Look up a god, ruler, temple, pyramid, dynasty or royal name, with links and mentions.

    Use this for background on a name you have read, such as "Ramesses II", "Karnak" or
    "Osiris". It follows links such as made_by, ruled_in, parent_of and dedicated_to, and returns
    corpus sentences naming the entity. Royal names also carry their Gardiner sign spellings.

    entity_type is one of god, pharaoh, royal_name, dynasty, temple or pyramid. Every entity and
    link has a reliability rating: curated and unverified content is background, and only corpus
    sentences are evidence. In corpus results, by_find_spot can cover a whole site, while
    inscribed_here means the text is carved in that pyramid.
    """
    return tools.lookup_entity(name, entity_type, limit, examples)


if __name__ == '__main__':
    server.run('stdio')
