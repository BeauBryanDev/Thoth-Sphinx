from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.llm import tools
from corpus.play_riddles import RiddleGame


MAX_HIT_TEXT = 160   # characters of a hit's text the model sees; enough to judge, cheap in tokens
MAX_BOOK_TEXT = 1000  # characters of a book passage: book prose needs more than a corpus line


@dataclass(frozen=True)
class AgentTool:
    """A corpus tool as the model sees it, plus the code that runs it."""
    name        : str
    description : str
    schema      : dict[str, Any]
    run         : Callable[..., dict[str, Any]]

    def definition(self) -> dict[str, Any]:
        """The Messages API tool definition."""
        return {'name': self.name, 
                'description': self.description,
                'input_schema': self.schema}


def _clip(text: Any) -> str:
    """Shorten corpus text for the model."""
    text = str(text or '')
    return text if len(text) <= MAX_HIT_TEXT else text[:MAX_HIT_TEXT] + '…'


def compact_hit(hit: dict[str, Any]) -> dict[str, Any]:
    """The fields of a parallel worth tokens: id, rating, text, translation, date, source."""
    return {
        'sentence_id'    : hit.get('sentence_id'),
        'rating'         : hit.get('rating'),
        'matched_by'     : hit.get('matched_by'),
        'transliteration': _clip(hit.get('transliteration')),
        'translation'    : _clip(hit.get('translation_en') or hit.get('translation_de')),
        'translation_lang': 'en' if hit.get('translation_en') else 'de',
        'date'           : hit.get('date'),
        'text_name'      : hit.get('text_name'),
    }


def run_search_parallels(line: int, translit: str = '', 
                         signs: str = ''
                         ) -> dict[str, Any]:
    """search_parallels for one numbered line, trimmed for the model."""
    result = tools.search_parallels(signs=signs or None,
                                    translit=translit or None, 
                                    limit=3)

    return {'line': line,
            'verdict': result['verdict'], 
            'guide': result['guide'],
            'query_translit': result.get('query_translit'),
            'hits': [compact_hit(h) for h in result['hits']]
            }


def run_lookup_lemma(word: str) -> dict[str, Any]:
    """lookup_lemma, top 5 senses."""
    result = tools.lookup_lemma(word, limit=5)
    return {'found': result['found'],
            'lemmas': [{k: lemma.get(k) for k in ('id', 'name', 'pos', 'gloss_en', 'sign_seq')
                        }  for lemma in result['lemmas']]
            }


def run_lookup_entity(name: str) -> dict[str, Any]:
    """lookup_entity, best match without its long link list."""
    result = tools.lookup_entity(name, limit=1, examples=2)
    
    return {'found': result['found'], 
            'note': result['note'],
            'entities': [{ k: e.get(k) for k in ('name', 'type', 'aliases', 'start_year', 'end_year',
                                                'reliability', 'reliability_guide')}
                         | {'corpus_examples': e.get('corpus')} for e in result['entities']]
            }


def run_search_books(query: str) -> dict[str, Any]:
    """search_books, top 5 passages with their text shortened for the model."""
    result = tools.search_books(query)
    for passage in result['passages']:
        text = passage['text']
        passage['text'] = text if len(text) <= MAX_BOOK_TEXT else text[:MAX_BOOK_TEXT] + '…'
    
    return result


SEARCH_PARALLELS = AgentTool(
    name='search_parallels',
    description=('Search the Thesaurus Linguae Aegyptiae corpus (104,285 Egyptian sentences) for '
                 'parallels to one numbered line. Give your transliteration (MdC ASCII such as '
                 'kA-nxt is fine) and the line\'s Gardiner codes. Returns a verdict — strong, '
                 'possible, none, too_short or unrated — and the closest corpus sentences with '
                 'their dates. "none" is normal: most sentences have no parallel. A parallel proves '
                 'the phrase is real Egyptian, not that it is carved here.'),
    schema={'type': 'object', 'properties': {
    'line'    : {'type': 'integer', 'description': 'the numbered line this search is for'},
        'translit': {'type': 'string', 'description': 'your transliteration of that line'},
        'signs'   : {'type': 'string', 'description': "the line's Gardiner codes, space-separated"},
    }, 'required': ['line', 'translit']},
    run=run_search_parallels,
)


LOOKUP_LEMMA = AgentTool(
    name='lookup_lemma',
    description=('Look up a word in the TLA dictionary (38,775 lemmas) by transliteration or '
                 'English gloss. Returns senses with their usual sign spelling, which helps pick '
                 'the right sense from the signs you see.'),
    schema={'type': 'object', 
            'properties': {
                            'word': {'type': 'string', 
                            'description': 'a transliteration such as xa or an English word'
                              },
    }, 'required': ['word']},
    run=run_lookup_lemma,
)


LOOKUP_ENTITY = AgentTool(
    name='lookup_entity',
    description=('Look up a king, god, temple or pyramid in the knowledge graph. Returns dates, '
                 'aliases and a reliability label; only corpus sentences count as evidence.'),
    schema={'type': 'object', 
            'properties': {
            'name': {
                    'type': 'string',
                    'description': 'e.g. Thutmose III, Amasis, Osiris'
                    },
            }, 'required': ['name']},
    run=run_lookup_entity,
)


SEARCH_BOOKS = AgentTool(
    name='search_books',
    description=('Search 17 Egyptology books (Budge, the Pyramid Texts, the Book of the Dead, an '
                 'encyclopedia of ancient Egypt, Egyptian tales, magic, gods, art, history of '
                 'Egyptology) for passages that answer a question. Ask in plain English, one topic per '
                 'search. Returns up to 5 passages with a citation (book › chapter › page). When '
                 'no_book_support is true, the books do not cover the question: say so instead of '
                 'quoting weak passages. Books are secondary sources, some over a century old.'),
    schema={'type': 'object', 
            'properties': {
                            'query': {'type': 'string', 
                                      'description': 'the question in plain English, e.g. who was the wife of Thoth'
                                      },
    }, 'required': ['query']
           },
    run=run_search_books,
)


def riddle_tools(game: RiddleGame) -> tuple[AgentTool, AgentTool]:
    """The two riddle tools of one conversation, closed over its game state."""

    def run_start_riddle_trial() -> dict[str, Any]:
        return game.start_trial()

    def run_submit_riddle_answer(guess: str) -> dict[str, Any]:
        return game.submit(guess)

    start = AgentTool(
        name='start_riddle_trial',
        description=('Begin the Sphinx\'s trial of three riddles for a seeker who asks for a riddle, a '
                     'game or the reward. Returns the first riddle: speak it exactly as returned, then '
                     'wait for the seeker to answer. Never call submit_riddle_answer in the same turn.'),
        schema={'type': 'object', 'properties': {}},
        run=run_start_riddle_trial,
    )
    submit = AgentTool(
        name='submit_riddle_answer',
        description=('Submit the seeker\'s own answer to the current riddle. Pass only what the seeker '
                     'said, translated into English if needed; never guess for them. The tool judges: '
                     'status is correct, wrong, hint or reveal, and when finished is true, passed says '
                     'whether the trial was won. Call it at most once per seeker message.'),
        schema={'type': 'object', 'properties': {
            'guess': {'type': 'string', 'description': 'the seeker\'s answer to the riddle, and nothing else'},
        }, 'required': ['guess']},
        run=run_submit_riddle_answer,
    )
    return start, submit


READING_TOOLS: tuple[AgentTool, ...] = (SEARCH_PARALLELS, LOOKUP_LEMMA, LOOKUP_ENTITY)


def summarize(name: str, result: dict[str, Any]) -> str:
    """One line for the trace: what a tool call found."""
    if name == 'search_parallels':
        best = (result.get('hits') or [{}])[0]
        where = f" — best: {best.get('transliteration', '')[:60]} ({best.get('date') or 'undated'})" if best else ''
        return f"line {result.get('line')}: {result.get('verdict')}{where}"
    
    if name == 'lookup_lemma':
        lemmas = result.get('lemmas') or []
        return f"{result.get('found', 0)} lemmas" + (
            f" — {lemmas[0].get('name')}: {lemmas[0].get('gloss_en')}" if lemmas else '')
    
    if name == 'lookup_entity':
        ents = result.get('entities') or []
        return f"{result.get('found', 0)} entities" + (
            f" — {ents[0].get('name')} ({ents[0].get('type')})" if ents else '')
    
    if name == 'search_books':
        if result.get('no_book_support'):
            return f"{result.get('found', 0)} passages — the books do not seem to cover this"
        
        top = (result.get('passages') or [{}])[0]
        return f"{result.get('found', 0)} passages" + (f" — best: {top.get('citation')}" if top else '')

    if name == 'start_riddle_trial':

       return f"riddle {result.get('riddle_number')} of {result.get('trial_size')} posed"


    if name == 'submit_riddle_answer':
        
        outcome = ' — trial passed' if result.get('passed') else ' — trial failed'
        return f"riddle {result.get('riddle_number')}: {result.get('status')}" + (
            outcome if result.get('finished') else '')

    if name == 'create_royal_cartouche':
        glyphs = ''.join(s.get('glyph', '') for s in result.get('signs') or [])
        return f"cartouche carved for {result.get('name')}: {glyphs}"

    return 'done'
