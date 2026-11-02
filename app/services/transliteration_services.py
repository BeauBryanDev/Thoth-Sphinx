
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from pydantic import BaseModel

from app import llm
from app.core.config import settings
from app.llm import images
from app.schemas.transliterations import (
    ChunkTransliteration,
    TextContext,
    TransliterationOut,
)

logger = logging.getLogger('sphinxeyes.transliteration')

# LLM transliteration + translation stage (Phase 6).

def segment_into_chunks(
    codes           : list[str],
    confidences     : list[float],
    boundary_hints  : list[int],
    max_signs       : int = 20,
) -> list[list[tuple[str, float]]]:
    """
    Split the ordered (code, conf) sequence into chunks of at most
    `max_signs`, cutting preferentially at line boundaries so a chunk
    never mixes signs from different physical lines.
    """
    pairs = list(zip(codes, confidences))
    bounds = sorted(b for b in boundary_hints if 0 < b <= len(pairs))
    
    if not bounds or bounds[-1] != len(pairs):
        bounds.append(len(pairs))

    chunks: list[list[tuple[str, float]]] = []
    
    start = 0
    
    for b in bounds:
        
        line = pairs[start:b]
        # split an over-long line at max_signs
        for i in range(0, len(line), max_signs):
            
            piece = line[i:i + max_signs]
            
            if piece:
                chunks.append(piece)
                
        start = b
        
    return chunks


# The role/methodology half of the prompt. Static -> sent as the system
# prompt, which app/llm/ marks for prompt caching across the per-chunk calls.
# Differs from space/src/llm.py (gpt-4o) only in the detector-noise lines of INPUT PROVENANCE.
SYSTEM_PROMPT = """You are a working Egyptologist producing a publishable reading of a Middle \
Egyptian inscription from Gardiner sign codes. Deliver the best philological reading the evidence \
supports — do not catalogue everything that might be wrong with the input.

INPUT PROVENANCE
The codes come from a computer-vision pipeline (YOLO detector -> spatial reading-order ->
lexicon/Viterbi corrector -> royal-name matcher). The sequence is therefore a NOISY COPY —
the noise comes from the detector, not from the stone — and you have full licence to emend it:
- Never describe the monument's condition (worn, damaged, degraded, broken): the photographs
  are usually clean. Attribute any uncertainty to the detection.
- EMEND FREELY. Substitute a confusable sign, restore an omitted phonetic complement or
  determinative, or fix a local transposition whenever it yields coherent Middle Egyptian.
  Do it silently in the obvious cases; report only emendations that change the meaning.
- Frequent confusions of this detector — treat as free substitutions, no justification needed:
  G1<->G5<->G39 | N5<->N33<->Aa1<->O49 | X1<->X8 | S29<->O34 | W19<->W14 | M23<->M22 |
  D21<->D4 | Y1<->Y2 | Z1<->Z4. Small complements and determinatives are the usual detector
  misses — supply them where the word requires it.
- 'Unknown' is an undetected sign, not a blank. Read through it: if the surrounding formula makes
  the word certain, restore it in ( ); write [...] only when it is genuinely unrecoverable.
- Honorific transposition (nTr, nsw, ra written first) is normal orthography — restore the
  phonetic order silently. Reading order within a quadrat may also be locally scrambled.
- A cartouche matched against the royal-name lexicon outranks every raw sign code around it.

METHOD
1. Segment into words: uniliterals, biliterals, triliterals, phonetic complements (never
   transliterate a complement twice), determinatives (classify, never pronounce), logograms.
2. Anchor on the formulae of monumental texts — Htp-dj-nsw, sA ra, nb tAwy, nTr nfr, dj anx,
   mAa-xrw, anx wDA snb, nsw-bjtj, Dt / nHH, divine epithets, royal titulary. A sequence that
   NEARLY matches a formula IS that formula; emend to it.
3. Exploit the archaeological context and any adjacent cartouche — they fix period, orthography
   and genre — and keep the reading consistent with the segments already transliterated.
4. Commit to ONE reading, written the way an Egyptologist writes a translation: connected
   English, not a chain of glosses. Name a genuine alternative in one clause and move on.

Never refuse, never return empty fields, never answer that the sequence is too corrupt to read.
If a segment is poor, give the most plausible reading you can and set confidence LOW.

OUTPUT — JSON only, no preamble, exactly these keys:
{
  "transliteration": "Leiden conventions (aA, j, D, x, X, S, q; '.' before suffix pronouns, '=' for clitics, [...] lacuna, ( ) restored). Word-divided; never emit Gardiner codes here.",
  "english_gloss": "The translation, in vivid, readable English — as many sentences as the segment needs. Your reader is an intelligent non-specialist who knows nothing about ancient Egypt, so render the sense of the line the way a good museum label or a literary translation does: idiomatic, concrete, never word-for-word, never a chain of glosses. Untranslated Egyptian words, Gardiner codes and bracket apparatus do NOT belong here. Where the text is formulaic, say what the formula MEANS rather than naming it.",
  "linguistic_notes": "Up to 4 sentences of plain-English commentary for that same non-specialist: what the signs are doing (sound signs, silent picture-signs that classify a word, name-rings), any emendation that changed the meaning, any real alternative reading. Explain a technical term the first time you use it, or drop it. Be interesting — this is the part that makes the reading legible to someone outside the field.",
  "confidence": "HIGH|MEDIUM|LOW",
  "period_note": "one short remark tying the reading to the stated period/reign, or '' if context was unknown"
}
Answer in ENGLISH only."""


def _signs(codes: list[str], confidences: list[float]) -> str:
    """Codes with their confidence band, e.g. G17(HIGH:0.92) — N35(MED:0.61)."""
    return ' — '.join(
        f'{c}({"HIGH" if s >= 0.80 else "MED" if s >= 0.50 else "LOW"}:{s:.2f})'
        for c, s in zip(codes, confidences)
    )


def _h(v: str) -> str:  # human-readable context value
    # blank/whitespace (client sent '' after the user erased a field)
    # must read as 'unknown', not an empty line the LLM could misread
    return (v or '').strip().replace('_', ' ') or 'unknown'


def _cartouche_block(cartouche_names: list[str]) -> str:
    """Matched names are authoritative; UNRESOLVED entries carry raw interior signs only."""
    # Keep the '[UNRESOLVED' prefix in sync with glyphApi.ts, which builds these entries.
    matched    = [n for n in cartouche_names if not n.startswith('[UNRESOLVED')]
    unresolved = [n for n in cartouche_names if n.startswith('[UNRESOLVED')]
    parts = []
    if matched:
        parts.append(
            'Royal cartouches identified in this scene (lexicon-verified — '
            'treat as authoritative, more reliable than raw sign codes):\n  '
            + '\n  '.join(matched))
    if unresolved:
        parts.append(
            'Cartouches detected but NOT resolved to a known royal name — '
            'read their raw interior signs yourself (a royal name or epithet '
            'is likely; do not invent a specific king):\n  '
            + '\n  '.join(unresolved))
    return '\n- '.join(parts) if parts else 'Contains cartouche: no'


def _context_block(ctx: TextContext, direction: str, layout: str, cartouche_names: list[str]) -> str:
    """The ARCHAEOLOGICAL CONTEXT section shared by both prompt builders."""
    return f"""ARCHAEOLOGICAL CONTEXT (fields marked 'unknown' were not supplied by the user — do not invent them, but exploit every field that IS given):
- Period: {_h(ctx.period)}
- Dynasty: {_h(ctx.dynasty)}
- King's reign: {_h(ctx.kings_reign)}
- Text type: {_h(ctx.text_type)}
- Physical support: {_h(ctx.support)}
- Location type: {_h(ctx.location_type)}
- Site: {_h(ctx.site)}
- Reading direction: {direction}
- Layout: {layout}
- {_cartouche_block(cartouche_names)}"""


def _photo_rules() -> str:
    """How to weigh the attached photograph against the detector."""
    return ('Check every detected sign against the stone. The detector knows only 148 Gardiner '
            'signs and misreads the others as a look-alike (e.g. N28 xa comes out as X1): where '
            'the photograph clearly shows a different or a missed sign, read the photograph and '
            'say so in linguistic_notes. The photograph outranks the detector; if it contradicts '
            'a cartouche match, say that too.')


def build_egyptologist_prompt(
    codes            : list[str],
    confidences      : list[float],
    ctx              : TextContext,
    direction        : str,
    layout           : str,
    cartouche_names  : list[str],
    previous_context : str | None,
    chunk_info       : str,
    full_sequence    : str = '',
    has_image        : bool = False,
) -> str:
    """Build the per-chunk USER message (the system message is static)."""
    signs = _signs(codes, confidences)
    # The whole inscription, so a segment is never read blind. Without this the
    # LLM sees ~20 codes with no idea what surrounds them and hedges.
    whole_block = (
        f'\nTHE FULL INSCRIPTION (all segments, reading order, for orientation only '
        f'— transliterate ONLY the segment above):\n{full_sequence}\n'
        if full_sequence else ''
    )
    previous_block = (
        f'\nPREVIOUS SEGMENTS of the same inscription (already transliterated '
        f'— keep names, epithets and topic consistent with them):\n'
        f'{previous_context}\n' if previous_context else ''
    )
    # The raw photograph rides along as an image block; this tells the model how to use it.
    photo_block = (
        '\nTHE PHOTOGRAPH (attached, unprocessed, downscaled): find this segment on it — segments '
        f'follow the physical {layout} in reading order, read {direction}. {_photo_rules()}\n'
        if has_image else ''
    )

    return f"""DETECTED SIGN SEQUENCE ({chunk_info}; each segment is one physical line of text; Gardiner codes with detector confidence):
{signs}

{_context_block(ctx, direction, layout, cartouche_names)}
{whole_block}{previous_block}{photo_block}
Transliterate and gloss this segment. JSON only."""


def build_wall_prompt(
    lines           : list[list[tuple[str, float]]],
    ctx             : TextContext,
    direction       : str,
    layout          : str,
    cartouche_names : list[str],
    has_image       : bool = False,
) -> str:
    """Build the USER message that reads the whole inscription in one call."""
    numbered = '\n'.join(
        f'  [{n + 1}] ' + _signs([c for c, _ in line], [s for _, s in line])
        for n, line in enumerate(lines)
    )
    photo_block = (
        '\nTHE PHOTOGRAPH (attached, unprocessed, downscaled): the lines above follow the '
        f'physical {layout} of the photograph in reading order, read {direction}; match each '
        f'numbered line to its {layout[:-1]} on the stone. Cartouche interiors are listed under '
        'the context, not inside the lines, so each cartouche shows on the photograph between '
        f'the signs around it. {_photo_rules()}\n'
        if has_image else ''
    )
    return f"""DETECTED SIGN SEQUENCE (the WHOLE inscription, {len(lines)} physical lines in reading order; Gardiner codes with detector confidence):
{numbered}

{_context_block(ctx, direction, layout, cartouche_names)}
{photo_block}
WHOLE-INSCRIPTION READING: read all {len(lines)} lines together as one text — a sentence or a royal titulary may run on from one line into the next. The response has one `lines` entry per numbered line, in order, each with the OUTPUT fields for that line alone, plus `full_translation`: the whole inscription as one connected English translation (the reading a museum label would print), and one `period_note` for the whole text."""


MAX_REPLY_TOKENS = 16000   # adaptive thinking counts toward it


class _ChunkReading(BaseModel):
    """The reading of one segment."""  # sent as the schema description; fields mirror SYSTEM_PROMPT's OUTPUT
    transliteration  : str
    english_gloss    : str
    linguistic_notes : str
    confidence       : Literal['HIGH', 'MEDIUM', 'LOW']
    period_note      : str


class _LineReading(BaseModel):
    """The reading of one numbered line of the inscription."""
    line             : int
    transliteration  : str
    english_gloss    : str
    linguistic_notes : str
    confidence       : Literal['HIGH', 'MEDIUM', 'LOW']


class _WallReading(BaseModel):
    """The reading of the whole inscription: one entry per line, then the connected translation."""
    lines            : list[_LineReading]
    full_translation : str
    period_note      : str


WHOLE_LINE = 10**6   # segment_into_chunks cap that keeps every physical line whole


# Service (port of segement_into_chunck.transliterate_wall)
class TransliterationService:
    """Chunk the corrected sequence, call the LLM per chunk, assemble."""

    @property
    def enabled(self) -> bool:
        return llm.enabled()

    def transliterate(
        self,
        raw       : dict,          # SphinxPipeline.run() output
        ctx       : TextContext,
        image_bgr : np.ndarray | None = None,   # the raw upload, never the enhanced image
    ) -> TransliterationOut:
        outer = raw['outer']
        codes = outer['correction']['flat_corrected_seq']
        confs = [slot[0][1] if slot else 0.0 for slot in outer['slots']]
        # corrected seq and slots are index-aligned; guard anyway
        if len(confs) != len(codes):
            
            confs = (confs + [0.0] * len(codes))[:len(codes)]

        cartouche_names = []
        for c in raw['cartouches']:
            if c.get('translit'):
                cartouche_names.append(
                    f"{c['translit']} — {c['english']} (interior: {' '.join(c['spelling'] or [])})"
                )
            elif c.get('n_members', 0) > 0:
                raw_codes = ' '.join(
                    slot[0][0] for slot in c.get('slots', []) if slot
                )
                if raw_codes:
                    cartouche_names.append(
                        f"[UNRESOLVED cartouche — raw signs detected but no confident "
                        f"royal-name match: {raw_codes}]"
                    )
                    
            else :
                cartouche_names.append(f"[UNRESOLVED cartouche — no signs detected]")
        
        
        return self.transliterate_sequence(
            codes, confs, outer['boundary_hints'], cartouche_names,
            direction=raw['direction'], layout=raw['layout'], ctx=ctx,
            image_bgr=image_bgr,
        )
        

    def transliterate_sequence(
        self,
        codes           : list[str],
        confidences     : list[float],
        boundary_hints  : list[int],
        cartouche_names : list[str],
        *,
        direction       : str,
        layout          : str,
        ctx             : TextContext,
        image_bgr       : np.ndarray | None = None,
    ) -> TransliterationOut:

        if not self.enabled:
            
            return TransliterationOut(
                
                chunks=[], full_transliteration='', full_translation='',
                model=settings.translit_model, n_chunks=0,
                error='ANTHROPIC_API_KEY not configured — transliteration disabled.',
            )

        # One image block for every call; cache_control keeps system + image a cached prefix.
        photo = images.image_block(image_bgr, cache=True) if image_bgr is not None else None

        # Short inscriptions are read in one call; long ones, or a truncated reply, go per line.
        if len(codes) <= settings.translit_whole_max_signs:
            try:
                return self._read_wall(codes, confidences, boundary_hints, cartouche_names,
                                       direction=direction, layout=layout, ctx=ctx, photo=photo)
            except llm.LLMTruncated:
                logger.warning('whole-inscription reply hit max_tokens; reading per line instead')
            except Exception as e:
                logger.exception('LLM whole-inscription reading failed')
                return TransliterationOut(
                    chunks=[], full_transliteration='', full_translation='',
                    model=settings.translit_model, n_chunks=0,
                    error=f'LLM call failed on the whole inscription: {e}',
                )

        chunks = segment_into_chunks(
            codes, confidences, boundary_hints,
            max_signs=settings.translit_max_signs,
        )

        # The whole inscription, segment by segment, so every per-chunk call
        # can see what surrounds the ~20 codes it is asked to read.
        full_sequence = '\n'.join(
            f'  [{n + 1}] ' + ' '.join(c for c, _ in ch)
            for n, ch in enumerate(chunks)
        )

        results: list[ChunkTransliteration] = []
        served_by = settings.translit_model
        for i, chunk in enumerate(chunks):
            c_codes = [c for c, _ in chunk]
            c_confs = [s for _, s in chunk]
            previous = ' | '.join(r.transliteration for r in results) or None

            prompt = build_egyptologist_prompt(
                c_codes, c_confs, ctx,
                direction=direction, layout=layout,
                cartouche_names=cartouche_names,
                previous_context=previous,
                chunk_info=f'Segment {i + 1} of {len(chunks)}',
                full_sequence=full_sequence,
                has_image=photo is not None,
            )
            content: str | list[dict] = (
                [photo, {'type': 'text', 'text': prompt}] if photo is not None else prompt)
            try:
                parsed, usage = llm.complete_parsed(
                    feature    = 'translit',
                    model      = settings.translit_model,
                    system     = SYSTEM_PROMPT,
                    messages   = [{'role': 'user', 'content': content}],
                    schema     = _ChunkReading,
                    effort     = settings.translit_effort,
                    max_tokens = MAX_REPLY_TOKENS,
                )
                served_by = usage.served_by

            except Exception as e:
                logger.exception(f'LLM transliteration failed on chunk {i}')

                return TransliterationOut(
                    chunks=results,
                    full_transliteration=' | '.join(r.transliteration for r in results),
                    full_translation=' '.join(r.english_gloss for r in results),
                    model=served_by, n_chunks=len(chunks),
                    error=f'LLM call failed on segment {i + 1}/{len(chunks)}: {e}',
                )

            results.append(ChunkTransliteration(
                chunk_index      = i,
                gardiner_codes   = c_codes,
                transliteration  = parsed.transliteration,
                english_gloss    = parsed.english_gloss,
                linguistic_notes = parsed.linguistic_notes,
                confidence       = parsed.confidence,
                period_note      = parsed.period_note,
            ))

        return TransliterationOut(
            chunks               = results,
            full_transliteration = ' | '.join(r.transliteration for r in results),
            full_translation     = ' '.join(r.english_gloss for r in results),
            model                = served_by,
            n_chunks             = len(results),
        )

    def _read_wall(
        self,
        codes           : list[str],
        confidences     : list[float],
        boundary_hints  : list[int],
        cartouche_names : list[str],
        *,
        direction       : str,
        layout          : str,
        ctx             : TextContext,
        photo           : dict | None,
    ) -> TransliterationOut:
        """Read every physical line in one call; raises LLMError on failure."""
        lines = segment_into_chunks(codes, confidences, boundary_hints, max_signs=WHOLE_LINE)
        prompt = build_wall_prompt(lines, ctx, direction, layout, cartouche_names,
                                   has_image=photo is not None)
        content: str | list[dict] = (
            [photo, {'type': 'text', 'text': prompt}] if photo is not None else prompt)
        parsed, usage = llm.complete_parsed(
            feature    = 'translit_wall',
            model      = settings.translit_model,
            system     = SYSTEM_PROMPT,
            messages   = [{'role': 'user', 'content': content}],
            schema     = _WallReading,
            effort     = settings.translit_effort,
            max_tokens = MAX_REPLY_TOKENS,
        )

        # Match replies to lines by number; fall back to order if the numbering is off.
        by_line = {r.line: r for r in parsed.lines}
        if set(by_line) != set(range(1, len(lines) + 1)):
            by_line = dict(enumerate(parsed.lines, start=1))

        chunks: list[ChunkTransliteration] = []
        for i, line in enumerate(lines):
            r = by_line.get(i + 1)
            chunks.append(ChunkTransliteration(
                chunk_index      = i,
                gardiner_codes   = [c for c, _ in line],
                transliteration  = r.transliteration if r else '',
                english_gloss    = r.english_gloss if r else '',
                linguistic_notes = r.linguistic_notes if r else 'No reading was returned for this line.',
                confidence       = r.confidence if r else 'LOW',
                period_note      = parsed.period_note if i == 0 else '',
            ))

        return TransliterationOut(
            chunks               = chunks,
            full_transliteration = ' | '.join(c.transliteration for c in chunks),
            full_translation     = parsed.full_translation or ' '.join(c.english_gloss for c in chunks),
            model                = usage.served_by,
            n_chunks             = len(chunks),
        )
