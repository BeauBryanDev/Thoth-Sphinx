from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from pydantic import ValidationError

from app.core.config import settings
from app.schemas.reading_chat import ReadingLine, ReadingSnapshot
from app.schemas.transliterations import HumanCorrection, TransliterateRequest, TransliterationOut

logger = logging.getLogger('sphinxeyes.reading_sessions')

# Agent state: the reading under discussion lives on the server, keyed by a server-made ID.
# In memory and per process, like the riddle games: lost on restart, so the client keeps a copy.


@dataclass
class ReadingState:
    """One reading and its history of human corrections."""
    snapshot    : ReadingSnapshot
    corrections : list[HumanCorrection] = field(default_factory=list)   # the full list sent with the last re-read
    version     : int = 1                                            # 1 = first read, +1 per re-read
    created     : float = field(default_factory=time.monotonic)
    last_used   : float = field(default_factory=time.monotonic)


class ReadingSessions:
    """In-memory readings keyed by reading_id; idle ones expire, the oldest go beyond the cap."""

    def __init__(self, ttl_s: int, max_sessions: int) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._states: OrderedDict[str, ReadingState] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, reading_id: str | None) -> ReadingState | None:
        """The live state for this ID, touched; None when unknown or expired."""
        if not reading_id:
            return None
        with self._lock:
            self._expire(time.monotonic())
            state = self._states.get(reading_id)
            if state is not None:
                state.last_used = time.monotonic()
                self._states.move_to_end(reading_id)
            return state

    def save(
        self,
        reading_id: str | None,
        snapshot: ReadingSnapshot,
        corrections: list[HumanCorrection],
    ) -> tuple[str, int]:
        """A re-read of a live reading keeps its ID and bumps the version; anything else starts a new one."""
        with self._lock:
            self._expire(time.monotonic())
            previous = self._states.get(reading_id) if reading_id else None
            if previous is not None and corrections:
                key, version = reading_id, previous.version + 1
            else:
                key, version = uuid.uuid4().hex, 1
            self._states[key] = ReadingState(snapshot=snapshot, 
                                            corrections=list(corrections), 
                                            version=version)
            self._states.move_to_end(key)
            while len(self._states) > self.max_sessions:
                self._states.popitem(last=False)
            return key, version

    def _expire(self, now: float) -> None:
        """Drop readings idle longer than the TTL, oldest first."""
        while self._states:
            key, state = next(iter(self._states.items()))
            if now - state.last_used <= self.ttl_s:
                break
            del self._states[key]


def snapshot_from(out: TransliterationOut, body: TransliterateRequest) -> ReadingSnapshot | None:
    """The reading as the chat sees it, built on the server from its own output; None if it breaks a cap."""
    router_step = next((e for e in out.trace if e.kind == 'router'), None)
    try:
        return ReadingSnapshot(
            lines=[
                ReadingLine(
                    line=c.chunk_index + 1,
                    codes=c.gardiner_codes,
                    transliteration=c.transliteration,
                    gloss=c.english_gloss,
                    confidence=c.confidence,
                )
                for c in out.chunks
            ],
            full_transliteration=out.full_transliteration,
            full_translation=out.full_translation,
            cartouches=body.cartouche_names,
            evidence=out.evidence,
            image_notes=out.image_notes,
            model=out.model,
            escalated=out.escalated,
            escalation_reasons=out.escalation_reasons,
            second_opinion_reason=(router_step.detail.get('haiku_reason') or '') if router_step else '',
            direction=body.direction,
            layout=body.layout,
            context=body.context,
        )
    except ValidationError as e:  # an unusually long reading: the chat falls back to the client's copy
        logger.warning('reading not kept on the server: %s', e.errors()[:2])
        return None


READING_SESSIONS = ReadingSessions(
    ttl_s=settings.reading_session_ttl_s,
    max_sessions=settings.reading_max_sessions,
)
