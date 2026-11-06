from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict

from app.core.config import settings
from corpus.play_riddles import RiddleGame


class RiddleSessions:
    """In-memory riddle games keyed by a server-minted ID; per process, lost on restart."""

    def __init__(self,
                 ttl_s: int,
                 max_sessions: int) -> None:
        self.ttl_s = ttl_s
        self.max_sessions = max_sessions
        self._games: OrderedDict[str, tuple[RiddleGame, float]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str | None) -> tuple[str, RiddleGame]:
        """The live game for this ID, or a fresh unstored game under a new ID."""
        now = time.monotonic()
        with self._lock:
            self._expire(now)
            if session_id and session_id in self._games:
                return session_id, self._games[session_id][0]

        return uuid.uuid4().hex, RiddleGame()

    def keep(self, session_id: str, game: RiddleGame) -> bool:
        """Store a game once its trial has begun; returns whether it is stored."""
        if game.trial is None and not game.trial_passed:
            return False
            
        with self._lock:
            self._games[session_id] = (game, time.monotonic())
            self._games.move_to_end(session_id)
            while len(self._games) > self.max_sessions:
                self._games.popitem(last=False)

        return True

    def _expire(self, now: float) -> None:
        """Drop games idle longer than the TTL (oldest first)."""
        while self._games:
            session_id, (_, last_used) = next(iter(self._games.items()))
            if now - last_used <= self.ttl_s:
                break
            del self._games[session_id]


RIDDLE_SESSIONS = RiddleSessions(
    ttl_s=settings.riddle_session_ttl_s,
    max_sessions=settings.riddle_max_sessions,
)
