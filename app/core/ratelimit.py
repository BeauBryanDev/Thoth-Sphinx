import hashlib
import threading
import time
from collections import deque

# Cost control, not security: state is in memory (per process, reset on restart), and identity
# is the client IP, which is shared behind NAT and can be rotated. app/core/guards.py applies
# these to the HTTP routes; app/llm/client.py feeds SPEND from every call's Usage.

DAY_S: int = 86_400
MAX_KEYS: int = 10_000   # keep memory bounded under many distinct clients


def client_key(ip: str | None) -> str:
    """A stable, non-reversible key for a client IP; raw IPs are never stored."""
    return hashlib.sha256(f'sphinx-reread:{ip or "unknown"}'.encode()).hexdigest()[:16]


class Cooldown:
    """One action per key per window. reserve() takes the slot; release() gives it back."""

    def __init__(self, window_s: float) -> None:
        self.window_s = window_s
        self._last: dict[str, float] = {}
        self._lock = threading.Lock()

    def reserve(self, key: str, 
    now: float | None = None
    ) -> int | None:
        """Take the slot and return None, or return the seconds left until it frees up."""
        if self.window_s <= 0:
            return None

        now = time.monotonic() if now is None else now

        with self._lock:
            last = self._last.get(key)
            if last is not None and now - last < self.window_s:
                return max(1, int(self.window_s - (now - last)) + 1)

            self._last[key] = now
            # Keep memory bounded: drop keys whose window has passed.
            if len(self._last) > 10_000:
                self._last = {k: t for k, t in self._last.items() if now - t < self.window_s}

            return None

    def release(self, key: str) -> None:
        """Give the slot back, e.g. when the paid call failed and should not count."""
        with self._lock:
            self._last.pop(key, None)


def seconds_to_utc_midnight(now: float | None = None) -> int:
    """Seconds until the next UTC day starts, at least 1."""
    now = time.time() if now is None else now

    return max(1, DAY_S - int(now) % DAY_S)


class SlidingWindow:
    """At most `limit` actions per key in any `window_s` seconds; every attempt counts."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def reserve(self, key: str, 
                limit: int, 
                window_s: float,
                now: float | None = None
                ) -> int | None:
        """Record the action and return None, or return the seconds until one is allowed."""
        if limit <= 0 or window_s <= 0:
            return None

        now = time.monotonic() if now is None else now

        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and now - hits[0] >= window_s:
                hits.popleft()
            if len(hits) >= limit:
                return max(1, int(window_s - (now - hits[0])) + 1)
            hits.append(now)
            if len(self._hits) > MAX_KEYS:
                self._hits = {k: h for k, h in self._hits.items() if h and now - h[-1] < window_s}
            return None


class DailyQuota:
    """At most `limit` actions per key per UTC day; release() refunds a failed one."""

    def __init__(self) -> None:
        self._day = -1
        self._counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def reserve(self, key: str, 
                limit: int, 
                now: float | None = None
                ) -> int | None:
        """Count the action and return None, or return the seconds until the quota resets."""
        if limit <= 0:
            return None
        now = time.time() if now is None else now
        with self._lock:
            self._roll(now)
            if self._counts.get(key, 0) >= limit:
                return seconds_to_utc_midnight(now)
            self._counts[key] = self._counts.get(key, 0) + 1
            return None

    def release(self, key: str) -> None:
        """Refund one action, e.g. when the paid call failed."""
        with self._lock:
            n = self._counts.get(key, 0)
            if n > 0:
                self._counts[key] = n - 1

    def _roll(self, now: float) -> None:
        """Start a fresh day at UTC midnight."""
        day = int(now // DAY_S)
        if day != self._day:
            self._day, self._counts = day, {}


class SpendCap:
    """USD spent on the LLM today (UTC), across all clients and features."""

    def __init__(self) -> None:
        self._day = -1
        self._spent = 0.0
        self._lock = threading.Lock()

    def add(self, usd: float, now: float | None = None) -> None:
        """Record the cost of one call."""
        with self._lock:
            self._roll(time.time() if now is None else now)
            self._spent += max(0.0, usd)

    def spent(self, now: float | None = None) -> float:
        """USD spent so far today."""
        with self._lock:
            self._roll(time.time() if now is None else now)
            return self._spent

    def exhausted(self, cap_usd: float,
                now: float | None = None
                ) -> bool:
        """True once today's spend reaches the cap; a cap of 0 disables it."""
        return cap_usd > 0 and self.spent(now) >= cap_usd

    def _roll(self, now: float) -> None:
        """Start a fresh day at UTC midnight."""
        day = int(now // DAY_S)
        if day != self._day:
            self._day, self._spent = day, 0.0


# Process-wide state; tests replace these with fresh instances 
REQUESTS = SlidingWindow()  # every LLM route, per client
TRANSLIT_DAILY = DailyQuota()   # transliteration runs, per client
SPEND = SpendCap()   # the whole agent, all clients
