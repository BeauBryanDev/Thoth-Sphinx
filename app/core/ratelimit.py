import hashlib
import threading
import time

# Cost control, not security: state is in memory (per process, reset on restart), and identity
# is the client IP, which is shared behind NAT and can be rotated. Rate limits and guardrails
# for the whole agent come later; this only stops repeated paid re-reads from one client.


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
