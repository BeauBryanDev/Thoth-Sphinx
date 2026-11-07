// ============================================================
// Reading session in localStorage — the client half of agent state
// ============================================================
// The server keeps the reading by its reading_id (app/services/reading_sessions.py),
// in memory, so a restart or a 60-minute idle loses it. The browser keeps the
// latest session id and its snapshot here: the chat sends only the id, and the
// snapshot is the fallback when the server answers 410. Storage can be blocked
// (private windows, cleared data), so every access is guarded.

import type { ReadingSnapshot } from "@/api/readingChatApi";

const KEY = "sphinx.readingSession";

export interface StoredReadingSession {
  sessionId: string;
  version: number | null;
  snapshot: ReadingSnapshot;
  savedAt: number;
}

export const saveReadingSession = (session: StoredReadingSession): void => {
  try {
    localStorage.setItem(KEY, JSON.stringify(session));
  } catch {
    // Storage full or blocked: the in-memory result still works.
  }
};

export const loadReadingSession = (sessionId: string | null): StoredReadingSession | null => {
  if (!sessionId) return null;
  try {
    const raw = localStorage.getItem(KEY);
    const stored = raw ? (JSON.parse(raw) as StoredReadingSession) : null;
    return stored?.sessionId === sessionId ? stored : null;
  } catch {
    return null;
  }
};
