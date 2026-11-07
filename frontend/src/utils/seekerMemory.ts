// ============================================================
// Seeker memory — what Thot-Sphinx remembers across visits (/chat/ only)
// ============================================================
// A small profile written by code, never by the model: readings decoded,
// kings seen, riddle trials passed, cartouche gifts and the name carved.
// It travels with each /chat/ turn as one short line (~40–60 tokens);
// the server cleans it and fences it as <seeker_memory>. Also holds the
// decode history (last 10 readings) shown on the profile page.

import { readJson, writeJson } from "@/utils/storage";
import type { GlyphDecodingResult, ToolCallOut } from "@/types";

const PROFILE_KEY = "sphinx.seeker";
const HISTORY_KEY = "sphinx.decodeHistory";
const MAX_HISTORY = 10;
const MAX_KINGS = 6;
export const MAX_MEMORY_CHARS = 400;   // mirrors the server cap on ChatRequest.seeker_memory

export interface SeekerProfile {
  name: string | null;
  readings: number;
  kings: Record<string, number>;
  lastReadingAt: number | null;
  trialsPassed: number;
  gifts: number;
  firstSeenAt: number;
}

export interface DecodeHistoryEntry {
  id: string;
  sessionId: string | null;
  at: number;
  kings: string[];
  lines: number;
  translation: string;
}

const emptyProfile = (): SeekerProfile => ({
  name: null,
  readings: 0,
  kings: {},
  lastReadingAt: null,
  trialsPassed: 0,
  gifts: 0,
  firstSeenAt: Date.now(),
});

export const loadProfile = (): SeekerProfile => ({ ...emptyProfile(), ...(readJson<SeekerProfile>(PROFILE_KEY) ?? {}) });

const saveProfile = (p: SeekerProfile): void => writeJson(PROFILE_KEY, p);

export const loadDecodeHistory = (): DecodeHistoryEntry[] => readJson<DecodeHistoryEntry[]>(HISTORY_KEY) ?? [];

/** The kings of a result: knowledge-graph names first, else the cartouche labels. */
const kingsOf = (result: GlyphDecodingResult): string[] => {
  const names = result.entities.length
    ? result.entities.map((e) => e.name)
    : result.royalNames.map((n) => n.replace(/^Cartouche:\s*/, "").replace(/\s*\(.*$/, ""));
  return [...new Set(names.filter(Boolean))];
};

/** A finished decode: count it, remember its kings, and put it at the top of the history. */
export const recordReading = (result: GlyphDecodingResult, isReread: boolean): void => {
  const kings = kingsOf(result);
  const entry: DecodeHistoryEntry = {
    id: result.id,
    sessionId: result.sessionId,
    at: Date.now(),
    kings,
    lines: result.readingLines.length,
    translation: (result.translation || result.transliteration).slice(0, 160),
  };
  // A re-read replaces its reading's entry instead of adding a new one.
  const history = loadDecodeHistory().filter((h) => !(isReread && result.sessionId && h.sessionId === result.sessionId));
  writeJson(HISTORY_KEY, [entry, ...history].slice(0, MAX_HISTORY));
  if (isReread) return;
  const p = loadProfile();
  p.readings += 1;
  p.lastReadingAt = Date.now();
  for (const k of kings) p.kings[k] = (p.kings[k] ?? 0) + 1;
  saveProfile(p);
};

/** A /chat/ reply: count a passed trial and a carved gift (with the name carved). */
export const recordChatTools = (calls: ToolCallOut[]): void => {
  const passed = calls.some((c) => c.ok && c.tool === "submit_riddle_answer" && c.summary.includes("trial passed"));
  const gift = calls.find((c) => c.ok && c.tool === "create_royal_cartouche");
  if (!passed && !gift) return;
  const p = loadProfile();
  if (passed) p.trialsPassed += 1;
  if (gift) {
    p.gifts += 1;
    const name = String((gift.input as { name?: unknown }).name ?? "").trim();
    if (name) p.name = name.slice(0, 40);
  }
  saveProfile(p);
};

/** One short line for the model; '' for a first-time seeker, so a new visitor costs nothing. */
export const seekerMemoryText = (): string => {
  const p = loadProfile();
  if (!p.readings && !p.trialsPassed && !p.gifts && !p.name) return "";
  const kings = Object.entries(p.kings)
    .sort((a, b) => b[1] - a[1])
    .slice(0, MAX_KINGS)
    .map(([k, n]) => (n > 1 ? `${k} x${n}` : k))
    .join(", ");
  const parts = [
    p.name && `name: ${p.name}`,
    p.readings && `walls brought to be read: ${p.readings}${kings ? ` (${kings})` : ""}`,
    p.lastReadingAt && `last reading: ${new Date(p.lastReadingAt).toISOString().slice(0, 10)}`,
    p.trialsPassed && `riddle trials passed: ${p.trialsPassed}`,
    p.gifts && `cartouche gifts received: ${p.gifts}`,
  ].filter(Boolean);
  return `returning seeker; ${parts.join("; ")}`.slice(0, MAX_MEMORY_CHARS);
};
