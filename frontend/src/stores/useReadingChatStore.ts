// ============================================================
// useReadingChatStore — the conversation about the reading on screen
//
// Keyed by the decode result's id: a new decode starts a new
// conversation, navigating away and back keeps it (Zustand, like the
// other stores). Every turn sends a fresh snapshot of the reading from
// useGlyphStore, so the agent always discusses what the user sees.
// Same {messages, isLoading, error, send, reset} shape as useChatStore,
// so <ChatPanel /> renders it unchanged.
// ============================================================

import { create } from "zustand";
import { RereadLimitError } from "@/api/glyphApi";
import { buildReadingSnapshot, sendReadingChat } from "@/api/readingChatApi";
import { useGlyphStore } from "@/stores/useGlyphStore";
import { generateId } from "@/utils/formatters";
import type { GlyphDecodingResult, ReadingChatMessage, SignCorrection } from "@/types";

// Mirrors the server's guardrail (reread_cooldown_s): one Apply & re-read per hour.
// The server is the authority; this only disables the buttons and says when to come back.
export const REREAD_COOLDOWN_MS = 60 * 60 * 1000;

// Longer than the general chat's 4: the user is working through one reading.
const MEMORY_WINDOW = 12;
const GREETING_ID = "reading-greeting";

const modelName = (id: string): string =>
  id.includes("sonnet") ? "Claude Sonnet" : id.includes("haiku") ? "Claude Haiku" : id;

/** A local first message that summarises the reading; no LLM call. */
const greeting = (result: GlyphDecodingResult): ReadingChatMessage => {
  const n = result.readingLines.length;
  const supported = result.evidence.filter((e) => e.verdict === "strong" || e.verdict === "possible").length;
  const who = result.reading?.escalated
    ? `I read it with a second opinion from ${modelName(result.reading.model)}`
    : "I read it";
  const corpus = result.evidence.length
    ? ` The Thesaurus Linguae Aegyptiae has parallels for ${supported} of ${n} line${n === 1 ? "" : "s"}.`
    : "";
  return {
    id: GREETING_ID,
    role: "assistant",
    content:
      `Greetings, seeker. This inscription has ${n} line${n === 1 ? "" : "s"}; ${who}.${corpus}\n\n` +
      "Ask me why I read a line the way I did, what the corpus says about it, " +
      "or tell me if a sign looks wrong — I will check it and propose a fix for you to accept.",
    timestamp: Date.now(),
  };
};

export type CorrectionStatus = "pending" | "applying" | "applied" | "dismissed" | "failed";

/** Key of a proposed correction: the message it came with and its index there. */
export const correctionKey = (messageId: string, index: number): string => `${messageId}:${index}`;

export interface ReadingChatStore {
  readingId: string | null;
  open: boolean;
  messages: ReadingChatMessage[];
  isLoading: boolean;
  error: string | null;
  /** Status per proposed correction; missing = pending. */
  correctionStatus: Record<string, CorrectionStatus>;
  /** When the next Apply & re-read is allowed (ms since epoch); null = now. */
  rereadAvailableAt: number | null;
  applyCorrection: (messageId: string, index: number) => Promise<void>;
  dismissCorrection: (messageId: string, index: number) => void;
  /** The user's own correction from the drawer form: posted as a card, then applied at once. */
  applyManualCorrection: (c: SignCorrection) => Promise<void>;
  openFor: (result: GlyphDecodingResult) => void;
  close: () => void;
  send: (prompt: string) => Promise<void>;
  reset: () => void;
}

export const useReadingChatStore = create<ReadingChatStore>((set, get) => ({
  readingId: null,
  open: false,
  messages: [],
  isLoading: false,
  error: null,
  correctionStatus: {},
  rereadAvailableAt: null,

  openFor: (result) => {
    if (get().readingId !== result.id) {
      // A different reading: start a new conversation about it.
      set({ readingId: result.id, messages: [greeting(result)], error: null, isLoading: false,
            correctionStatus: {} });
    }
    set({ open: true });
  },

  close: () => set({ open: false }),

  send: async (prompt: string) => {
    const trimmed = prompt.trim();
    const { result, context, image } = useGlyphStore.getState();
    if (!trimmed || get().isLoading || !result) return;
    if (get().readingId !== result.id) get().openFor(result);

    const userMsg: ReadingChatMessage = {
      id: generateId(),
      role: "user",
      content: trimmed,
      timestamp: Date.now(),
    };
    // The greeting is local and the snapshot already says the same: don't send it.
    const history = get()
      .messages.filter((m) => m.id !== GREETING_ID)
      .slice(-MEMORY_WINDOW)
      .map((m) => ({ role: m.role, content: m.content }));

    set((s) => ({ messages: [...s.messages, userMsg], isLoading: true, error: null }));
    const readingId = result.id;

    try {
      const reply = await sendReadingChat({
        prompt: trimmed,
        history,
        reading: buildReadingSnapshot(result, context),
        image: image?.preview,
      });
      // The user may have decoded another image meanwhile; drop a stale reply.
      if (get().readingId !== readingId) return;
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: reply.messageId || generateId(),
            role: "assistant",
            content: reply.reply,
            timestamp: Date.now(),
            toolCalls: reply.toolCalls,
            corrections: reply.corrections,
          },
        ],
      }));
    } catch (err) {
      if (get().readingId === readingId) {
        set({ error: err instanceof Error ? err.message : "Failed to reach the Sphinx." });
      }
    } finally {
      if (get().readingId === readingId) set({ isLoading: false });
    }
  },

  reset: () => {
    const { result } = useGlyphStore.getState();
    set({ messages: result ? [greeting(result)] : [], error: null, isLoading: false, correctionStatus: {} });
  },

  applyManualCorrection: async (c) => {
    const availableAt = get().rereadAvailableAt;
    if (get().isLoading || (availableAt && Date.now() < availableAt)) return;
    const id = generateId();
    set((s) => ({
      messages: [
        ...s.messages,
        {
          id,
          role: "assistant",
          content:
            `You corrected **line ${c.line}, sign ${c.position}: ${c.from_code} → ${c.to_code}** ` +
            "from your own look at the stone. I will re-read the whole inscription with it.",
          timestamp: Date.now(),
          corrections: [c],
        },
      ],
    }));
    await get().applyCorrection(id, 0);
  },

  dismissCorrection: (messageId, index) =>
    set((s) => ({ correctionStatus: { ...s.correctionStatus, [correctionKey(messageId, index)]: "dismissed" } })),

  applyCorrection: async (messageId, index) => {
    const key = correctionKey(messageId, index);
    const message = get().messages.find((m) => m.id === messageId);
    const correction = message?.corrections?.[index];
    if (!correction || get().isLoading) return;
    const availableAt = get().rereadAvailableAt;
    if (availableAt && Date.now() < availableAt) {
      set({ error: `One correction re-read per hour. Next one at ${new Date(availableAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}.` });
      return;
    }

    const setStatus = (status: CorrectionStatus) =>
      set((s) => ({ correctionStatus: { ...s.correctionStatus, [key]: status } }));
    setStatus("applying");
    set({ isLoading: true, error: null });

    try {
      const updated = await useGlyphStore.getState().applyCorrection(correction);
      // Same conversation, new reading: follow the new result id so the chat is not reset.
      set((s) => ({
        readingId: updated.id,
        messages: [
          ...s.messages,
          {
            id: generateId(),
            role: "assistant",
            content:
              `✓ Applied your correction — line ${correction.line}, sign ${correction.position}: ` +
              `**${correction.from_code} → ${correction.to_code}**. I re-read the whole inscription ` +
              `with it:\n\n> ${updated.translation || updated.transliteration}\n\n` +
              "The cards on the page now show the new reading, its corpus evidence and its trace.",
            timestamp: Date.now(),
          },
        ],
      }));
      setStatus("applied");
      set({ rereadAvailableAt: Date.now() + REREAD_COOLDOWN_MS });
    } catch (err) {
      if (err instanceof RereadLimitError) {
        // Not a failure of this correction: it can still be applied once the hour is up.
        setStatus("pending");
        set({ rereadAvailableAt: Date.now() + err.retryAfterSeconds * 1000, error: err.message });
        return;
      }
      setStatus("failed");
      set({ error: err instanceof Error ? err.message : "The re-read failed." });
    } finally {
      set({ isLoading: false });
    }
  },
}));
