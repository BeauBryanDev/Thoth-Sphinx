// ============================================================
// useChatStore — Zustand store for the SphinxChat conversation
//
// Global (module-level) store: the conversation survives page
// navigation, and since agent memory (2026-11-06) it is persisted to
// localStorage, so a reload keeps it too. Each turn also sends one short
// line of seeker memory (utils/seekerMemory.ts): /chat/ only.
// ============================================================

import { create } from "zustand";
import { persist } from "zustand/middleware";
import { sendChatMessage } from "@/api/chatApi";
import { generateId } from "@/utils/formatters";
import { recordChatTools, seekerMemoryText } from "@/utils/seekerMemory";
import { memoryStorage } from "@/utils/storage";
import type { ChatMessage } from "@/types";

// Short conversational memory: only the last N messages travel to the
// backend each turn.
// 10 messages: a riddle trial runs over many turns; the seeker's name must still be in view.
const MEMORY_WINDOW = 10;
// Kept in localStorage; gift images are large, so only the latest one is kept.
const STORED_MESSAGES = 50;

const INITIAL_MESSAGE: ChatMessage = {
  id: "welcome",
  role: "assistant",
  content:
    "Greetings, seeker of knowledge. I am Sphinx, guardian of ancient wisdom. Ask me anything about Ancient Egypt — its history, gods, pharaohs, and mysteries.",
  timestamp: Date.now(),
};

export interface ChatStore {
  messages: ChatMessage[];
  isLoading: boolean;
  error: string | null;
  /** Server-side riddle game ID; the trial's state lives on the server, never here. */
  riddleSession?: string | null;
  send: (prompt: string) => Promise<void>;
  reset: () => void;
}

/** What goes to localStorage: the last messages, with only the newest gift image. */
const storedMessages = (messages: ChatMessage[]): ChatMessage[] => {
  const kept = messages.slice(-STORED_MESSAGES);
  const lastGift = kept.map((m) => Boolean(m.cartoucheImage)).lastIndexOf(true);
  return kept.map((m, i) => (m.cartoucheImage && i !== lastGift ? { ...m, cartoucheImage: undefined } : m));
};

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      messages: [INITIAL_MESSAGE],
      isLoading: false,
      error: null,
      riddleSession: null,

      send: async (prompt: string) => {
        const trimmed = prompt.trim();
        if (!trimmed || get().isLoading) return;

        const userMsg: ChatMessage = {
          id: generateId(),
          role: "user",
          content: trimmed,
          timestamp: Date.now(),
        };
        const history = get()
          .messages.filter((m) => m.id !== "welcome")
          .slice(-MEMORY_WINDOW)
          .map((m) => ({ role: m.role, content: m.content }));

        set((s) => ({
          messages: [...s.messages, userMsg],
          isLoading: true,
          error: null,
        }));

        try {
          const { reply, messageId, toolCalls, bookPassages, riddleSession, cartoucheImage } =
            await sendChatMessage({
              prompt: trimmed,
              history,
              riddleSession: get().riddleSession,
              seekerMemory: seekerMemoryText(),
            });
          recordChatTools(toolCalls);
          const assistantMsg: ChatMessage = {
            id: messageId || generateId(),
            role: "assistant",
            content: reply,
            timestamp: Date.now(),
            toolCalls,
            bookPassages,
            cartoucheImage,
          };
          set((s) => ({ messages: [...s.messages, assistantMsg], riddleSession }));
        } catch (err) {
          set({
            error:
              err instanceof Error ? err.message : "Failed to reach the Sphinx.",
          });
        } finally {
          set({ isLoading: false });
        }
      },

      // A new conversation; the seeker memory stays (Forget me clears it).
      reset: () => set({ messages: [INITIAL_MESSAGE], error: null, riddleSession: null }),
    }),
    {
      name: "sphinx.chat",
      version: 1,
      storage: memoryStorage,
      partialize: (s) => ({ messages: storedMessages(s.messages), riddleSession: s.riddleSession }),
    },
  ),
);
