// ============================================================
// useChatStore — Zustand store for the SphinxChat conversation
//
// Global (module-level) store: the conversation SURVIVES page
// navigation, unlike component useState which dies on unmount.
// No persistence yet — a page reload still clears it (Phase 6
// PostgreSQL will add real history).
// ============================================================

import { create } from "zustand";
import { sendChatMessage } from "@/api/chatApi";
import { generateId } from "@/utils/formatters";
import type { ChatMessage } from "@/types";

// Short conversational memory: only the last N messages travel to the
// backend each turn.
// 10 messages: a riddle trial runs over many turns; the seeker's name must still be in view.
const MEMORY_WINDOW = 10;

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

export const useChatStore = create<ChatStore>((set, get) => ({
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
        });
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

  reset: () => set({ messages: [INITIAL_MESSAGE], error: null, riddleSession: null }),
}));
