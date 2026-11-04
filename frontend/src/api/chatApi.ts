// ============================================================
// Chat API — endpoint wrappers for SphinxChat (Thot-Sphinx)
// ============================================================

import http from "@/services/http";
import type { BookPassage, ChatMessage, ToolCallOut } from "@/types";

export interface SendMessagePayload {
  prompt: string;
  history?: Pick<ChatMessage, "role" | "content">[];
}

export interface SendMessageResponse {
  reply: string;
  messageId: string;
  toolCalls: ToolCallOut[];
  bookPassages: BookPassage[];
}

const CHAT_TIMEOUT_MS = 60_000;

/** Send a prompt to Thot-Sphinx. Backend returns the reply bare (no envelope). */
export const sendChatMessage = async (
  payload: SendMessagePayload,
): Promise<SendMessageResponse> => {
  const { data } = await http.post<{
    reply: string;
    message_id: string;
    tool_calls?: ToolCallOut[];
    book_passages?: BookPassage[];
  }>(
    "/chat/",
    { prompt: payload.prompt, history: payload.history ?? [] },
    { timeout: CHAT_TIMEOUT_MS },
  );
  return {
    reply: data.reply,
    messageId: data.message_id,
    toolCalls: data.tool_calls ?? [],
    bookPassages: data.book_passages ?? [],
  };
};

/**
 * Fetch conversation history.
 * NOT IMPLEMENTED on the backend yet (Phase 6, needs PostgreSQL) —
 * conversations are client-side only; returns an empty list.
 */
export const fetchChatHistory = async (
  _sessionId: string,
): Promise<ChatMessage[]> => {
  return [];
};
