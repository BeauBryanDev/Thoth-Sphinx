// ============================================================
// Reading chat API — POST /chat/reading/
// ============================================================
// Stateless like /transliterate/: every turn sends a snapshot of the
// reading on screen, so Thot-Sphinx never starts from scratch. Keep the
// snapshot shape in sync with ReadingSnapshot in app/schemas/reading_chat.py.

import http from "@/services/http";
import type {
  ChatMessage,
  GlyphDecodingResult,
  LineEvidence,
  SignCorrection,
  TextContext,
  ToolCallOut,
} from "@/types";

const READING_CHAT_TIMEOUT_MS = 90_000;   // a turn may include corpus searches

export interface ReadingSnapshot {
  lines: { line: number; codes: string[]; transliteration: string; gloss: string; confidence: string }[];
  full_transliteration: string;
  full_translation: string;
  cartouches: string[];
  evidence: LineEvidence[];
  image_notes: string[];
  model: string;
  escalated: boolean;
  escalation_reasons: string[];
  second_opinion_reason: string;
  direction: GlyphDecodingResult["direction"];
  layout: GlyphDecodingResult["layout"];
  context: TextContext;
}

export interface ReadingChatReply {
  reply: string;
  messageId: string;
  corrections: SignCorrection[];
  toolCalls: ToolCallOut[];
  model: string;
  costUsd: number | null;
}

/** The reading on screen, in the shape the backend expects. */
export const buildReadingSnapshot = (
  result: GlyphDecodingResult,
  context: TextContext,
): ReadingSnapshot => ({
  lines: result.readingLines,
  full_transliteration: result.transliteration,
  full_translation: result.translation,
  cartouches: result.royalNames,
  evidence: result.evidence,
  image_notes: result.imageNotes,
  model: result.reading?.model ?? "",
  escalated: result.reading?.escalated ?? false,
  escalation_reasons: result.reading?.reasons ?? [],
  second_opinion_reason: result.secondOpinionReason,
  direction: result.direction,
  layout: result.layout,
  context,
});

export const sendReadingChat = async (payload: {
  prompt: string;
  history: Pick<ChatMessage, "role" | "content">[];
  reading: ReadingSnapshot;
  image?: string;
}): Promise<ReadingChatReply> => {
  const { data } = await http.post<{
    reply: string;
    message_id: string;
    corrections: SignCorrection[];
    tool_calls: ToolCallOut[];
    model: string;
    cost_usd: number | null;
  }>("/chat/reading/", payload, { timeout: READING_CHAT_TIMEOUT_MS });
  return {
    reply: data.reply,
    messageId: data.message_id,
    corrections: data.corrections ?? [],
    toolCalls: data.tool_calls ?? [],
    model: data.model,
    costUsd: data.cost_usd ?? null,
  };
};
