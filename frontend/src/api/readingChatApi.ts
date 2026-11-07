// ============================================================
// Reading chat API — POST /chat/reading/
// ============================================================
// Agent state: the server keeps the reading by its reading_id, so a turn
// sends only the id. The snapshot goes along only when the server has lost
// the reading (410), as a fallback. Keep the snapshot shape in sync with
// ReadingSnapshot in app/schemas/reading_chat.py.

import axios from "axios";
import http from "@/services/http";
import type {
  BookPassage,
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
  bookPassages: BookPassage[];
  model: string;
  costUsd: number | null;
  /** Whose copy of the reading the agent read: the server session or the client snapshot. */
  stateSource: "server" | "client";
  readingVersion: number | null;
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

type ReadingChatWire = {
  reply: string;
  message_id: string;
  corrections: SignCorrection[];
  tool_calls: ToolCallOut[];
  book_passages?: BookPassage[];
  model: string;
  cost_usd: number | null;
  state_source?: "server" | "client";
  reading_version?: number | null;
};

export const sendReadingChat = async (payload: {
  prompt: string;
  history: Pick<ChatMessage, "role" | "content">[];
  /** The server session of the reading; null when the server did not keep it. */
  readingId: string | null;
  /** Built only when needed: no session, or the server answered 410. */
  snapshot: () => ReadingSnapshot;
  image?: string;
}): Promise<ReadingChatReply> => {
  const { prompt, history, readingId, snapshot, image } = payload;
  const post = (body: Record<string, unknown>) =>
    http.post<ReadingChatWire>("/chat/reading/", body, { timeout: READING_CHAT_TIMEOUT_MS });

  let data: ReadingChatWire;
  try {
    ({ data } = await post(
      readingId
        ? { prompt, history, reading_id: readingId, image }
        : { prompt, history, reading: snapshot(), image },
    ));
  } catch (err) {
    // 410: the server forgot the reading (restart or idle timeout); send our copy once.
    if (!readingId || !(axios.isAxiosError(err) && err.response?.status === 410)) throw err;
    ({ data } = await post({ prompt, history, reading: snapshot(), image }));
  }
  return {
    reply: data.reply,
    messageId: data.message_id,
    corrections: data.corrections ?? [],
    toolCalls: data.tool_calls ?? [],
    bookPassages: data.book_passages ?? [],
    model: data.model,
    costUsd: data.cost_usd ?? null,
    stateSource: data.state_source ?? "client",
    readingVersion: data.reading_version ?? null,
  };
};
