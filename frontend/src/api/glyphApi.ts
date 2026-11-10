// ============================================================
// Glyph API — endpoint wrappers for the hieroglyph decoder
//
// Adapter layer: the backend serves POST /predict/ returning a
// PredictResponse; components consume GlyphDecodingResult. The
// mapping lives HERE, not in components .
// ============================================================

import axios from "axios";
import http from "@/services/http";
import type {
  DecodeOptions,
  GlyphDecodingResult,
  PredictResponse,
  SignCorrection,
  TextContext,
  TransliterationOut,
} from "@/types";
import { isCartouche } from "@/utils/gardinerUnicode";

// Inference on CPU can take a while for dense walls — override the
// 15 s default on this call only.
const DECODE_TIMEOUT_MS = 180_000;   // agent reads with tools + escalation take 45–70 s

/** Map the backend PredictResponse to the UI's GlyphDecodingResult. */
const toDecodingResult = (
  res: PredictResponse,
  imagePreview: string,
): GlyphDecodingResult => {
  const { correction } = res.outer;

  // Mean top-1 confidence over real (non-synthetic) slots.
  // Cartouche markers are places in a line, not signs: they stay out of the mean.
  const top1 = res.outer.slots
    .filter((slot) => !isCartouche(slot[0]?.[0] ?? ""))
    .map((slot) => slot[0]?.[1] ?? 0)
    .filter((c) => c > 0);
  const confidence =
    top1.length > 0 ? top1.reduce((a, b) => a + b, 0) / top1.length : 0;

  // Cartouche royal names — rendered as their own section in the UI.
  const royalNames = res.cartouches
    .filter((c) => c.english)
    .map((c) => `Cartouche: ${c.english} (${c.translit})`);

  // Group the corrected sequence by physical line using boundary_hints
  // (cut positions into the flat sequence) — spatial reading order.
  const codes = correction.flat_corrected_seq;
  const bounds = [...res.outer.boundary_hints]
    .filter((b) => b > 0 && b <= codes.length)
    .sort((a, b) => a - b);
  if (bounds.length === 0 || bounds[bounds.length - 1] !== codes.length) {
    bounds.push(codes.length);
  }
  const lines: string[][] = [];
  let start = 0;
  for (const b of bounds) {
    const line = codes.slice(start, b);
    if (line.length) lines.push(line);
    start = b;
  }

  // Per-sign top-1 confidence aligned with the corrected sequence.
  // Raw scores can exceed 1.0 (obj * class product quirks) — the UI
  // shows probabilities, so clamp to [0, 0.998].
  const signConfidences = codes.map((_, i) =>
    Math.min(res.outer.slots[i]?.[0]?.[1] ?? 0, 0.998),
  );

  // The LLM stage, when it ran successfully, outranks the local
  // dictionary gloss for both transliteration and translation.
  const llm = res.transliteration;
  const llmOk = llm && !llm.error && llm.full_transliteration;

  return {
    id: crypto.randomUUID(),
    originalImage: imagePreview,
    detectedGlyphs: codes,
    lines,
    signConfidences,
    annotatedImage: res.annotated_image ?? null,
    layout: res.layout,
    direction: res.direction,
    royalNames,
    transliteration: llmOk
      ? llm.full_transliteration
      : correction.flat_translit,
    translation: llmOk ? llm.full_translation : correction.flat_translation,
    confidence,
    processedAt: Date.now(),
    reading: llmOk
      ? {
          model: llm.model,
          escalated: llm.escalated ?? false,
          reasons: llm.escalation_reasons ?? [],
          note: llm.escalation_note ?? null,
          costUsd: llm.cost_usd ?? null,
        }
      : null,
    imageNotes: llmOk ? llm.image_notes ?? [] : [],
    evidence: llmOk ? llm.evidence ?? [] : [],
    toolCalls: llmOk ? llm.tool_calls ?? [] : [],
    entities: llm?.entities ?? [],
    semantic: llmOk ? llm.semantic ?? null : null,
    trace: llmOk ? llm.trace ?? [] : [],
    readingLines: llmOk
      ? llm.chunks.map((c) => ({
          line: c.chunk_index + 1,
          codes: c.gardiner_codes,
          transliteration: c.transliteration,
          gloss: c.english_gloss,
          confidence: c.confidence,
        }))
      : lines.map((codes, i) => ({ line: i + 1, codes, transliteration: "", gloss: "", confidence: "LOW" })),
    secondOpinionReason: llmOk
      ? String(llm.trace?.find((e) => e.kind === "router")?.detail?.haiku_reason ?? "")
      : "",
    sessionId: llmOk ? llm.reading_id ?? null : null,
    readingVersion: llmOk ? llm.reading_version ?? null : null,
  };
};

const buildForm = (file: File, options: DecodeOptions): FormData => {
  const form = new FormData();
  form.append("file", file);
  form.append("direction", options.direction);
  form.append("layout", options.layout);
  form.append("preset", options.preset ?? "none");
  form.append("translate", String(options.translate ?? true));
  const ctx = options.context ?? {};
  for (const key of [
    "period",
    "text_type",
    "support",
    "location_type",
    "site",
    "dynasty",
    "kings_reign",
  ] as const) {
    const value = ctx[key];
    if (value && value.trim()) form.append(key, value.trim());
  }
  return form;
};

/** Raw variant for callers that need slots / cartouches / hints. */
export const decodeGlyphsRaw = async (
  file: File,
  options: DecodeOptions,
): Promise<PredictResponse> => {
  // The backend returns PredictResponse directly (no ApiResponse
  // envelope), so call the axios instance rather than request<T>().
  const { data } = await http.post<PredictResponse>(
    "/predict/",
    buildForm(file, options),
    {
      headers: { "Content-Type": "multipart/form-data" },
      timeout: DECODE_TIMEOUT_MS,
    },
  );
  return data;
};

/** Upload an image of hieroglyphs for detection + transliteration. */
export const decodeGlyphs = async (
  file: File,
  options: DecodeOptions,
  imagePreview = "",
): Promise<GlyphDecodingResult> => {
  const raw = await decodeGlyphsRaw(file, options);
  return toDecodingResult(raw, imagePreview);
};

/**
 * Fetch previous decodings for the current user.
 * NOT IMPLEMENTED on the backend yet (Phase 6, needs PostgreSQL).
 * Returns an empty list so the UI renders without a history service.
 */
export const fetchDecodingHistory = async (): Promise<
  GlyphDecodingResult[]
> => {
  return [];
};

/**
 * Second stage: LLM transliteration of an already-detected sequence.
 * Called when the user hits Decode Glyphs; detection ran earlier in the
 * background (fired on direction selection) for latency hiding.
 */
export const transliterateGlyphs = async (
  detection: PredictResponse,
  context: TextContext,
  // The raw upload as a data URL (never the annotated image); the backend
  // downscales it to 640 px and shows it to the LLM next to the codes.
  image?: string,
  // Signs the user corrected in the reading chat (human in the loop); already applied in the codes.
  humanCorrections: Omit<SignCorrection, "reason">[] = [],
  // The server session of the reading being re-read, so it keeps its ID (agent state).
  sessionId: string | null = null,
): Promise<TransliterationOut> => {
  const { correction } = detection.outer;
  const body = {
    codes: correction.flat_corrected_seq,
    confidences: detection.outer.slots.map((slot) => slot[0]?.[1] ?? 0),
    boundary_hints: detection.outer.boundary_hints,
    // Mirrors the backend one-shot path (transliteration_services.py):
    // matched cartouches are authoritative; REFUSED ones still forward
    // their raw interior signs instead of vanishing from the LLM input.
    cartouche_names: detection.cartouches.map((c) => {
      if (c.translit) {
        return `${c.translit} — ${c.english ?? ""} (interior: ${(c.spelling ?? []).join(" ")})`;
      }
      const rawCodes = (c.interior_codes ?? []).join(" ");
      return rawCodes
        ? `[UNRESOLVED cartouche — raw signs detected but no confident royal-name match: ${rawCodes}]`
        : "[UNRESOLVED cartouche — no signs detected]";
    }),
    direction: detection.direction,
    layout: detection.layout,
    context,
    image,
    human_corrections: humanCorrections.map(({ line, position, from_code, to_code }) => ({
      line, position, from_code, to_code,
    })),
    reading_id: sessionId ?? undefined,
  };
  try {
    const { data } = await http.post<TransliterationOut>(
      "/transliterate/",
      body,
      { timeout: DECODE_TIMEOUT_MS },
    );
    return data;
  } catch (err) {
    // The server allows one correction re-read per client per hour (app/routers/transliterate.py).
    // Other 429s (daily quota, request window) carry another X-Guard and surface as plain errors.
    if (axios.isAxiosError(err) && err.response?.status === 429
        && err.response.headers["x-guard"] === "reread") {
      const retry = Number(err.response.headers["retry-after"]) || 3600;
      const detail = (err.response.data as { detail?: string } | undefined)?.detail;
      throw new RereadLimitError(retry, detail ?? "One correction re-read per hour.");
    }
    throw err;
  }
};

/** Raised when the server's one-re-read-per-hour guardrail refuses a correction. */
export class RereadLimitError extends Error {
  constructor(public readonly retryAfterSeconds: number, message: string) {
    super(message);
    this.name = "RereadLimitError";
  }
}

/** Merge a detection + LLM result into the UI's GlyphDecodingResult. */
export const mergeDecodingResult = (
  detection: PredictResponse,
  translit: TransliterationOut | null,
  imagePreview: string,
): GlyphDecodingResult =>
  toDecodingResult(
    { ...detection, transliteration: translit },
    imagePreview,
  );
