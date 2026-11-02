// ============================================================
// Domain Types — Central Type Definitions for SphinxEyes
// ============================================================

// ---------- Chat / AI ----------
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: number;
}

// ---------- Backend contract (POST /predict/) ----------
export type ReadingDirection = "rtl" | "ltr";
export type WallLayout = "rows" | "columns";
export type EnhancePreset = "none" | "default" | "aggressive" | "gentle";

export interface DecodeOptions {
  direction: ReadingDirection;
  layout: WallLayout;          // required by the backend — no auto-detect
  preset?: EnhancePreset;      // defaults to "none" (recommended)
  translate?: boolean;         // run the LLM transliteration stage (default true)
  context?: TextContext;       // archaeological context — all fields optional
}

// Archaeological context for the LLM stage. Every field defaults to
// "unknown" server-side, so a naive tourist can leave everything blank.
export interface TextContext {
  period?: string;
  text_type?: string;
  support?: string;
  location_type?: string;
  site?: string;
  dynasty?: string;
  kings_reign?: string;
}

export interface ChunkTransliteration {
  chunk_index: number;
  gardiner_codes: string[];
  transliteration: string;
  english_gloss: string;
  linguistic_notes: string;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  period_note: string;
  is_cartouche: boolean;
}

export interface TransliterationOut {
  chunks: ChunkTransliteration[];
  full_transliteration: string;
  full_translation: string;
  model: string;
  n_chunks: number;
  error: string | null;
  escalated?: boolean;              // a Sonnet second opinion produced this reading
  escalation_reasons?: string[];    // router reason codes (app/llm/router.py)
  escalation_note?: string | null;  // e.g. second opinion failed; first reading kept
  cost_usd?: number | null;
  image_notes?: string[];           // Claude's notes on the photograph
}

export interface SegmentedWordOut {
  codes: string[];
  translit: string;
  translation: string;
  freq: number;
  edit_dist: number;
  confidence: number;
  source: string;
}

export interface CorrectionOut {
  segmented_words: SegmentedWordOut[];
  flat_corrected_seq: string[];
  flat_translit: string;
  flat_translation: string;
  score: number;
  had_fallback: boolean;
}

export interface CartoucheOut {
  bbox: [number, number, number, number];
  n_members: number;
  inferred: boolean;
  translit: string | null;
  english: string | null;
  spelling: string[] | null;
  score: number | null;
  verified: boolean | null;
  /** Raw interior top-1 Gardiner codes — present even when the royal-name
   *  match REFUSED (translit null), so the LLM stage still sees the signs. */
  interior_codes?: string[] | null;
}

export interface PredictResponse {
  layout: WallLayout;
  direction: ReadingDirection;
  image_shape: [number, number];
  n_detections: number;
  n_cartouches: number;
  outer: {
    slots: [string, number][][];
    boundary_hints: number[];
    n_synthetic: number;
    correction: CorrectionOut;
  };
  cartouches: CartoucheOut[];
  transliteration: TransliterationOut | null;
  /** YOLO-annotated image (boxes + labels) as a JPEG data URL. */
  annotated_image: string | null;
}

// ---------- Backend contract (POST /reverse/) ----------
export type ReverseRegister =
  | "unknown"
  | "monumental"
  | "literary"
  | "letter"
  | "religious";

export interface ReverseWord {
  english: string;
  transliteration: string;
  gardiner_codes: string[];
  literal: string;
  note: string;
}

export interface ReverseTranslationOut {
  source_text: string;
  normalized_english: string;
  transliteration: string;
  gardiner_codes: string[];
  words: ReverseWord[];
  grammar_notes: string;
  confidence: "HIGH" | "MEDIUM" | "LOW";
  model: string;
  error: string | null;
}

// ---------- Glyph Decoder ----------
export interface GlyphDecodingResult {
  id: string;
  originalImage: string;
  detectedGlyphs: string[];
  /** Codes grouped by physical line (boundary_hints) — spatial order. */
  lines: string[][];
  /** Per-sign detector confidence, aligned with detectedGlyphs (flat
      reading order), each clamped to [0, 0.998]. */
  signConfidences: number[];
  /** Annotated detection image (JPEG data URL) for display + download. */
  annotatedImage: string | null;
  layout: WallLayout;
  direction: ReadingDirection;
  /** Lexicon-verified cartouche headlines, e.g. "Cartouche: Unas (Dynasty V) (wnjs)". */
  royalNames: string[];
  transliteration: string;
  /** English gloss only — royal names are NOT merged in anymore. */
  translation: string;
  confidence: number;
  processedAt: number;
  /** Who produced the LLM reading and why; null when the local fallback is shown. */
  reading: ReadingProvenance | null;
  /** Claude's notes on the photograph; empty when there are none. */
  imageNotes: string[];
}

/** Provenance of the LLM reading: first reader, router decision, cost. */
export interface ReadingProvenance {
  model: string;
  escalated: boolean;
  reasons: string[];
  note: string | null;
  costUsd: number | null;
}

export interface UploadedImage {
  file: File;
  preview: string;
}

// ---------- Analytics / Dashboard ----------
export interface AnalyticsMetric {
  label: string;
  value: number;
  unit?: string;
  trend?: "up" | "down" | "neutral";
}

export interface TimeSeriesPoint {
  name: string; // x-axis label (e.g., day, month)
  queries: number;
  decodings: number;
}

// ---------- Navigation ----------
export type PageKey =
  | "home"
  | "chat"
  | "glyphs"
  | "transliteration"
  | "learn"
  | "profile";

export interface NavItem {
  key: PageKey;
  label: string;
  icon: string; // emoji or svg id
}

// ---------- User ----------
export interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  plan: "free" | "scholar" | "pharaoh";
  joinedAt: number;
}

// ---------- Generic API envelope ----------
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}
