// ============================================================
// Domain Types — Central Type Definitions for SphinxEyes
// ============================================================

// ---------- Chat / AI ----------
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: number;
  /** Tools Thot-Sphinx called for this reply (assistant messages only). */
  toolCalls?: ToolCallOut[];
  /** Book passages search_books retrieved for this reply (top 5, assistant messages only). */
  bookPassages?: BookPassage[];
  /** WebP data URL of the royal cartouche gift carved for this reply (assistant messages only). */
  cartoucheImage?: string;
}

/** One passage from the Egyptology books (mirrors app/schemas/books.py). */
export interface BookPassage {
  chunk_id: string;
  book: string | null;
  author: string | null;
  section: string;
  pages: string;
  citation: string;            // "Book › Section › p. N"
  similarity: number | null;   // cosine to the question, 0–1; null when meaning search was down
  matched_by: string[];        // "words" and/or "meaning"
  text: string;                // clipped to ~1000 characters
  weak: boolean;               // the search found no strong support
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
  evidence?: LineEvidence[];        // closest corpus match per line (TLA)
  tool_calls?: ToolCallOut[];       // every tool call the models made
  trace?: TraceEvent[];             // the whole read, step by step
  entities?: EntityCard[];          // kings of the matched cartouches (knowledge graph)
  semantic?: SemanticTranslation | null; // second translation; full_translation stays the raw LLM one
  reading_id?: string | null;       // server session for the reading chat (agent state)
  reading_version?: number | null;  // 1 = first read, +1 per correction re-read
}

/** A published translation passage — mirrors app/schemas/transliterations.py TranslationPassage. */
export interface TranslationPassage {
  passage_id: string;
  work: string;
  author: string;
  king: string | null;
  location: string | null;
  section: string | null;
  spell: number | null;
  title: string | null;
  page: string | null;
  citation: string;
  similarity: number | null;
  rerank_score?: number | null;     // Cohere relevance 0–1; null when rerank was unavailable
  hybrid_rank?: number | null;      // the spell's place before reranking
  matched_by: string[];
  text: string;                     // the matching part
  spell_text?: string;              // the whole spell (small-to-big), what the editor read
}

/** The LLM translation smoothed against the closest published passages. */
export interface SemanticTranslation {
  query: string;
  king_filter: string | null;
  search_mode: string;
  passages: TranslationPassage[];
  translation: string;
  match: "same_text" | "related" | "none";
  used_passage: string | null;
  note: string;
  model: string | null;
  cost_usd: number | null;
  latency_ms: number | null;
  error: string | null;
}

/** The king behind a matched cartouche — mirrors app/schemas/transliterations.py EntityCard. */
export interface EntityCard {
  id: string;
  name: string;
  cartouche: string;
  dynasty: string | null;
  period: string | null;
  reign: string | null;
  notes: string;
  monuments: string[];
  wikipedia_url: string | null;
  source: string;
  reliability: string;
}

/** The closest corpus sentence for a line — mirrors app/schemas/transliterations.py. */
export interface CorpusMatch {
  sentence_id: string;
  rating: string;
  transliteration: string;
  translation: string;
  translation_lang: "en" | "de";
  date: string | null;
  text_name: string | null;
  matched_by: string[];
}

export interface LineEvidence {
  line: number;
  verdict: string;                  // strong | possible | none | too_short | unrated | error
  query_translit: string | null;
  best: CorpusMatch | null;
  searched_by: string;              // a model id, or "evidence_check"
}

export interface ToolCallOut {
  model: string;
  tool: string;
  input: Record<string, unknown>;
  summary: string;
  ok: boolean;
  latency_ms: number;
}

export interface TraceEvent {
  step: number;
  t_ms: number;
  kind: string;                     // detection | llm_call | tool_call | router | escalation | evidence_check | final | error
  actor: string;
  summary: string;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cost_usd: number | null;
  detail: Record<string, unknown>;
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
  /** Agent activity: corpus evidence per line, tool calls, and the full trace. */
  evidence: LineEvidence[];
  toolCalls: ToolCallOut[];
  /** Kings of the matched cartouches, from the knowledge graph. */
  entities: EntityCard[];
  /** Second translation from the published-translation search; null when it did not run. */
  semantic: SemanticTranslation | null;
  trace: TraceEvent[];
  /** Claude's reading per physical line (for the reading chat snapshot). */
  readingLines: ReadingLineView[];
  /** The first reader's own words when it asked for a second opinion ('' otherwise). */
  secondOpinionReason: string;
  /** Server-side session of this reading (agent state); null when the server did not keep it. */
  sessionId: string | null;
  /** 1 = first read, +1 per correction re-read. */
  readingVersion: number | null;
}

export interface ReadingLineView {
  line: number;
  codes: string[];
  transliteration: string;
  gloss: string;
  confidence: string;
}

/** A sign the agent proposes to change — mirrors app/schemas/reading_chat.py. */
export interface SignCorrection {
  line: number;
  position: number;
  from_code: string;
  to_code: string;
  reason: string;
}

/** A chat message about the reading: may carry corpus tool calls and proposed corrections. */
export interface ReadingChatMessage extends ChatMessage {
  toolCalls?: ToolCallOut[];
  corrections?: SignCorrection[];
  /** The user's own corrections from the drawer form: applied together in one re-read. */
  correctionBatch?: boolean;
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
  | "rosetta"
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
