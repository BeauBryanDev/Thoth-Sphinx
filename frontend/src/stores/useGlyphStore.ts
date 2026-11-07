// ============================================================
// useGlyphStore — Zustand store for the glyph decoder workflow
//
// Global store so an uploaded image, its background detection and
// the decode result all SURVIVE navigating away to Chat/Reverse
// and back. Two-stage latency-hiding flow preserved:
//   1. picking a reading direction fires /predict/ in background;
//   2. "Decode Glyphs" awaits it, then calls /transliterate/.
// ============================================================

import { create } from "zustand";
import { persist } from "zustand/middleware";
import {
  decodeGlyphsRaw,
  mergeDecodingResult,
  transliterateGlyphs,
} from "@/api/glyphApi";
import { buildReadingSnapshot } from "@/api/readingChatApi";
import { saveReadingSession } from "@/utils/readingSession";
import { recordReading } from "@/utils/seekerMemory";
import { memoryStorage } from "@/utils/storage";
import type {
  GlyphDecodingResult,
  PredictResponse,
  ReadingDirection,
  SignCorrection,
  TextContext,
  UploadedImage,
  WallLayout,
} from "@/types";

/** Keep the reading's server session and its snapshot in localStorage (the chat's fallback). */
const rememberSession = (result: GlyphDecodingResult, context: TextContext): void => {
  if (!result.sessionId) return;
  saveReadingSession({
    sessionId: result.sessionId,
    version: result.readingVersion,
    snapshot: buildReadingSnapshot(result, context),
    savedAt: Date.now(),
  });
};

const readAsDataUrl = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });

// In-flight background detection. Module-level (not store state):
// it's never rendered, and promises don't belong in reactive state.
interface PendingDetection {
  key: string;
  promise: Promise<PredictResponse>;
}
let pending: PendingDetection | null = null;

const detectionKey = (file: File, dir: ReadingDirection, lay: WallLayout) =>
  `${file.name}:${file.size}:${file.lastModified}:${dir}:${lay}`;

export interface GlyphStore {
  image: UploadedImage | null;
  result: GlyphDecodingResult | null;
  /** The detection behind `result`, with any applied corrections (needed to re-read). */
  detection: PredictResponse | null;
  /** Corrections the user accepted for this image, oldest first. */
  appliedCorrections: SignCorrection[];
  isDetecting: boolean;
  isTranslating: boolean;
  error: string | null;
  direction: ReadingDirection | null;
  layout: WallLayout;
  context: TextContext;
  selectDirection: (d: ReadingDirection) => void;
  setLayout: (l: WallLayout) => void;
  setContext: (patch: Partial<TextContext>) => void;
  setImage: (file: File) => Promise<void>;
  clearImage: () => void;
  reset: () => void;
  decode: () => Promise<void>;
  /** Human in the loop: swap one sign, re-run /transliterate/, return the new result. */
  /** Swap one or more signs (at most MAX_CORRECTIONS) and re-read the whole inscription once. */
  applyCorrections: (cs: SignCorrection[]) => Promise<GlyphDecodingResult>;
}

/** Index into the flat corrected sequence for a 1-based (line, position) of the reading. */
/** Signs the user may correct in one re-read (the re-read itself stays once per hour). */
export const MAX_CORRECTIONS = 3;

const flatIndex = (result: GlyphDecodingResult, line: number, position: number): number => {
  const lines = result.readingLines;
  const target = lines.find((l) => l.line === line);
  if (!target || position < 1 || position > target.codes.length) return -1;
  const offset = lines
    .filter((l) => l.line < line)
    .reduce((sum, l) => sum + l.codes.length, 0);
  return offset + position - 1;
};

export const useGlyphStore = create<GlyphStore>()(persist((set, get) => {
  /** Fire (or reuse) the background detection for the current settings. */
  const ensureDetection = (
    file: File,
    dir: ReadingDirection,
    lay: WallLayout,
  ): Promise<PredictResponse> => {
    const key = detectionKey(file, dir, lay);
    if (pending?.key === key) return pending.promise;

    set({ isDetecting: true });
    const promise = decodeGlyphsRaw(file, {
      direction: dir,
      layout: lay,
      translate: false, // stage 1 only — LLM comes later
    }).finally(() => {
      if (pending?.key === key) set({ isDetecting: false });
    });
    pending = { key, promise };
    return promise;
  };

  return {
    image: null,
    result: null,
    detection: null,
    appliedCorrections: [],
    isDetecting: false,
    isTranslating: false,
    error: null,
    direction: null, // null until the user picks — that pick is the trigger
    layout: "rows",
    context: {},

    selectDirection: (d) => {
      set({ direction: d, error: null });
      const { image, layout } = get();
      if (image) {
        ensureDetection(image.file, d, layout).catch(() => {
          /* surfaced on decode(); avoid unhandled rejection here */
        });
      }
    },

    setLayout: (l) => set({ layout: l }),

    setContext: (patch) =>
      set((s) => ({ context: { ...s.context, ...patch } })),

    setImage: async (file) => {
      const preview = await readAsDataUrl(file);
      pending = null; // stale detection belongs to old image
      set({
        image: { file, preview },
        result: null,
        detection: null,
        appliedCorrections: [],
        error: null,
        direction: null, // force a fresh direction pick = trigger
      });
    },

    clearImage: () => {
      pending = null;
      set({ image: null, result: null, detection: null, appliedCorrections: [], error: null, direction: null });
    },

    reset: () => {
      pending = null;
      set({
        image: null,
        result: null,
        detection: null,
        appliedCorrections: [],
        error: null,
        isDetecting: false,
        isTranslating: false,
        direction: null,
        layout: "rows",
        context: {},
      });
    },

    decode: async () => {
      const { image, direction, layout, context } = get();
      if (!image || !direction) return;
      set({ error: null, isTranslating: true });
      try {
        // Usually already resolved — the user spent ~10 s on the form.
        const detection = await ensureDetection(image.file, direction, layout);
        const translit = await transliterateGlyphs(detection, context, image.preview);
        const decoded = mergeDecodingResult(detection, translit, image.preview);
        rememberSession(decoded, context);
        if (!translit.error) recordReading(decoded, false);
        set({
          result: decoded,
          detection,
          appliedCorrections: [], // a fresh decode starts from the detector again
          error: translit.error ?? null,
        });
      } catch (err) {
        set({
          error: err instanceof Error ? err.message : "Decoding failed.",
        });
      } finally {
        set({ isTranslating: false });
      }
    },

    applyCorrections: async (cs) => {
      const { image, result, detection, context, appliedCorrections, isTranslating } = get();
      if (!image && result) {
        // Restored from memory after a reload: the photo is not stored, so it must come back first.
        throw new Error("Upload the photo again to re-read it with your corrections.");
      }
      if (!image || !result || !detection) throw new Error("There is no reading to correct.");
      if (isTranslating) throw new Error("A reading is already in progress.");
      if (!cs.length || cs.length > MAX_CORRECTIONS) {
        throw new Error(`Correct between 1 and ${MAX_CORRECTIONS} signs per re-read.`);
      }

      // Copy, swap each code, and mark each slot as confirmed by the user (confidence 1.0).
      const next: PredictResponse = structuredClone(detection);
      const seen = new Set<number>();
      for (const c of cs) {
        const i = flatIndex(result, c.line, c.position);
        if (i < 0 || detection.outer.correction.flat_corrected_seq[i] !== c.from_code) {
          throw new Error(`Line ${c.line}, sign ${c.position} is no longer ${c.from_code}.`);
        }
        if (seen.has(i)) throw new Error(`Line ${c.line}, sign ${c.position} is corrected twice.`);
        seen.add(i);
        next.outer.correction.flat_corrected_seq[i] = c.to_code;
        next.outer.slots[i] = [[c.to_code, 1.0], ...(next.outer.slots[i] ?? []).slice(1)];
      }
      const corrections = [...appliedCorrections, ...cs];

      set({ isTranslating: true, error: null });
      try {
        const translit = await transliterateGlyphs(next, context, image.preview, corrections, result.sessionId);
        if (translit.error) throw new Error(translit.error);
        const updated = mergeDecodingResult(next, translit, image.preview);
        rememberSession(updated, context);
        recordReading(updated, true);
        set({ result: updated, detection: next, appliedCorrections: corrections });
        return updated;
      } finally {
        set({ isTranslating: false });
      }
    },
  };
}, {
  // Agent memory: the last reading survives a reload. Images stay out (too big for localStorage):
  // the photo must be uploaded again to re-read, but the reading, the wall and the chat come back.
  name: "sphinx.glyph",
  version: 1,
  storage: memoryStorage,
  partialize: (s) => ({
    result: s.result ? { ...s.result, originalImage: "", annotatedImage: null } : null,
    detection: s.detection ? { ...s.detection, annotated_image: null } : null,
    appliedCorrections: s.appliedCorrections,
    layout: s.layout,
    context: s.context,
  }),
}));
