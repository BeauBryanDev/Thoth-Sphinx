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
import {
  decodeGlyphsRaw,
  mergeDecodingResult,
  transliterateGlyphs,
} from "@/api/glyphApi";
import type {
  GlyphDecodingResult,
  PredictResponse,
  ReadingDirection,
  TextContext,
  UploadedImage,
  WallLayout,
} from "@/types";

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
}

export const useGlyphStore = create<GlyphStore>((set, get) => {
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
        error: null,
        direction: null, // force a fresh direction pick = trigger
      });
    },

    clearImage: () => {
      pending = null;
      set({ image: null, result: null, error: null, direction: null });
    },

    reset: () => {
      pending = null;
      set({
        image: null,
        result: null,
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
        set({
          result: mergeDecodingResult(detection, translit, image.preview),
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
  };
});
