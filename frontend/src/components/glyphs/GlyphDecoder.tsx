// ============================================================
// <GlyphDecoder /> — image upload + transliteration result
// ============================================================

import {
  useCallback,
  useRef,
  useState,
  type DragEvent,
  type ChangeEvent,
} from "react";
import { Card, Button } from "@/components/common";
import {
  useGlyphDecoder,
  type UseGlyphDecoderReturn,
} from "@/hooks/useGlyphDecoder";
import { formatConfidence } from "@/utils/formatters";
import { SignFrequencyChart } from "./SignFrequencyChart";
import { ReadingProvenanceCard } from "./ReadingProvenanceCard";
import { CorpusEvidenceCard } from "./CorpusEvidenceCard";
import { AgentTrace } from "./AgentTrace";
import { Entities } from "./Entities";
import { SemanticTranslation } from "./SemanticTranslation";
import { ReadingChatDrawer } from "@/components/chat/ReadingChatDrawer";
import { useReadingChatStore } from "@/stores";
import { HintButton, HintModal, LayoutHint, DirectionHint } from "./DecoderHints";
import queenIcon from "@/assets/qeen.svg";
import pharaohIcon from "@/assets/pharaoh.svg";

// Controlled vocabularies — must match app/schemas/transliterations.py.
// "unknown" first = default, for the tourist who knows nothing.
const PERIODS = [
  "unknown", "old_kingdom", "first_intermediate", "middle_kingdom",
  "second_intermediate", "new_kingdom", "third_intermediate",
  "late_period", "ptolemaic", "roman",
];
const TEXT_TYPES = [
  "unknown", "stela", "temple_wall", "tomb_wall", "papyrus",
  "sarcophagus", "obelisk", "statue", "offering_table", "pyramid_texts",
];
const SUPPORTS = [
  "unknown", "limestone", "sandstone", "granite", "papyrus",
  "wood", "plaster", "faience", "metal",
];
const LOCATION_TYPES = [
  "unknown", "pyramid", "temple", "tomb", "museum", "open_site",
];

const label = (v: string) =>
  v === "unknown" ? "Unknown" : v.replace(/_/g, " ");

interface GlyphDecoderProps {
  /** Share decoder state with sibling panels (e.g. <GlyphWall />). */
  controller?: UseGlyphDecoderReturn;
}

export const GlyphDecoder = ({ controller }: GlyphDecoderProps = {}) => {
  // Hooks must run unconditionally; the internal instance is simply
  // unused when the page supplies a shared controller.
  const internal = useGlyphDecoder();
  const {
    image,
    result,
    error,
    direction,
    layout,
    context,
    isDetecting,
    isTranslating,
    selectDirection,
    setLayout,
    setContext,
    setImage,
    clearImage,
    reset,
    decode,
  } = controller ?? internal;
  const openReadingChat = useReadingChatStore((s) => s.openFor);
  const inputRef = useRef<HTMLInputElement>(null);
  const dragActiveRef = useRef(false);
  const [hint, setHint] = useState<"layout" | "direction" | null>(null);
  const closeHint = useCallback(() => setHint(null), []);

  const onFile = useCallback(
    (file: File | undefined | null) => {
      if (!file) return;
      if (!file.type.startsWith("image/")) return;
      setImage(file);
    },
    [setImage],
  );

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragActiveRef.current = false;
    onFile(e.dataTransfer.files?.[0]);
  };

  const onDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    dragActiveRef.current = true;
  };

  const onDragLeave = () => {
    dragActiveRef.current = false;
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    onFile(e.target.files?.[0]);
  };

  return (
    <Card
      title="Glyph Decoder"
      subtitle="Upload an image to transliterate hieroglyphs"
      icon="𓂀"
      className="flex h-full flex-col"
      headerStart={
        <HintButton
          icon={queenIcon}
          caption="Layout"
          label="How to choose the layout (rows or columns)"
          onClick={() => setHint("layout")}
        />
      }
      headerEnd={
        <HintButton
          icon={pharaohIcon}
          caption="Direction"
          label="How to choose the reading direction"
          onClick={() => setHint("direction")}
        />
      }
    >
      <HintModal
        open={hint === "layout"}
        onClose={closeHint}
        icon={queenIcon}
        title="Image layout"
      >
        <LayoutHint
          layout={layout}
          onPick={(l) => {
            setLayout(l);
            closeHint();
          }}
        />
      </HintModal>
      <HintModal
        open={hint === "direction"}
        onClose={closeHint}
        icon={pharaohIcon}
        title="Reading direction"
      >
        <DirectionHint
          direction={direction}
          onPick={(d) => {
            selectDirection(d);
            closeHint();
          }}
        />
      </HintModal>

      {/* Dropzone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        role="button"
        tabIndex={0}
        className="flex cursor-pointer flex-col items-center justify-center rounded-md border-2 border-dashed border-amber-700/60 bg-amber-950/40 px-6 py-10 text-center transition-colors hover:border-amber-500/70 hover:bg-amber-900/40"
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={onChange}
        />

        {image ? (
          <div className="flex w-full flex-col items-center gap-3">
            <div className="relative overflow-hidden rounded">
              <img
                src={image.preview}
                alt="Uploaded hieroglyph preview"
                className="max-h-48 rounded border border-amber-700/50 object-contain shadow"
              />
              {isTranslating && (
                <div className="scan-overlay pointer-events-none absolute inset-0" />
              )}
            </div>
            <p className="text-xs text-amber-300/80">{image.file.name}</p>
            <button
              onClick={(e) => {
                e.stopPropagation();
                clearImage();
              }}
              className="text-xs text-amber-400 underline-offset-2 hover:underline"
            >
              Clear image
            </button>
          </div>
        ) : (
          <>
            <div className="mb-2 text-4xl text-amber-500/80">🖼</div>
            <p className="text-sm font-semibold tracking-wide text-amber-200">
              DRAG &amp; DROP AN IMAGE HERE
            </p>
            <p className="mt-1 text-xs text-amber-400/70">
              or click to browse
            </p>
          </>
        )}
      </div>

      {/* Reading options — layout is required by the backend */}
      <div className="mt-4 grid grid-cols-2 gap-3">
        <div>
          <p className="mb-1 text-[10px] uppercase tracking-[0.2em] text-amber-400/80">
            Layout
          </p>
          <div className="flex overflow-hidden rounded-md border border-amber-700/60">
            {(["rows", "columns"] as const).map((l) => (
              <button
                key={l}
                onClick={() => setLayout(l)}
                className={`flex-1 px-2 py-1.5 text-xs font-semibold uppercase tracking-wide transition-colors ${
                  layout === l
                    ? "bg-amber-600 text-stone-950"
                    : "bg-stone-900/60 text-amber-300/70 hover:bg-amber-900/40"
                }`}
              >
                {l}
              </button>
            ))}
          </div>
        </div>
        <div>
          <p className="mb-1 text-[10px] uppercase tracking-[0.2em] text-amber-400/80">
            Direction {isDetecting && (
              <span className="ml-1 animate-pulse normal-case tracking-normal text-amber-500/90">
                — reading the wall…
              </span>
            )}
          </p>
          <div className="flex overflow-hidden rounded-md border border-amber-700/60">
            {(["rtl", "ltr"] as const).map((d) => (
              <button
                key={d}
                onClick={() => selectDirection(d)}
                className={`flex-1 px-2 py-1.5 text-xs font-semibold uppercase tracking-wide transition-colors ${
                  direction === d
                    ? "bg-amber-600 text-stone-950"
                    : "bg-stone-900/60 text-amber-300/70 hover:bg-amber-900/40"
                }`}
              >
                {d === "rtl" ? "Right → Left" : "Left → Right"}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Archaeological context — MANDATORY. It feeds the LLM stage;
          a naive tourist just leaves every select on "Unknown". */}
      <div className="mt-4">
        <div className="flex w-full items-center justify-between rounded-t-md border border-b-0 border-amber-700/50 bg-stone-900/60 px-3 py-2 text-xs font-semibold uppercase tracking-[0.15em] text-amber-300/90">
          <span>𓊹 Archaeological context</span>
          <span className="rounded bg-amber-600/20 px-2 py-0.5 text-[9px] tracking-widest text-amber-400">
            required
          </span>
        </div>

        <div className="grid grid-cols-1 gap-3 rounded-b-md border border-amber-700/40 bg-stone-900/40 p-3 sm:grid-cols-2">
            {(
              [
                ["Period", "period", PERIODS],
                ["Text type", "text_type", TEXT_TYPES],
                ["Physical support", "support", SUPPORTS],
                ["Location type", "location_type", LOCATION_TYPES],
              ] as const
            ).map(([title, key, options]) => (
              <label key={key} className="block">
                <span className="mb-1 block text-[10px] uppercase tracking-[0.2em] text-amber-400/80">
                  {title}
                </span>
                <select
                  value={context[key] ?? "unknown"}
                  onChange={(e) => setContext({ [key]: e.target.value })}
                  className="w-full rounded border border-amber-700/50 bg-stone-950 px-2 py-1.5 text-xs capitalize text-amber-100"
                >
                  {options.map((opt) => (
                    <option key={opt} value={opt}>
                      {label(opt)}
                    </option>
                  ))}
                </select>
              </label>
            ))}
            {(
              [
                ["Site / location", "site", "e.g. Karnak, Saqqara"],
                ["Dynasty", "dynasty", "e.g. IV, XVIII"],
                ["King's reign", "kings_reign", "e.g. Thutmose III"],
              ] as const
            ).map(([title, key, placeholder]) => (
              <label key={key} className="block">
                <span className="mb-1 block text-[10px] uppercase tracking-[0.2em] text-amber-400/80">
                  {title}
                </span>
                <input
                  type="text"
                  value={context[key] ?? ""}
                  placeholder={placeholder}
                  onChange={(e) => setContext({ [key]: e.target.value })}
                  className="w-full rounded border border-amber-700/50 bg-stone-950 px-2 py-1.5 text-xs text-amber-100 placeholder:text-amber-600/50"
                />
              </label>
            ))}
            <p className="text-[10px] leading-relaxed text-amber-500/70 sm:col-span-2">
              Don't know something? Just leave it as Unknown — the reading
              still works. Every detail you add improves the translation.
            </p>
        </div>
      </div>

      {/* Result block */}
      <div className="mt-5 rounded-md border border-amber-700/50 bg-stone-900/60 p-5 text-center">
        <p className="mb-3 text-xs uppercase tracking-[0.3em] text-amber-400/80">
          — Transliteration Result —
        </p>
        {result ? (
          <div className="text-left">
            {/* 1. Gardiner codes in spatial reading order, one block per
                   physical line — like the bounding-box coordinate view. */}
            <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
              Gardiner codes · spatial order ·{" "}
              {result.layout === "columns" ? "columns" : "rows"} ·{" "}
              {result.direction === "rtl" ? "right → left" : "left → right"}
            </p>
            <div className="space-y-1.5 rounded border border-amber-700/40 bg-stone-950/60 p-3">
              {result.lines.map((line, i) => (
                <div key={i} className="flex items-baseline gap-2">
                  <span className="w-14 shrink-0 font-mono text-[10px] uppercase tracking-wider text-amber-600/80">
                    {result.layout === "columns" ? "col" : "row"} {i + 1}
                  </span>
                  <div className="flex flex-wrap gap-1">
                    {line.map((code, j) => (
                      <span
                        key={j}
                        className={`rounded px-2 py-0.5 font-mono text-sm ${
                          code === "Unknown"
                            ? "border border-stone-600/60 bg-stone-800/60 text-stone-400"
                            : "border border-amber-700/50 bg-amber-950/60 text-amber-200"
                        }`}
                        title={code === "Unknown" ? "Undetected sign (lacuna)" : code}
                      >
                        {code === "Unknown" ? "?" : code}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            {/* 2. Cartouches — lexicon-verified royal names. */}
            {result.royalNames.length > 0 && (
              <div className="mt-3 rounded border border-amber-600/50 bg-amber-950/40 p-3">
                <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                  𓍷 Royal cartouches
                </p>
                <div className="flex flex-wrap gap-2">
                  {result.royalNames.map((name, i) => (
                    <span
                      key={i}
                      className="rounded-full border border-amber-500/60 bg-amber-900/50 px-3 py-1 text-xs font-semibold text-amber-100"
                    >
                      {name}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* 3. Transliteration. */}
            <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
              <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                Transliteration
              </p>
              <p className="font-serif text-base italic leading-relaxed text-amber-200">
                {result.transliteration}
              </p>
            </div>

            {/* 4. English gloss — below everything; scrolling is fine. */}
            <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
              <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                English translation · gloss
              </p>
              {/* The translation is always italic; Claude's notes below stay roman, so the
                  reader can tell the ancient text from the commentary. */}
              <p className="font-serif text-base italic leading-relaxed text-amber-100/90">
                “{result.translation}”
              </p>
            </div>

            {/* 4b. Second translation: the raw one above, smoothed against published passages. */}
            <SemanticTranslation semantic={result.semantic} />

            {/* 4·. The kings of the matched cartouches (knowledge graph). */}
            <Entities entities={result.entities} />

            {/* 4a. Claude's notes on the photograph. */}
            {result.imageNotes.length > 0 && (
              <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
                <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                  𓏞 Notes from Claude
                </p>
                <ul className="space-y-2">
                  {result.imageNotes.map((note, i) => (
                    <li
                      key={i}
                      className="flex gap-2 text-sm leading-relaxed text-amber-100/90"
                    >
                      <span aria-hidden className="mt-0.5 text-amber-500/80">𓋹</span>
                      <span>{note}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {/* 4b. Who read it: first reader, router decision, second opinion. */}
            {result.reading && <ReadingProvenanceCard reading={result.reading} />}

            {/* 4c. What the corpus said, line by line (TLA). */}
            {result.evidence.length > 0 && <CorpusEvidenceCard evidence={result.evidence} />}

            {/* 4d. The agent's full trace: turns, tool calls, router, escalation. */}
            {result.trace.length > 0 && <AgentTrace trace={result.trace} />}

            {/* 5. YOLO-annotated detection image + download. */}
            {result.annotatedImage && (
              <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
                <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                  Detections on your image
                </p>
                <img
                  src={result.annotatedImage}
                  alt="Detected hieroglyphs with bounding boxes and confidence labels"
                  className="w-full rounded border border-amber-800/50 shadow"
                />
                <a
                  href={result.annotatedImage}
                  download="sphinxeyes_detections.jpg"
                  className="mt-2 flex items-center justify-center gap-2 rounded-md border border-amber-600/60 bg-amber-900/40 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-amber-200 transition-colors hover:bg-amber-800/50"
                >
                  ⤓ Download annotated image
                </a>
              </div>
            )}

            {/* 6. Sign-frequency bar chart — basic detection stats. */}
            <SignFrequencyChart result={result} />

            {/* 6b. Talk about this reading: opens the chat drawer with the reading as context. */}
            <Button
              type="button"
              size="lg"
              onClick={() => openReadingChat(result)}
              leftIcon={<span aria-hidden>𓂀</span>}
              className="mt-4 w-full rounded-xl border-2 text-sm font-bold uppercase tracking-[0.2em]"
            >
              Ask Thot-Sphinx about this reading
            </Button>
            <ReadingChatDrawer />

            {/* 7. Cross-check invitation. Lives here rather than under the
                sign-confidence table, which grows unbounded on dense walls and
                buries it. The DETECTION is what these sources verify sign by
                sign; the transliteration and gloss are an LLM reading and
                deliberately out of scope. */}
            <div className="mt-6 rounded-xl border-2 border-amber-700/60 bg-stone-950/50 p-4">
              <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-[0.25em] text-amber-300/90">
                𓂀 Verify the detection
              </h4>
              <p className="mb-3 text-[11px] leading-relaxed text-amber-500/80">
                Don&apos;t take our word for it — check the{" "}
                <span className="text-amber-300/90">sign detection</span> above
                against an independent source of truth. (The transliteration and
                English translation are a language model&apos;s reading, not a
                verified one.)
              </p>
              <div className="flex flex-col gap-3 sm:flex-row">
                <a
                  href="https://hieroglyphs.ai/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex flex-1 items-center justify-center gap-2.5 rounded-xl border-2 border-amber-600/70 bg-gradient-to-b from-amber-900/60 to-stone-900/80 px-5 py-3.5 text-sm font-semibold uppercase tracking-[0.18em] text-amber-200 shadow-inner transition-colors hover:border-amber-400 hover:text-amber-100 focus:outline-none focus:ring-2 focus:ring-amber-500/60"
                >
                  <span
                    className="font-hieroglyph text-2xl leading-none"
                    aria-hidden="true"
                  >
                    𓂀
                  </span>
                  Validate
                  <span className="text-xs text-amber-500/70" aria-hidden="true">
                    ↗
                  </span>
                  <span className="sr-only">
                    on Hieroglyphs.ai (opens a new tab)
                  </span>
                </a>
                <a
                  href="https://en.wikipedia.org/wiki/Gardiner%27s_sign_list"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex flex-1 items-center justify-center gap-2.5 rounded-xl border-2 border-amber-700/50 bg-stone-900/70 px-5 py-3.5 text-sm font-semibold uppercase tracking-[0.18em] text-amber-300/90 shadow-inner transition-colors hover:border-amber-500/80 hover:text-amber-100 focus:outline-none focus:ring-2 focus:ring-amber-500/60"
                >
                  <span
                    className="font-hieroglyph text-2xl leading-none"
                    aria-hidden="true"
                  >
                    𓏛
                  </span>
                  Check by Yourself
                  <span className="text-xs text-amber-500/70" aria-hidden="true">
                    ↗
                  </span>
                  <span className="sr-only">
                    on Gardiner&apos;s sign list, Wikipedia (opens a new tab)
                  </span>
                </a>
              </div>
            </div>

            <p className="mt-3 text-center text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
              Detector confidence {formatConfidence(result.confidence)}
            </p>
          </div>
        ) : (
          <>
            <div className="mb-3 flex flex-wrap items-center justify-center gap-3 text-3xl text-amber-500/60">
              <span>𓇋</span>
              <span>𓂀</span>
              <span>𓃭</span>
              <span>𓇋</span>
              <span>𓊪</span>
              <span>𓋴</span>
            </div>
            <p className="font-serif text-lg italic text-amber-500/80">
              ntr nfr iri m htp
            </p>
            <p className="mt-2 text-sm text-amber-400/70">
              “Good god, in peace”
            </p>
          </>
        )}
      </div>

      {error && (
        <p className="mt-3 rounded border border-red-500/50 bg-red-900/40 px-3 py-2 text-xs text-red-200">
          {error}
        </p>
      )}

      <div className="mt-5 flex gap-3">
        <Button
          className="flex-1"
          size="lg"
          onClick={decode}
          isLoading={isTranslating}
          disabled={!image || !direction}
          leftIcon="𓂀"
        >
          {direction ? "Decode Glyphs" : "Pick a reading direction first"}
        </Button>
        {(image || result) && (
          <button
            onClick={reset}
            disabled={isTranslating}
            title="Clear everything and start over"
            className="rounded-md border border-amber-700/60 bg-stone-900/60 px-4 text-xs font-semibold uppercase tracking-wide text-amber-300/90 transition-colors hover:bg-amber-900/40 disabled:cursor-not-allowed disabled:opacity-40"
          >
            ↺ Reset
          </button>
        )}
      </div>
    </Card>
  );
};

export default GlyphDecoder;
