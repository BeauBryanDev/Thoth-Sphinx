// ============================================================
// <ReversePage /> — retro-translation workspace
//
// English -> Middle Egyptian. The user writes a short modern
// phrase (INPUT, left); GPT-4o composes genuine Middle Egyptian
// (grammar -> Leiden transliteration -> Gardiner codes) and the
// OUTPUT (right) renders the signs with the existing Unicode
// hieroglyph map — plain golden rows, no per-sign boxes.
// ============================================================

import { Card, Button } from "@/components/common";
import { useReverseStore, MAX_REVERSE_CHARS } from "@/stores";
import { toHieroglyph } from "@/utils/gardinerUnicode";
import type { ReverseRegister } from "@/types";
import horusIcon from "@/assets/horus.svg";

/** Horus at one side of a card title; the right one is mirrored so the pair faces the title. */
const Horus = ({ mirrored = false }: { mirrored?: boolean }) => (
  <img
    src={horusIcon}
    alt=""
    aria-hidden="true"
    className={`mx-auto h-9 w-auto select-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.6)] sm:h-12 ${mirrored ? "-scale-x-100" : ""}`}
    draggable={false}
  />
);

const REGISTERS: { value: ReverseRegister; label: string }[] = [
  { value: "unknown", label: "Unknown — let the scribe decide" },
  { value: "monumental", label: "Monumental (temple / stela)" },
  { value: "literary", label: "Literary (Middle-Kingdom tale)" },
  { value: "letter", label: "Letter (epistolary)" },
  { value: "religious", label: "Religious (hymn / offering)" },
];

const MAX_CHARS = MAX_REVERSE_CHARS;

/** Golden glyph run — falls back to the Gardiner code for the rare
    sign outside the Unicode map's coverage. */
const GlyphRun = ({ codes }: { codes: string[] }) => (
  <div className="flex flex-wrap items-center gap-x-1 gap-y-2">
    {codes.map((code, i) => {
      const glyph = toHieroglyph(code);
      return glyph ? (
        <span
          key={i}
          title={code}
          className="font-hieroglyph glyph-gold text-4xl"
        >
          {glyph}
        </span>
      ) : (
        <span
          key={i}
          title="No glyph in font map"
          className="rounded border border-amber-800/50 px-1 py-0.5 font-mono text-[10px] text-amber-500/80"
        >
          {code}
        </span>
      );
    })}
  </div>
);

export const ReversePage = () => {
  // Global Zustand store — draft + result survive page navigation.
  const {
    text,
    register,
    result,
    isLoading,
    error,
    setText,
    setRegister,
    translate: onTranslate,
  } = useReverseStore();

  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
      {/* INPUT — modern English */}
      <Card
        title="Input · English"
        subtitle="Write a short phrase to render into Middle Egyptian"
        icon="𓍯"
        headerBalance
        headerStart={<Horus />}
        headerEnd={<Horus mirrored />}
      >
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
          rows={5}
          className="w-full rounded-md border border-amber-700/60 bg-amber-950/60 p-3 text-base text-amber-100 placeholder:text-amber-600/50 focus:border-amber-400 focus:outline-none"
          placeholder="e.g. The scribe loves beer, given life forever"
        />
        <p className="mt-1 text-right font-mono text-[10px] text-amber-600/70">
          {text.length}/{MAX_CHARS}
        </p>

        <label className="mt-3 block">
          <span className="mb-1 block text-[10px] uppercase tracking-[0.2em] text-amber-400/80">
            Register · style of the composition
          </span>
          <select
            value={register}
            onChange={(e) => setRegister(e.target.value as ReverseRegister)}
            className="w-full rounded border border-amber-700/50 bg-stone-950 px-2 py-1.5 text-xs text-amber-100"
          >
            {REGISTERS.map((r) => (
              <option key={r.value} value={r.value}>
                {r.label}
              </option>
            ))}
          </select>
        </label>

        <p className="mt-3 rounded border border-amber-600/40 bg-amber-950/50 px-3 py-2 text-[11px] leading-relaxed text-amber-300/90">
          ⚠ Keep it to a few short sentences. Middle Egyptian has no words
          for modern concepts — the scribe will substitute the closest
          ancient notion (and tell you when it does). Long or very modern
          text degrades the composition.
        </p>

        <Button
          className="mt-4 w-full"
          size="lg"
          onClick={onTranslate}
          isLoading={isLoading}
          disabled={!text.trim()}
          leftIcon="𓆓"
        >
          Translate to Middle Egyptian
        </Button>

        {error && (
          <p className="mt-3 rounded border border-red-500/50 bg-red-900/40 px-3 py-2 text-xs text-red-200">
            {error}
          </p>
        )}
      </Card>

      {/* OUTPUT — Middle Egyptian */}
      <Card
        title="Output · Middle Egyptian"
        subtitle="How a scribe would carve your words"
        icon="𓊹"
        headerBalance
        headerStart={<Horus />}
        headerEnd={<Horus mirrored />}
      >
        {result ? (
          <div className="space-y-4">
            {/* Hieroglyphs — plain golden rows, no boxes */}
            <div className="rounded-md border border-amber-700/50 bg-gradient-to-b from-stone-900/80 to-stone-950/80 p-4 shadow-inner">
              <GlyphRun codes={result.gardiner_codes} />
            </div>

            <div>
              <p className="mb-1 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                Transliteration
              </p>
              <p className="font-serif text-lg italic text-amber-200">
                {result.transliteration}
              </p>
            </div>

            {result.normalized_english &&
              result.normalized_english.toLowerCase() !==
                result.source_text.toLowerCase() && (
                <div>
                  <p className="mb-1 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                    What was actually translated
                  </p>
                  <p className="text-sm text-amber-100/90">
                    “{result.normalized_english}”
                  </p>
                </div>
              )}

            {result.words.length > 0 && (
              <div>
                <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                  Word by word
                </p>
                <ul className="divide-y divide-amber-800/40 rounded-md border border-amber-800/40 bg-stone-950/50">
                  {result.words.map((w, i) => (
                    <li key={i} className="flex items-center gap-3 px-3 py-2">
                      <span className="shrink-0">
                        <GlyphRun codes={w.gardiner_codes} />
                      </span>
                      <span className="ml-auto text-right">
                        <span className="block font-serif text-sm italic text-amber-200">
                          {w.transliteration}
                        </span>
                        <span className="block text-xs text-amber-400/80">
                          {w.english}
                          {w.note && (
                            <span className="text-amber-600/80"> · {w.note}</span>
                          )}
                        </span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {result.grammar_notes && (
              <div>
                <p className="mb-1 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
                  Scribe's notes
                </p>
                <p className="text-xs leading-relaxed text-amber-300/80">
                  {result.grammar_notes}
                </p>
              </div>
            )}

            <p className="text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
              Confidence {result.confidence} · {result.model}
            </p>
          </div>
        ) : (
          <div className="flex min-h-[220px] flex-col items-center justify-center gap-3 rounded-md border border-amber-700/40 bg-stone-900/50 p-6 text-center">
            <span className="font-hieroglyph glyph-gold text-4xl">
              𓋹𓍑𓋴
            </span>
            <p className="text-sm text-amber-500/80">
              {isLoading
                ? "The scribe is composing…"
                : "Your phrase in hieroglyphs will appear here."}
            </p>
          </div>
        )}
      </Card>
    </div>
  );
};

export default ReversePage;
