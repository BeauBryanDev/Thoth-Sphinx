// ============================================================
// <SemanticTranslation /> — the second translation, under the raw one
// ============================================================
// The LLM's English is searched against published translations (Allen's
// Pyramid Texts, Faulkner's Coffin Texts, temple plates from Karnak and
// Medinet Habu, museum catalogue inscriptions), then smoothed against the
// closest passages (app/services/refine_translation.py).
// The raw LLM translation above stays untouched; this card shows the
// refined text, how the passages were used, and the passages themselves.

import { useState } from "react";
import type { SemanticTranslation as Semantic, TranslationPassage } from "@/types";

const MATCH_LABEL: Record<Semantic["match"], { text: string; tone: string }> = {
  same_text: { text: "same text", tone: "border-emerald-500/60 bg-emerald-950/40 text-emerald-200" },
  related: { text: "related text", tone: "border-amber-500/60 bg-amber-950/40 text-amber-200" },
  none: { text: "no match", tone: "border-stone-500/60 bg-stone-900/60 text-stone-300" },
};

const Similarity = ({ value }: { value: number | null }) => {
  if (value === null) return <span className="font-mono text-xs text-amber-300/50">—</span>;
  const v = Math.max(0, Math.min(1, value));
  return (
    <span className="flex items-center gap-1.5" title={`cosine similarity ${v.toFixed(3)}`}>
      <span className="font-mono text-xs font-bold text-amber-300">{v.toFixed(2)}</span>
      <span className="h-1.5 w-10 overflow-hidden rounded-full bg-stone-800">
        <span className="block h-full bg-amber-400" style={{ width: `${v * 100}%` }} />
      </span>
    </span>
  );
};

// Roman numerals stay upper case: 'RAMESSES II' -> 'Ramesses II', not 'Ramesses Ii'.
const titleCase = (s: string) =>
  s
    .toLowerCase()
    .replace(/(^|[\s,–-])\p{L}/gu, (m) => m.toUpperCase())
    .replace(/\b[IVX][ivx]+\b/g, (m) => m.toUpperCase());

const WHOLE_LABEL: Record<string, string> = {
  Spell: "Whole spell",
  Utterance: "Whole utterance",
  Plate: "Whole plate",
  "Catalog No.": "Whole catalogue entry",
};

const PassageRow = ({ passage, rank, used }: { passage: TranslationPassage; rank: number; used: boolean }) => {
  const [open, setOpen] = useState(false);
  const where = [passage.king, passage.location].filter(Boolean).map((x) => titleCase(x as string)).join(" · ");
  return (
    <li>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={passage.citation}
        className={`w-full rounded border px-3 py-2 text-left transition-colors hover:bg-amber-900/50 ${
          used ? "border-emerald-500/60 bg-emerald-950/20" : "border-amber-700/50 bg-amber-950/40"
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <span className="text-xs font-bold text-amber-500">#{rank}</span>
            {passage.rerank_score != null && (
              <span
                className="rounded border border-sky-500/50 bg-sky-950/40 px-1.5 py-px font-mono text-[10px] text-sky-200"
                title={`Cohere rerank relevance; ranked #${passage.hybrid_rank ?? "?"} by hybrid search before reranking`}
              >
                rerank {passage.rerank_score.toFixed(2)}
                {passage.hybrid_rank != null && passage.hybrid_rank !== rank && ` · was #${passage.hybrid_rank}`}
              </span>
            )}
            <Similarity value={passage.similarity} />
            {used && (
              <span className="rounded-full border border-emerald-500/60 px-1.5 py-px text-[10px] uppercase tracking-wide text-emerald-200">
                used
              </span>
            )}
          </span>
          <span className="shrink-0 text-xs text-amber-200/80">
            {passage.spell !== null && `${passage.unit ?? "Spell"} ${passage.spell}`}
            {passage.page && ` · p. ${passage.page}`}
          </span>
        </div>
        <p className="mt-1 text-sm font-semibold text-amber-200">
          {passage.author}, <span className="italic">{passage.work}</span>
        </p>
        {(where || passage.title) && (
          <p className="text-xs text-amber-300/80">
            {where}
            {where && passage.title && " — "}
            {passage.title && titleCase(passage.title)}
          </p>
        )}
        <p className="mt-0.5 flex flex-wrap gap-x-3 font-mono text-[10px] text-amber-300/50">
          <span>{passage.passage_id}</span>
          <span>matched by {passage.matched_by.join(" + ")}</span>
        </p>

        {open && (
          <div className="mt-2 border-t border-amber-800/50 pt-2">
            <p className="text-[10px] uppercase tracking-wide text-amber-500/70">
              {passage.spell_text
                ? `${WHOLE_LABEL[passage.unit ?? "Spell"] ?? "Whole text"} (what the editor read)`
                : "Matching part"}
            </p>
            <p className="mt-1 whitespace-pre-line font-serif text-sm italic leading-relaxed text-stone-200/90">
              {passage.spell_text || passage.text}
            </p>
          </div>
        )}
      </button>
    </li>
  );
};

export const SemanticTranslation = ({ semantic }: { semantic: Semantic | null }) => {
  if (!semantic) return null;
  const match = MATCH_LABEL[semantic.match];

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
        <p className="text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
          𓂋 Semantic translation · published parallels
        </p>
        {!semantic.error && (
          <span className={`rounded-full border px-2 py-px text-[10px] uppercase tracking-wide ${match.tone}`}>
            {match.text}
          </span>
        )}
      </div>

      {semantic.error ? (
        <p className="text-sm italic text-red-200/80">The second translation could not be made: {semantic.error}</p>
      ) : (
        <>
          <p className="font-serif text-base italic leading-relaxed text-amber-100/90">“{semantic.translation}”</p>
          {semantic.note && <p className="mt-2 text-sm leading-relaxed text-amber-100/70">{semantic.note}</p>}
        </>
      )}

      <div className="mt-3">
        <p className="mb-1.5 flex flex-wrap gap-x-3 text-[10px] uppercase tracking-wide text-amber-500/70">
          <span>{semantic.passages.length} closest passages</span>
          <span>search: {semantic.search_mode}</span>
          {semantic.king_filter && <span>king: {titleCase(semantic.king_filter)}</span>}
          {semantic.latency_ms !== null && <span>{(semantic.latency_ms / 1000).toFixed(1)} s</span>}
          {semantic.cost_usd !== null && <span>${semantic.cost_usd.toFixed(4)}</span>}
        </p>
        {semantic.passages.length > 0 ? (
          <ol className="space-y-2">
            {semantic.passages.map((p, i) => (
              <PassageRow key={p.passage_id} passage={p} rank={i + 1} used={p.passage_id === semantic.used_passage} />
            ))}
          </ol>
        ) : (
          <p className="text-sm italic text-amber-300/60">No published passage was found.</p>
        )}
      </div>
    </div>
  );
};
