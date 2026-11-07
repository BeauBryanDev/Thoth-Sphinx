// ============================================================
// <CorpusEvidenceCard /> — closest corpus match per line (TLA)
// ============================================================
// One row per line of the reading: the search_parallels verdict, the
// closest sentence in the Thesaurus Linguae Aegyptiae, and who searched
// (the model itself, or the pipeline's evidence check for a line the
// model skipped). A parallel shows a phrase is real Egyptian; it does not
// prove the phrase is carved on this stone.

import type { LineEvidence } from "@/types";

const VERDICT_STYLE: Record<string, string> = {
  strong: "border-emerald-500/70 bg-emerald-950/60 text-emerald-200",
  possible: "border-amber-500/70 bg-amber-900/50 text-amber-100",
  none: "border-stone-500/60 bg-stone-800/60 text-stone-300",
  too_short: "border-stone-500/60 bg-stone-800/60 text-stone-300",
  unrated: "border-stone-500/60 bg-stone-800/60 text-stone-300",
  error: "border-red-500/60 bg-red-950/50 text-red-200",
};

const VERDICT_LABEL: Record<string, string> = {
  strong: "strong parallel",
  possible: "possible parallel",
  none: "no parallel",
  too_short: "too short to rate",
  unrated: "signs only",
  error: "corpus unavailable",
};

// Only these verdicts mean the corpus supports the line; any other hit is a weak, unrelated match.
const SUPPORTING = new Set(["strong", "possible"]);

const searcherLabel = (id: string): string =>
  id === "evidence_check"
    ? "evidence check"
    : id.includes("sonnet")
      ? "searched by Claude Sonnet"
      : id.includes("haiku")
        ? "searched by Claude Haiku"
        : id;

interface CorpusEvidenceCardProps {
  evidence: LineEvidence[];
}

export const CorpusEvidenceCard = ({ evidence }: CorpusEvidenceCardProps) => {
  const supported = evidence.filter((e) => e.verdict === "strong" || e.verdict === "possible").length;

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
          𓏛 Corpus evidence · TLA
        </p>
        <p className="text-[10px] uppercase tracking-[0.2em] text-amber-600/80">
          {supported}/{evidence.length} lines with a parallel · 104,285 sentences searched
        </p>
      </div>

      <ol className="space-y-2">
        {evidence.map((e) => (
          <li key={e.line} className="rounded border border-amber-900/50 bg-stone-900/50 p-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-[11px] font-semibold text-amber-300/90">Line {e.line}</span>
              <span
                className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide ${
                  VERDICT_STYLE[e.verdict] ?? VERDICT_STYLE.unrated
                }`}
              >
                {VERDICT_LABEL[e.verdict] ?? e.verdict}
              </span>
              <span className="ml-auto text-[10px] text-amber-600/80">{searcherLabel(e.searched_by)}</span>
            </div>

            {e.query_translit && (
              <p className="mt-1 text-[11px] text-amber-500/80">
                query: <span className="font-serif italic text-amber-200/90">{e.query_translit}</span>
              </p>
            )}

            {e.best && !SUPPORTING.has(e.verdict) && (
              <p className="mt-1 text-[11px] text-stone-400">
                No parallel. Closest weak match, shown for reference only:
              </p>
            )}
            {e.best ? (
              <div
                className={`mt-1.5 border-l-2 pl-2 ${
                  SUPPORTING.has(e.verdict) ? "border-amber-700/60" : "border-stone-600/60 opacity-60"
                }`}
              >
                <p className="font-serif text-sm italic leading-relaxed text-amber-100">
                  {e.best.transliteration}
                </p>
                {e.best.translation && (
                  <p className="text-xs leading-relaxed text-amber-100/80">
                    “{e.best.translation}”
                    {e.best.translation_lang === "de" && (
                      <span className="ml-1 text-[10px] uppercase text-amber-600/80">(German)</span>
                    )}
                  </p>
                )}
                <p className="mt-0.5 text-[10px] text-amber-600/90">
                  {[e.best.date ?? "undated", e.best.text_name, `match: ${e.best.rating}`, e.best.sentence_id]
                    .filter(Boolean)
                    .join(" · ")}
                </p>
              </div>
            ) : (
              e.verdict === "none" && (
                <p className="mt-1 text-[11px] text-stone-400">
                  No parallel found. That is the usual outcome: most Egyptian sentences have none.
                </p>
              )
            )}
          </li>
        ))}
      </ol>

      <p className="mt-2 text-[10px] leading-relaxed text-amber-600/80">
        A parallel shows a phrase is attested Egyptian, not that it is carved on this stone.
        Corpus: Thesaurus Linguae Aegyptiae, BBAW · CC BY-SA 4.0.
      </p>
    </div>
  );
};
