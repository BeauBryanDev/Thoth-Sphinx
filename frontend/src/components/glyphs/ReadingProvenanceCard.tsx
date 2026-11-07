// ============================================================
// <ReadingProvenanceCard /> — who read the inscription, and why
// ============================================================
// Shows the backend router's decision (app/llm/router.py): Claude Haiku
// reads first; a Sonnet second opinion is asked for when the reading is
// hard to settle. Reason codes are a fixed vocabulary on the backend —
// keep REASON_LABELS in sync with router.REASONS.

import type { ReadingProvenance } from "@/types";

const MODEL_LABELS: Record<string, string> = {
  "claude-haiku-5-5": "Claude Haiku",
  "claude-sonnet-5-5": "Claude Sonnet",
};

const REASON_LABELS: Record<string, string> = {
  haiku_confident: "The first reading was clear, so no second opinion was needed.",
  haiku_requested_second_opinion: "The first reader asked for a second opinion.",
  most_lines_low_confidence: "Most lines were uncertain on the first reading.",
  no_corpus_support: "The corpus found no parallel for any line of the first reading.",
  unresolved_cartouche: "A royal cartouche could not be matched to a known king.",
  detector_low_confidence: "Many signs were detected with low confidence.",
  escalation_disabled: "Second opinions are switched off on this server.",
};

const modelLabel = (id: string): string => MODEL_LABELS[id] ?? id;

const formatCost = (usd: number): string =>
  usd < 0.01 ? "under 1¢" : `${(usd * 100).toFixed(1)}¢`;

interface ReadingProvenanceCardProps {
  reading: ReadingProvenance;
}

export const ReadingProvenanceCard = ({ reading }: ReadingProvenanceCardProps) => {
  const { escalated, reasons, note, costUsd } = reading;

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
        𓁟 Who read it
      </p>

      {/* Reader chain: Haiku, then Sonnet when the router escalated. */}
      <div className="flex flex-wrap items-center gap-2 text-xs font-semibold">
        <span className="rounded-full border border-amber-500/60 bg-amber-900/50 px-3 py-1 text-amber-100">
          {escalated ? "Claude Haiku · first reading" : modelLabel(reading.model)}
        </span>
        {escalated && (
          <>
            <span aria-hidden className="text-amber-500/80">→</span>
            <span className="rounded-full border border-sky-400/60 bg-sky-950/60 px-3 py-1 text-sky-100">
              {modelLabel(reading.model)} · second opinion
            </span>
          </>
        )}
      </div>

      {reasons.length > 0 && (
        <ul className="mt-2 space-y-0.5 text-[11px] leading-relaxed text-amber-200/80">
          {reasons.map((r) => (
            <li key={r}>{REASON_LABELS[r] ?? r}</li>
          ))}
        </ul>
      )}

      {/* Second opinion failed: the first reading is shown, never silently. */}
      {note && (
        <p className="mt-2 rounded border border-amber-600/50 bg-amber-950/50 px-2 py-1.5 text-[11px] leading-relaxed text-amber-300">
          ⚠ {note}
        </p>
      )}

      {costUsd != null && (
        <p className="mt-2 text-[10px] uppercase tracking-[0.2em] text-amber-600/80">
          Estimated cost of this reading: {formatCost(costUsd)}
        </p>
      )}
    </div>
  );
};
