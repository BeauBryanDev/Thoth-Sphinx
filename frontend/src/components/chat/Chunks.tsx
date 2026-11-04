// ============================================================
// <Chunks /> — the book passages behind Thot-Sphinx's last reply
// ============================================================
// Top 5 passages search_books retrieved, most relevant first (the
// hybrid ranking). Each row: similarity, chunk id, book, page.
// Click a row to unfold the passage text.

import { useState } from "react";
import type { BookPassage } from "@/types";

interface ChunksProps {
  passages: BookPassage[];
  /** The reply is still being written: the scrolls are being consulted. */
  loading?: boolean;
}

const Similarity = ({ value }: { value: number | null }) => {
  if (value === null) {
    return <span className="font-mono text-xs text-amber-300/50" title="meaning search unavailable">—</span>;
  }
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

const ChunkRow = ({ passage, rank }: { passage: BookPassage; rank: number }) => {
  const [open, setOpen] = useState(false);
  return (
    <li>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        title={passage.citation}
        className="w-full rounded border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-left transition-colors hover:bg-amber-900/50"
      >
        <div className="flex items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <span className="text-xs font-bold text-amber-500">#{rank}</span>
            <Similarity value={passage.similarity} />
          </span>
          <span className="shrink-0 text-xs text-amber-200/80">{passage.pages}</span>
        </div>
        <p className="mt-1 truncate text-sm font-semibold text-amber-200">
          {passage.book ?? "Unknown book"}
        </p>
        <p className="truncate font-mono text-[10px] text-amber-300/50">{passage.chunk_id}</p>
        {passage.weak && (
          <span className="mt-1 inline-block rounded-full border border-red-500/50 bg-red-950/40 px-1.5 py-px text-[10px] uppercase tracking-wide text-red-200">
            weak match
          </span>
        )}

        {open && (
          <div className="mt-2 border-t border-amber-800/50 pt-2">
            <p className="text-[11px] italic text-amber-300/70">
              {passage.section}
              {passage.author ? ` · ${passage.author}` : ""}
            </p>
            <p className="mt-1.5 whitespace-pre-line text-xs leading-relaxed text-stone-200/90">
              {passage.text}
            </p>
          </div>
        )}
      </button>
    </li>
  );
};

export const Chunks = ({ passages, loading = false }: ChunksProps) => {
  if (loading) {
    return <p className="animate-pulse text-sm italic text-amber-300/70">Consulting the scrolls…</p>;
  }
  if (!passages.length) {
    return (
      <p className="text-sm italic text-amber-300/60">
        No scrolls were opened for the last reply. Ask about a god, a king or a ritual and the
        passages Thot-Sphinx read will appear here.
      </p>
    );
  }
  return (
    <ol className="space-y-2">
      {passages.map((p, i) => (
        <ChunkRow key={p.chunk_id} passage={p} rank={i + 1} />
      ))}
    </ol>
  );
};

export default Chunks;
