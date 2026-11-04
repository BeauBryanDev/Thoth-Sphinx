// ============================================================
// ToolCallChips — the tools Thot-Sphinx called for one reply
// ============================================================
// Shared by the general chat and the reading-chat drawer: a chip per
// call, gold when it worked, red when the tool failed.

import type { ToolCallOut } from "@/types";

export const TOOL_LABEL: Record<string, string> = {
  search_parallels: "corpus search",
  lookup_lemma: "dictionary",
  lookup_entity: "knowledge graph",
};

export const ToolCallChips = ({ calls }: { calls: ToolCallOut[] }) => {
  if (!calls.length) return null;
  return (
    <div className="flex flex-wrap gap-1.5">
      {calls.map((c, i) => (
        <span
          key={i}
          title={JSON.stringify(c.input)}
          className={`inline-flex max-w-full items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] ${
            c.ok
              ? "border-amber-700/60 bg-stone-900/70 text-amber-200/90"
              : "border-red-500/50 bg-red-950/40 text-red-200"
          }`}
        >
          <span aria-hidden>⌕</span>
          <span className="font-semibold uppercase tracking-wide">{TOOL_LABEL[c.tool] ?? c.tool}</span>
          <span className="truncate text-amber-300/80">{c.summary}</span>
        </span>
      ))}
    </div>
  );
};
