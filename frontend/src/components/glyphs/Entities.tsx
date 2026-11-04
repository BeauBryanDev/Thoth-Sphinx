// ============================================================
// <Entities /> — the kings behind the matched royal cartouches
// ============================================================
// Cards come from the knowledge graph (app/services/entity_cards.py):
// the pipeline looks each matched cartouche's king up before Claude
// reads the wall, so the same facts reach the prompt and this card.
// Background, not evidence — the footer says where each fact comes from.

import { toHieroglyph } from "@/utils/gardinerUnicode";
import type { EntityCard } from "@/types";

const KING_SIGN = toHieroglyph("M23") ?? "♔"; // the sedge, nsw "king"

const Fact = ({ label, value }: { label: string; value: string | null }) =>
  value ? (
    <span className="rounded-full border border-amber-700/50 bg-stone-900/70 px-2.5 py-0.5 text-xs text-amber-100">
      <span className="text-amber-500/80">{label} </span>
      {value}
    </span>
  ) : null;

export const Entities = ({ entities }: { entities: EntityCard[] }) => {
  if (!entities.length) return null;

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <p className="mb-2 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
        <span className="font-hieroglyph mr-1 text-sm normal-case tracking-normal">{KING_SIGN}</span>
        {entities.length === 1 ? "The king in the cartouche" : "The kings in the cartouches"}
      </p>

      <div className="space-y-3">
        {entities.map((e) => (
          <article key={e.id} className="rounded-lg border border-amber-600/40 bg-amber-950/30 p-3">
            <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <h4 className="font-serif text-lg font-semibold text-amber-200">{e.name}</h4>
              <span className="text-[11px] text-amber-400/80">cartouche: {e.cartouche}</span>
            </header>

            <div className="mt-2 flex flex-wrap gap-1.5">
              <Fact label="Dynasty" value={e.dynasty} />
              <Fact label="Period" value={e.period} />
              <Fact label="Reign" value={e.reign} />
            </div>

            {e.monuments.length > 0 && (
              <p className="mt-2 text-sm text-amber-100/90">
                <span className="text-amber-500/80">Monuments: </span>
                {e.monuments.join(" · ")}
              </p>
            )}

            {e.notes && <p className="mt-2 text-sm leading-relaxed text-amber-100/80">{e.notes}</p>}

            <footer className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-amber-500/70">
              <span>
                Knowledge graph · {e.reliability} · {e.source}
              </span>
              {e.wikipedia_url && (
                <a
                  href={e.wikipedia_url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-amber-300 underline decoration-amber-600/60 underline-offset-2 hover:text-amber-200"
                >
                  Read more ↗
                </a>
              )}
            </footer>
          </article>
        ))}
      </div>
    </div>
  );
};
