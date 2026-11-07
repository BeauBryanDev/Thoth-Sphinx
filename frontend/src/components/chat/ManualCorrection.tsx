// ============================================================
// <ManualCorrection /> — the user corrects signs directly
// ============================================================
// Human in the loop without waiting for the agent: for each row pick a
// line, pick the sign (shown with its current code) and type the right
// Gardiner code. [+] adds a row, up to MAX_CORRECTIONS. Apply posts the
// corrections in the chat as cards and re-reads the whole inscription
// once with all of them (still one re-read per hour).

import { useState } from "react";
import { MAX_CORRECTIONS, useGlyphStore, useReadingChatStore } from "@/stores";
import { useRereadCooldown } from "@/hooks/useRereadCooldown";
import type { SignCorrection } from "@/types";

// Same shape the backend accepts: A1, Aa15, N28, W25a…
const CODE_RE = /^(?:[A-Z]|Aa|NL|NU)\d{1,3}[a-z]?$/;

/** g1 -> G1, aa1 -> Aa1, as the detector and the corpus write codes. */
const normalizeCode = (raw: string): string => {
  const code = raw.trim();
  if (code.slice(0, 2).toLowerCase() === "aa") return "Aa" + code.slice(2);
  return code.charAt(0).toUpperCase() + code.slice(1);
};

const fieldClass =
  "rounded-md border border-amber-700/60 bg-stone-900/80 px-2 py-1.5 text-xs text-amber-100 focus:border-amber-400 focus:outline-none";

const labelClass = "flex flex-col gap-1 text-[10px] uppercase tracking-wide text-amber-500/90";

interface Row {
  id: number;
  line: number;
  position: number;
  code: string;
}

type RowState = "empty" | "bad-code" | "same" | "duplicate" | "ok";

export const ManualCorrection = () => {
  const result = useGlyphStore((s) => s.result);
  const busy = useReadingChatStore((s) => s.isLoading);
  const applyManual = useReadingChatStore((s) => s.applyManualCorrections);
  const { coolingDown, availableAtLabel } = useRereadCooldown();
  const [open, setOpen] = useState(false);
  const [rows, setRows] = useState<Row[]>([{ id: 0, line: 1, position: 1, code: "" }]);
  const [nextId, setNextId] = useState(1);

  const lines = result?.readingLines ?? [];
  if (!lines.length) return null;

  const signsOf = (line: number): string[] => lines.find((l) => l.line === line)?.codes ?? [];
  const update = (id: number, patch: Partial<Row>) =>
    setRows((rs) => rs.map((r) => (r.id === id ? { ...r, ...patch } : r)));
  const addRow = () => {
    if (rows.length >= MAX_CORRECTIONS) return;
    const last = rows[rows.length - 1];
    // Start the new row on the sign after the last one, a common case.
    const position = Math.min(last.position + 1, Math.max(signsOf(last.line).length, 1));
    setRows((rs) => [...rs, { id: nextId, line: last.line, position, code: "" }]);
    setNextId((n) => n + 1);
  };
  const removeRow = (id: number) => setRows((rs) => rs.filter((r) => r.id !== id));

  const stateOf = (row: Row): RowState => {
    const current = signsOf(row.line)[row.position - 1];
    const toCode = normalizeCode(row.code);
    if (!row.code.trim() || !current) return "empty";
    if (!CODE_RE.test(toCode)) return "bad-code";
    if (toCode === current) return "same";
    const twin = rows.find((r) => r.id !== row.id && r.line === row.line && r.position === row.position);
    if (twin && twin.id < row.id) return "duplicate";
    return "ok";
  };
  const states = rows.map(stateOf);
  const valid = states.every((s) => s === "ok");

  const submit = () => {
    if (!valid || busy || coolingDown) return;
    const corrections: SignCorrection[] = rows.map((r) => ({
      line: r.line,
      position: r.position,
      from_code: signsOf(r.line)[r.position - 1],
      to_code: normalizeCode(r.code),
      reason: "Corrected by the user against the stone.",
    }));
    applyManual(corrections);
    setRows([{ id: nextId, line: 1, position: 1, code: "" }]);
    setNextId((n) => n + 1);
    setOpen(false);
  };

  const hint = (row: Row, state: RowState): string => {
    const current = signsOf(row.line)[row.position - 1];
    if (state === "bad-code") return "Use a Gardiner code, e.g. N28, G1, Aa15.";
    if (state === "same") return `Already ${current}.`;
    if (state === "duplicate") return "This sign is already in another row.";
    return `${current ?? "?"} → ${row.code ? normalizeCode(row.code) : "?"}`;
  };

  return (
    <div className="rounded-xl border-2 border-amber-600/60 bg-gradient-to-b from-stone-900 to-amber-950/60 px-3 py-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left text-[11px] font-semibold uppercase tracking-[0.2em] text-amber-300"
      >
        <span>✍ Correct signs yourself</span>
        <span aria-hidden className="text-amber-500">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          {rows.map((row, k) => {
            const signs = signsOf(row.line);
            return (
              <div key={row.id} className="space-y-1">
                <div className="flex flex-wrap items-end gap-2">
                  <label className={labelClass}>
                    Line
                    <select
                      value={row.line}
                      onChange={(e) => update(row.id, { line: Number(e.target.value), position: 1 })}
                      className={fieldClass}
                    >
                      {lines.map((l) => (
                        <option key={l.line} value={l.line}>
                          {l.line}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={`${labelClass} min-w-0 flex-1`}>
                    Sign
                    <select
                      value={row.position}
                      onChange={(e) => update(row.id, { position: Number(e.target.value) })}
                      className={fieldClass}
                    >
                      {signs.map((c, i) => (
                        <option key={i} value={i + 1}>
                          {i + 1} · {c}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className={`${labelClass} w-20`}>
                    Should be
                    <input
                      value={row.code}
                      onChange={(e) => update(row.id, { code: e.target.value })}
                      onKeyDown={(e) => e.key === "Enter" && submit()}
                      placeholder="N28"
                      maxLength={6}
                      className={`${fieldClass} font-mono`}
                    />
                  </label>
                  {rows.length > 1 && (
                    <button
                      type="button"
                      onClick={() => removeRow(row.id)}
                      aria-label={`Remove correction ${k + 1}`}
                      className="mb-0.5 flex h-7 w-7 items-center justify-center rounded-full border border-amber-700/60 text-amber-300 hover:bg-amber-900/60"
                    >
                      −
                    </button>
                  )}
                </div>
                {row.code && (
                  <p className={`text-[10px] ${states[k] === "ok" ? "text-amber-500/80" : "text-red-300"}`}>
                    {hint(row, states[k])}
                  </p>
                )}
              </div>
            );
          })}

          <div className="flex flex-wrap items-center gap-2">
            {rows.length < MAX_CORRECTIONS && (
              <button
                type="button"
                onClick={addRow}
                aria-label="Add another sign"
                title={`Correct up to ${MAX_CORRECTIONS} signs in one re-read`}
                className="flex h-7 w-7 items-center justify-center rounded-full border border-amber-500/80 bg-amber-900/40 text-sm font-bold text-amber-200 hover:bg-amber-800/60"
              >
                +
              </button>
            )}
            <button
              type="button"
              onClick={submit}
              disabled={!valid || busy || coolingDown}
              className="rounded-md border border-amber-300 bg-gradient-to-b from-amber-400 to-amber-600 px-3 py-1 text-[11px] font-bold uppercase tracking-wide text-amber-950 shadow-[0_2px_8px_rgba(180,120,30,0.35)] hover:from-amber-300 hover:to-amber-500 disabled:opacity-50"
            >
              Apply {rows.length > 1 ? `${rows.length} fixes` : ""} & re-read
            </button>
            <span className="text-[10px] text-amber-500/80">
              {coolingDown
                ? `⏳ One correction re-read per hour — next at ${availableAtLabel}.`
                : `${rows.length} of ${MAX_CORRECTIONS} signs · one re-read`}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ManualCorrection;
