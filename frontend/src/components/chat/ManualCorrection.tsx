// ============================================================
// <ManualCorrection /> — the user corrects a sign directly
// ============================================================
// Human in the loop without waiting for the agent: pick a line, pick the
// sign (shown with its current code), type the right Gardiner code, and
// apply. The correction is posted in the chat as a card and the whole
// inscription is re-read with it.

import { useMemo, useState } from "react";
import { useGlyphStore, useReadingChatStore } from "@/stores";
import { useRereadCooldown } from "@/hooks/useRereadCooldown";

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

export const ManualCorrection = () => {
  const result = useGlyphStore((s) => s.result);
  const busy = useReadingChatStore((s) => s.isLoading);
  const applyManual = useReadingChatStore((s) => s.applyManualCorrection);
  const { coolingDown, availableAtLabel } = useRereadCooldown();
  const [open, setOpen] = useState(false);
  const [line, setLine] = useState(1);
  const [position, setPosition] = useState(1);
  const [code, setCode] = useState("");

  const lines = result?.readingLines ?? [];
  const signs = useMemo(() => lines.find((l) => l.line === line)?.codes ?? [], [lines, line]);
  const current = signs[position - 1];
  const toCode = normalizeCode(code);
  const valid = Boolean(current) && CODE_RE.test(toCode) && toCode !== current;

  if (!lines.length) return null;

  const submit = () => {
    if (!valid || busy || coolingDown || !current) return;
    applyManual({
      line,
      position,
      from_code: current,
      to_code: toCode,
      reason: "Corrected by the user against the stone.",
    });
    setCode("");
    setOpen(false);
  };

  return (
    <div className="rounded-xl border-2 border-amber-600/60 bg-gradient-to-b from-stone-900 to-amber-950/60 px-3 py-2">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        className="flex w-full items-center justify-between text-left text-[11px] font-semibold uppercase tracking-[0.2em] text-amber-300"
      >
        <span>✍ Correct a sign yourself</span>
        <span aria-hidden className="text-amber-500">{open ? "▾" : "▸"}</span>
      </button>

      {open && (
        <div className="mt-2 space-y-2">
          <div className="flex flex-wrap items-end gap-2">
            <label className="flex flex-col gap-1 text-[10px] uppercase tracking-wide text-amber-500/90">
              Line
              <select
                value={line}
                onChange={(e) => {
                  setLine(Number(e.target.value));
                  setPosition(1);
                }}
                className={fieldClass}
              >
                {lines.map((l) => (
                  <option key={l.line} value={l.line}>
                    {l.line}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex min-w-0 flex-1 flex-col gap-1 text-[10px] uppercase tracking-wide text-amber-500/90">
              Sign
              <select
                value={position}
                onChange={(e) => setPosition(Number(e.target.value))}
                className={fieldClass}
              >
                {signs.map((c, i) => (
                  <option key={i} value={i + 1}>
                    {i + 1} · {c}
                  </option>
                ))}
              </select>
            </label>
            <label className="flex w-24 flex-col gap-1 text-[10px] uppercase tracking-wide text-amber-500/90">
              Should be
              <input
                value={code}
                onChange={(e) => setCode(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && submit()}
                placeholder="N28"
                maxLength={6}
                className={`${fieldClass} font-mono`}
              />
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={submit}
              disabled={!valid || busy || coolingDown}
              className="rounded-md border border-amber-300 bg-gradient-to-b from-amber-400 to-amber-600 px-3 py-1 text-[11px] font-bold uppercase tracking-wide text-amber-950 shadow-[0_2px_8px_rgba(180,120,30,0.35)] hover:from-amber-300 hover:to-amber-500 disabled:opacity-50"
            >
              Apply & re-read
            </button>
            <span className="text-[10px] text-amber-500/80">
              {coolingDown
                ? `⏳ One correction re-read per hour — next at ${availableAtLabel}.`
                : current
                ? code && !CODE_RE.test(toCode)
                  ? "Use a Gardiner code, e.g. N28, G1, Aa15."
                  : `${current} → ${code ? toCode : "?"}`
                : "Pick a sign."}
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default ManualCorrection;
