// ============================================================
// <SignConfidenceTable /> — per-sign detector confidence
//
// Sits below <GlyphWall /> on the Glyphs page. One row per
// DETECTION (repeats are expected), in the SAME spatial reading
// order as the wall, grouped by physical line. Stone-tablet
// styling: brown borders, chiseled bands, colored probability
// bars — green ≥ 0.80, yellow 0.50–0.80, red below 0.50.
// Confidences arrive already clamped to ≤ 0.998 (never 1.0).
// ============================================================

import { Fragment } from "react";
import { Card } from "@/components/common";
import { isCartouche, toHieroglyph } from "@/utils/gardinerUnicode";
import type { GlyphDecodingResult } from "@/types";

const barColor = (p: number): string =>
  p >= 0.8
    ? "bg-gradient-to-r from-emerald-600 to-emerald-400"
    : p >= 0.5
      ? "bg-gradient-to-r from-yellow-600 to-yellow-400"
      : "bg-gradient-to-r from-red-700 to-red-500";

const textColor = (p: number): string =>
  p >= 0.8 ? "text-emerald-300" : p >= 0.5 ? "text-yellow-300" : "text-red-300";

interface SignConfidenceTableProps {
  result: GlyphDecodingResult | null;
}

export const SignConfidenceTable = ({ result }: SignConfidenceTableProps) => {
  if (!result) return null;

  const lineLabel = result.layout === "columns" ? "col" : "row";

  // Walk lines with a running flat index so confidences stay aligned
  // with the spatial ordering shown on the wall above.
  let flat = 0;
  // A cartouche marker keeps its place in the flat index but is not a sign: no row.
  const groups = result.lines.map((line) =>
    line
      .map((code) => ({ code, conf: result.signConfidences[flat++] ?? 0 }))
      .filter(({ code }) => !isCartouche(code)),
  );
  const nSigns = groups.reduce((n, line) => n + line.length, 0);

  return (
    <Card
      title="Sign Confidence"
      subtitle={`${nSigns} detections · spatial reading order`}
      icon="𓏞"
    >
      <div className="overflow-hidden rounded-md border-2 border-amber-800/70 shadow-inner">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr className="border-b-2 border-amber-800/70 bg-gradient-to-b from-stone-700/80 to-stone-800/80 text-[10px] uppercase tracking-[0.2em] text-amber-300/90">
              <th className="px-2 py-2 text-left font-semibold">Sign</th>
              <th className="px-2 py-2 text-left font-semibold">Code</th>
              <th className="px-2 py-2 text-right font-semibold">Prob</th>
              <th className="w-1/2 px-2 py-2 text-left font-semibold">
                Confidence
              </th>
            </tr>
          </thead>
          <tbody>
            {groups.map((line, li) => (
              <Fragment key={li}>
                <tr className="border-y border-amber-800/60 bg-gradient-to-b from-amber-950/90 to-stone-900/90">
                  <td
                    colSpan={4}
                    className="px-2 py-1 font-mono text-[10px] uppercase tracking-[0.25em] text-amber-500/90"
                  >
                    𓊛 {lineLabel} {li + 1}
                  </td>
                </tr>
                {line.map(({ code, conf }, i) => {
                  const glyph = toHieroglyph(code);
                  return (
                    <tr
                      key={`${li}-${i}`}
                      className="border-b border-amber-900/40 bg-stone-900/60 transition-colors odd:bg-stone-950/60 hover:bg-amber-950/50"
                    >
                      <td className="px-2 py-1.5">
                        {glyph ? (
                          <span className="font-hieroglyph glyph-gold text-xl">
                            {glyph}
                          </span>
                        ) : (
                          <span className="text-stone-500">▯</span>
                        )}
                      </td>
                      <td className="px-2 py-1.5 font-mono text-amber-200">
                        {code}
                      </td>
                      <td
                        className={`px-2 py-1.5 text-right font-mono ${textColor(conf)}`}
                      >
                        {conf.toFixed(3)}
                      </td>
                      <td className="px-2 py-1.5">
                        <div className="h-2.5 w-full overflow-hidden rounded-sm border border-stone-700/60 bg-stone-950/80">
                          <div
                            className={`h-full ${barColor(conf)}`}
                            style={{ width: `${Math.max(conf * 100, 2)}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 flex flex-wrap gap-x-3 text-[10px] text-amber-500/70">
        <span>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-emerald-500" />
          ≥ 0.80 strong
        </span>
        <span>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-yellow-500" />
          0.50–0.80 fair
        </span>
        <span>
          <span className="mr-1 inline-block h-2 w-2 rounded-sm bg-red-600" />
          &lt; 0.50 weak
        </span>
      </p>
    </Card>
  );
};

export default SignConfidenceTable;
