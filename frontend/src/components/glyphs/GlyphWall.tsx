// ============================================================
// <GlyphWall /> — JSesh-style rendering of the decoded signs
//
// Right-hand panel of the Glyphs page. Turns the pipeline's
// Gardiner codes into real Unicode hieroglyphs (U+13000 block,
// Noto Sans Egyptian Hieroglyphs) laid out in a golden grid in
// the SAME spatial order as the Gardiner-code view: one block
// per physical line (row/column). Before any decode it falls
// back to a small reference key of common signs.
// ============================================================

import { Card } from "@/components/common";
import { toHieroglyph } from "@/utils/gardinerUnicode";
import type { GlyphDecodingResult } from "@/types";

const FALLBACK_KEY = [
  { glyph: "𓁹", translit: "jr", meaning: "Eye / to do" },
  { glyph: "𓂀", translit: "wḏꜣt", meaning: "Eye of Horus" },
  { glyph: "𓋹", translit: "ꜥnḫ", meaning: "Life (ankh)" },
  { glyph: "𓉐", translit: "pr", meaning: "House" },
  { glyph: "𓇋", translit: "j", meaning: "Reed leaf" },
  { glyph: "𓊃", translit: "z", meaning: "Door bolt" },
  { glyph: "𓏏", translit: "t", meaning: "Bread loaf" },
];

interface GlyphCellProps {
  code: string;
}

const GlyphCell = ({ code }: GlyphCellProps) => {
  const glyph = toHieroglyph(code);
  // A real sign outside the detector's 148 classes (e.g. N28 confirmed by the user): show its code.
  const unmapped = !glyph && code !== "Unknown" && code !== "unknown";
  return (
    <span
      title={glyph ? code : unmapped ? `${code} (no glyph in this font map)` : "Undetected sign (lacuna)"}
      className={`flex h-14 w-14 items-center justify-center rounded border ${
        glyph
          ? "border-amber-600/40 bg-stone-950/70 hover:border-amber-400/70"
          : unmapped
            ? "border-amber-500/60 bg-amber-950/50"
            : "border-stone-600/50 bg-stone-800/50"
      } transition-colors`}
    >
      {glyph ? (
        <span className="font-hieroglyph glyph-gold text-[34px]">{glyph}</span>
      ) : unmapped ? (
        <span className="font-mono text-xs font-semibold text-amber-300">{code}</span>
      ) : (
        <span className="text-sm text-stone-500">▯</span>
      )}
    </span>
  );
};

interface GlyphWallProps {
  result: GlyphDecodingResult | null;
}

export const GlyphWall = ({ result }: GlyphWallProps) => {
  if (!result) {
    return (
      <Card
        title="Glyph Wall"
        subtitle="Decoded signs will appear here as real hieroglyphs"
        icon="𓊹"
      >
        <p className="mb-3 text-[10px] uppercase tracking-[0.2em] text-amber-500/70">
          Reference key — common signs
        </p>
        <ul className="divide-y divide-amber-700/40 text-sm">
          {FALLBACK_KEY.map((g) => (
            <li
              key={g.glyph}
              className="flex items-center justify-between gap-3 py-2"
            >
              <span className="font-hieroglyph glyph-gold text-2xl">
                {g.glyph}
              </span>
              <span className="flex-1 text-right font-mono text-xs text-amber-200">
                {g.translit}
              </span>
              <span className="w-28 text-right text-xs text-amber-400/80">
                {g.meaning}
              </span>
            </li>
          ))}
        </ul>
      </Card>
    );
  }

  const isColumns = result.layout === "columns";
  const isRtl = result.direction === "rtl";

  return (
    <Card
      title="Glyph Wall"
      subtitle={`As read from your image · ${isColumns ? "columns" : "rows"}, ${
        isRtl ? "right → left" : "left → right"
      }`}
      icon="𓊹"
    >
      {isColumns ? (
        /* COLUMNS — replicate the wall: each physical line is a vertical
           column of stacked signs. Reading rtl puts col 1 at the RIGHT
           edge (flex-row-reverse), exactly like the original image. */
        <div
          className={`flex ${isRtl ? "flex-row-reverse" : "flex-row"} justify-start gap-2 overflow-x-auto rounded-md border border-amber-700/40 bg-gradient-to-b from-stone-900/80 to-stone-950/80 p-3 shadow-inner`}
        >
          {result.lines.map((line, i) => (
            <div key={i} className="flex shrink-0 flex-col items-center">
              <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-amber-600/80">
                c{i + 1}
              </p>
              <div className="flex flex-col gap-1.5 border-x border-amber-800/30 px-1.5 py-1">
                {line.map((code, j) => (
                  <GlyphCell key={j} code={code} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        /* ROWS — each physical line is a horizontal band. Reading rtl
           puts the first sign of each row at the RIGHT edge. */
        <div className="space-y-3">
          {result.lines.map((line, i) => (
            <div key={i}>
              <p className="mb-1 font-mono text-[10px] uppercase tracking-wider text-amber-600/80">
                row {i + 1}
              </p>
              <div
                className={`flex flex-wrap ${isRtl ? "flex-row-reverse" : "flex-row"} gap-1.5 rounded-md border border-amber-700/40 bg-gradient-to-b from-stone-900/80 to-stone-950/80 p-2 shadow-inner`}
              >
                {line.map((code, j) => (
                  <GlyphCell key={j} code={code} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
      <p className="mt-3 text-[10px] leading-relaxed text-amber-500/70">
        Laid out as on the original wall — read each{" "}
        {isColumns ? "column top to bottom, columns" : "row"}{" "}
        {isRtl ? "right to left" : "left to right"} · hover a sign for its
        Gardiner code · ▯ marks an undetected sign.
      </p>
    </Card>
  );
};

export default GlyphWall;
