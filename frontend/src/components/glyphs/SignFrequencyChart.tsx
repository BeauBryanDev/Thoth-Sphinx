// ============================================================
// <SignFrequencyChart /> — Gardiner sign frequency (Recharts)
//
// Basic stats about the current detection: X axis = each distinct
// detected sign, Y axis = how many times it was detected (integer
// counts, NOT probabilities). Sorted most-frequent first.
// ============================================================

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { GlyphDecodingResult } from "@/types";
import { isCartouche } from "@/utils/gardinerUnicode";

const GOLD = "#f59e0b";
const GOLD_DIM = "#b45309";

interface SignFrequencyChartProps {
  result: GlyphDecodingResult | null;
}

export const SignFrequencyChart = ({ result }: SignFrequencyChartProps) => {
  if (!result || result.detectedGlyphs.length === 0) return null;

  // Cartouche markers are places in a line, not signs.
  const signs = result.detectedGlyphs.filter((code) => !isCartouche(code));
  const counts = new Map<string, number>();
  for (const code of signs) {
    counts.set(code, (counts.get(code) ?? 0) + 1);
  }
  const data = [...counts.entries()]
    .map(([sign, count]) => ({ sign, count }))
    .sort((a, b) => b.count - a.count || a.sign.localeCompare(b.sign));

  const maxCount = data[0]?.count ?? 1;

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <p className="mb-1.5 text-[10px] uppercase tracking-[0.25em] text-amber-500/80">
        Sign frequency · {data.length} distinct signs ·{" "}
        {signs.length} detections
      </p>
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={data}
            margin={{ top: 8, right: 8, bottom: 4, left: -22 }}
          >
            <CartesianGrid
              strokeDasharray="3 3"
              stroke="rgba(217,119,6,0.18)"
              vertical={false}
            />
            <XAxis
              dataKey="sign"
              interval={0}
              angle={-55}
              textAnchor="end"
              height={44}
              tick={{ fill: "#d97706", fontSize: 10, fontFamily: "monospace" }}
              axisLine={{ stroke: "rgba(217,119,6,0.4)" }}
              tickLine={{ stroke: "rgba(217,119,6,0.4)" }}
            />
            <YAxis
              allowDecimals={false}
              domain={[0, maxCount]}
              tick={{ fill: "#d97706", fontSize: 10, fontFamily: "monospace" }}
              axisLine={{ stroke: "rgba(217,119,6,0.4)" }}
              tickLine={{ stroke: "rgba(217,119,6,0.4)" }}
            />
            <Tooltip
              cursor={{ fill: "rgba(245,158,11,0.08)" }}
              contentStyle={{
                background: "#1c1208",
                border: "1px solid rgba(217,119,6,0.5)",
                borderRadius: 6,
                fontSize: 12,
                color: "#fde68a",
              }}
              labelStyle={{ color: "#fbbf24", fontFamily: "monospace" }}
              formatter={(value: number) => [
                `${value} time${value === 1 ? "" : "s"}`,
                "detected",
              ]}
            />
            <Bar
              dataKey="count"
              fill={GOLD}
              stroke={GOLD_DIM}
              strokeWidth={1}
              radius={[3, 3, 0, 0]}
              maxBarSize={28}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default SignFrequencyChart;
