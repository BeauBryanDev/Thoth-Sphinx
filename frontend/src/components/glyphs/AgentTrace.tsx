// ============================================================
// <AgentTrace /> — every step of the read, in order
// ============================================================
// Renders TransliterationOut.trace (app/llm/trace.py): detection, each
// model turn with tokens and cost, each tool call, the router's decision
// with its signals, the escalation, and the final answer. Details expand
// per step so an engineer can see exactly what the agent did.

import { useState } from "react";
import type { TraceEvent } from "@/types";

const KIND_ICON: Record<string, string> = {
  detection: "𓁹",
  human_feedback: "✍",
  llm_call: "✦",
  tool_call: "⌕",
  evidence_check: "⌕",
  router: "⑂",
  escalation: "↗",
  final: "✓",
  error: "⚠",
};

const KIND_LABEL: Record<string, string> = {
  detection: "detection",
  human_feedback: "human feedback",
  llm_call: "model turn",
  tool_call: "tool call",
  evidence_check: "evidence check",
  router: "router",
  escalation: "escalation",
  final: "answer",
  error: "error",
};

const actorStyle = (actor: string): string => {
  if (actor.includes("sonnet")) return "border-sky-400/60 bg-sky-950/60 text-sky-100";
  if (actor.includes("haiku")) return "border-amber-500/60 bg-amber-900/50 text-amber-100";
  if (actor === "router") return "border-violet-400/60 bg-violet-950/50 text-violet-100";
  if (actor.startsWith("yolo")) return "border-emerald-500/60 bg-emerald-950/50 text-emerald-100";
  if (actor === "user") return "border-rose-400/60 bg-rose-950/50 text-rose-100";
  return "border-stone-500/60 bg-stone-800/60 text-stone-200";
};

const actorLabel = (actor: string): string =>
  actor.includes("sonnet") ? "Claude Sonnet" : actor.includes("haiku") ? "Claude Haiku" : actor;

const formatMs = (ms: number): string => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);

const formatCost = (usd: number): string =>
  usd < 0.001 ? `$${usd.toFixed(5)}` : `$${usd.toFixed(4)}`;

interface AgentTraceProps {
  trace: TraceEvent[];
}

export const AgentTrace = ({ trace }: AgentTraceProps) => {
  const [open, setOpen] = useState<number | null>(null);
  const last = trace[trace.length - 1];
  const toolCalls = trace.filter((e) => e.kind === "tool_call" || e.kind === "evidence_check").length;
  const turns = trace.filter((e) => e.kind === "llm_call").length;
  const cost = trace.reduce((sum, e) => sum + (e.kind === "llm_call" ? e.cost_usd ?? 0 : 0), 0);

  return (
    <div className="mt-3 rounded border border-amber-700/40 bg-stone-950/60 p-3">
      <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-[10px] uppercase tracking-[0.25em] text-amber-500/80">⛬ Agent trace</p>
        <p className="text-[10px] uppercase tracking-[0.2em] text-amber-600/80">
          {trace.length} steps · {turns} model turns · {toolCalls} corpus calls
          {last ? ` · ${formatMs(last.t_ms)}` : ""} · {formatCost(cost)}
        </p>
      </div>

      <ol className="relative ml-2 border-l border-amber-800/60">
        {trace.map((e) => {
          const expanded = open === e.step;
          const hasDetail = Object.keys(e.detail ?? {}).length > 0;
          return (
            <li key={e.step} className="mb-2 ml-4">
              <span
                aria-hidden
                className="absolute -left-[9px] flex h-[18px] w-[18px] items-center justify-center rounded-full border border-amber-700/70 bg-stone-950 text-[10px] text-amber-300"
              >
                {KIND_ICON[e.kind] ?? "•"}
              </span>
              <button
                type="button"
                disabled={!hasDetail}
                onClick={() => setOpen(expanded ? null : e.step)}
                className="w-full text-left disabled:cursor-default"
              >
                <div className="flex flex-wrap items-center gap-1.5">
                  <span className="text-[10px] uppercase tracking-wide text-amber-600/90">
                    {e.step}. {KIND_LABEL[e.kind] ?? e.kind}
                  </span>
                  <span className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold ${actorStyle(e.actor)}`}>
                    {actorLabel(e.actor)}
                  </span>
                  <span className="ml-auto text-[10px] tabular-nums text-amber-600/80">
                    {[
                      e.latency_ms != null ? formatMs(e.latency_ms) : null,
                      e.input_tokens != null ? `${e.input_tokens}→${e.output_tokens ?? 0} tok` : null,
                      e.cost_usd != null ? formatCost(e.cost_usd) : null,
                      `+${formatMs(e.t_ms)}`,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </div>
                <p className="mt-0.5 break-words text-xs leading-relaxed text-amber-100/90">
                  {e.summary}
                  {hasDetail && (
                    <span className="ml-1 text-[10px] text-amber-600/80">{expanded ? "▾" : "▸"}</span>
                  )}
                </p>
              </button>
              {expanded && (
                <pre className="mt-1 max-h-60 overflow-auto rounded border border-amber-900/50 bg-stone-900/70 p-2 text-[10px] leading-snug text-amber-200/90">
                  {JSON.stringify(e.detail, null, 2)}
                </pre>
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
};
