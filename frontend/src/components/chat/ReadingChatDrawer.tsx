// ============================================================
// <ReadingChatDrawer /> — chat with Thot-Sphinx about this reading
// ============================================================
// Bottom sheet on mobile, right-hand drawer from `sm` up, so the result
// stays visible behind it. Reuses <ChatPanel /> (same gold card, bubbles,
// markdown, input) with the reading-chat store as its controller, and
// renders each reply's corpus tool calls and proposed sign corrections.

import { useEffect } from "react";
import { ChatPanel } from "@/components/chat/ChatPanel";
import { ManualCorrection } from "@/components/chat/ManualCorrection";
import { ToolCallChips } from "@/components/chat/ToolCallChips";
import { Chunks } from "@/components/chat/Chunks";
import { correctionKey, useReadingChatStore } from "@/stores";
import { useRereadCooldown } from "@/hooks/useRereadCooldown";
import type { ChatMessage, ReadingChatMessage } from "@/types";

/** Human in the loop: the user accepts or dismisses each proposed correction. */
const CorrectionActions = ({ messageId, index }: { messageId: string; index: number }) => {
  const status = useReadingChatStore((s) => s.correctionStatus[correctionKey(messageId, index)] ?? "pending");
  const busy = useReadingChatStore((s) => s.isLoading);
  const apply = useReadingChatStore((s) => s.applyCorrection);
  const dismiss = useReadingChatStore((s) => s.dismissCorrection);
  const { coolingDown, availableAtLabel } = useRereadCooldown();

  if (status === "applied") {
    return <p className="mt-2 text-[11px] font-semibold text-emerald-300">✓ Applied — the inscription was re-read with it.</p>;
  }
  if (status === "dismissed") {
    return <p className="mt-2 text-[11px] text-stone-400">Dismissed — the reading is unchanged.</p>;
  }
  return (
    <div className="mt-2 flex flex-wrap items-center gap-2">
      <button
        type="button"
        onClick={() => apply(messageId, index)}
        disabled={busy || coolingDown}
        title={coolingDown ? `One correction re-read per hour — next at ${availableAtLabel}` : undefined}
        className="rounded-md border border-amber-300 bg-gradient-to-b from-amber-400 to-amber-600 px-3 py-1 text-[11px] font-bold uppercase tracking-wide text-amber-950 shadow-[0_2px_8px_rgba(180,120,30,0.35)] hover:from-amber-300 hover:to-amber-500 disabled:opacity-50"
      >
        {status === "applying" ? "Re-reading…" : status === "failed" ? "Retry apply & re-read" : "Apply & re-read"}
      </button>
      {status !== "applying" && (
        <button
          type="button"
          onClick={() => dismiss(messageId, index)}
          disabled={busy}
          className="rounded-md border border-amber-700/60 bg-amber-900/40 px-3 py-1 text-[11px] font-semibold uppercase tracking-wide text-amber-200 hover:bg-amber-800/60 disabled:opacity-50"
        >
          Dismiss
        </button>
      )}
      <span className="text-[10px] text-amber-500/80">
        {status === "applying"
          ? "Claude is re-reading the whole inscription (about 30–70 s)…"
          : coolingDown
            ? `⏳ One correction re-read per hour — next at ${availableAtLabel}.`
            : "You decide — nothing changes until you accept."}
      </span>
    </div>
  );
};

const MessageExtras = ({ message }: { message: ChatMessage }) => {
  const {
    toolCalls = [],
    bookPassages = [],
    corrections = [],
    correctionBatch = false,
  } = message as ReadingChatMessage;
  if (!toolCalls.length && !bookPassages.length && !corrections.length) return null;

  return (
    <div className="mt-3 space-y-2 border-t border-amber-800/50 pt-2">
      <ToolCallChips calls={toolCalls} />

      {bookPassages.length > 0 && (
        <details className="group rounded-lg border border-amber-700/50 bg-stone-900/40">
          <summary className="cursor-pointer select-none px-2.5 py-1.5 text-[10px] uppercase tracking-[0.2em] text-amber-400 hover:text-amber-300">
            📜 Scrolls consulted ({bookPassages.length})
            <span className="ml-1 normal-case tracking-normal text-amber-500/70 group-open:hidden">— click to open</span>
          </summary>
          <div className="px-2 pb-2">
            <Chunks passages={bookPassages} />
          </div>
        </details>
      )}

      {corrections.map((c, i) => (
        <div key={i} className="rounded-lg border-2 border-amber-500/70 bg-amber-950/50 p-2.5">
          <p className="text-[10px] uppercase tracking-[0.2em] text-amber-500/90">𓂀 Proposed correction</p>
          <p className="mt-1 text-sm font-semibold text-amber-100">
            Line {c.line}, sign {c.position}:{" "}
            <span className="rounded bg-stone-900/70 px-1.5 py-0.5 font-mono text-amber-300">{c.from_code}</span>
            <span className="mx-1.5 text-amber-500">→</span>
            <span className="rounded bg-amber-800/70 px-1.5 py-0.5 font-mono text-amber-50">{c.to_code}</span>
          </p>
          <p className="mt-1 text-xs leading-relaxed text-amber-200/80">{c.reason}</p>
          {/* The user's own batch is applied together: one action row below all its cards. */}
          {!correctionBatch && <CorrectionActions messageId={message.id} index={i} />}
        </div>
      ))}
      {correctionBatch && corrections.length > 0 && <CorrectionActions messageId={message.id} index={0} />}
    </div>
  );
};

export const ReadingChatDrawer = () => {
  const chat = useReadingChatStore();
  const { open, close } = chat;

  // Esc closes; the page behind does not scroll while the drawer is open.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && close();
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [open, close]);

  if (!open) return null;

  const suggestions = [
    "Why did you read it this way?",
    "What does the corpus confirm?",
    "Which signs are you least sure of?",
    "Is any sign misread? Propose a fix.",
  ];

  return (
    <div className="fixed inset-0 z-[60]" role="dialog" aria-modal="true" aria-label="Chat about this reading">
      <button
        type="button"
        aria-label="Close chat"
        onClick={close}
        className="absolute inset-0 bg-stone-950/70 backdrop-blur-[2px]"
      />
      <aside className="reading-drawer absolute inset-x-0 bottom-0 flex h-[88dvh] flex-col p-2 sm:inset-y-0 sm:left-auto sm:right-0 sm:h-full sm:w-[460px] sm:p-3">
        {/* Header row: the manual correction form, and the close button beside it (never on top of it). */}
        <div className="mb-2 flex items-start gap-2">
          <div className="min-w-0 flex-1">
            <ManualCorrection />
          </div>
          <button
            type="button"
            onClick={close}
            aria-label="Close chat"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-amber-600/70 bg-stone-900/90 text-amber-200 shadow hover:bg-amber-900/60"
          >
            ✕
          </button>
        </div>
        <ChatPanel
          controller={chat}
          title="Ask Thot-Sphinx"
          subtitle="About this reading · the TLA corpus at hand"
          icon="𓂀"
          suggestions={suggestions}
          placeholder="Ask about this reading, or point out a sign…"
          className="min-h-0 flex-1 bg-gradient-to-b from-stone-900 to-stone-950 shadow-[0_0_40px_rgba(180,120,30,0.25)]"
          renderExtras={(m) => <MessageExtras message={m} />}
        />
      </aside>
    </div>
  );
};

export default ReadingChatDrawer;
