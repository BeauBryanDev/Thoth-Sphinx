// ============================================================
// Decoder hints — clickable icons + modals that explain the
// layout and reading-direction toggles to first-time visitors.
// ============================================================

import { useEffect, useId, useRef, type ReactNode } from "react";
import type { ReadingDirection, WallLayout } from "@/types";

interface HintButtonProps {
  icon: string;
  caption: string;
  label: string;
  onClick: () => void;
}

export const HintButton = ({ icon, caption, label, onClick }: HintButtonProps) => (
  <button
    type="button"
    onClick={onClick}
    aria-label={label}
    title={label}
    className="group flex flex-col items-center gap-0.5 rounded-lg p-1 transition-colors hover:bg-amber-900/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-amber-400"
  >
    <img
      src={icon}
      alt=""
      className="h-10 w-10 transition-transform group-hover:scale-110 sm:h-14 sm:w-14"
    />
    <span className="text-[10px] font-bold tracking-wide text-amber-400">
      [?] <span className="hidden sm:inline">{caption}</span>
    </span>
  </button>
);

interface HintModalProps {
  open: boolean;
  onClose: () => void;
  icon: string;
  title: string;
  children: ReactNode;
}

export const HintModal = ({ open, onClose, icon, title, children }: HintModalProps) => {
  const titleId = useId();
  const closeRef = useRef<HTMLButtonElement>(null);
  // Ref keeps the effect from re-running (and re-focusing) on every parent render.
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCloseRef.current();
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4"
      onClick={onClose}
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        onClick={(e) => e.stopPropagation()}
        className="relative max-h-[90dvh] w-full max-w-md overflow-y-auto rounded-2xl border-2 border-amber-500/70 bg-gradient-to-b from-amber-950 to-stone-900 p-6 text-sm text-amber-100/90 shadow-[0_8px_30px_rgba(0,0,0,0.6)]"
      >
        <button
          ref={closeRef}
          type="button"
          onClick={onClose}
          aria-label="Close hint"
          className="absolute right-3 top-3 rounded px-2 py-0.5 text-amber-400 hover:bg-amber-900/40"
        >
          ✕
        </button>
        <div className="mb-4 flex items-center gap-3 pr-8">
          <img src={icon} alt="" className="h-12 w-12" />
          <h4
            id={titleId}
            className="text-base font-bold uppercase tracking-[0.2em] text-amber-300"
          >
            {title}
          </h4>
        </div>
        {children}
      </div>
    </div>
  );
};

const choiceClass = (active: boolean) =>
  `flex flex-1 flex-col items-center gap-2 rounded-lg border p-3 transition-colors ${
    active
      ? "border-amber-400 bg-amber-600/20"
      : "border-amber-700/50 bg-stone-900/50 hover:bg-amber-900/30"
  }`;

const SAMPLE = ["𓅓", "𓂋", "𓏏"];

interface LayoutHintProps {
  layout: WallLayout;
  onPick: (l: WallLayout) => void;
}

export const LayoutHint = ({ layout, onPick }: LayoutHintProps) => (
  <div className="space-y-3">
    <p>
      Look at how the signs are lined up in your photo. Egyptian scribes carved
      them either in horizontal <b className="text-amber-300">rows</b> or in
      vertical <b className="text-amber-300">columns</b>, often separated by
      carved lines.
    </p>
    <p className="font-semibold text-amber-300">
      Tell us: is your image laid out in rows or in columns?
    </p>
    <div className="flex gap-3">
      <button type="button" onClick={() => onPick("rows")} className={choiceClass(layout === "rows")}>
        <div className="font-hieroglyph glyph-gold space-y-1 text-xl leading-none">
          {[0, 1, 2].map((r) => (
            <div key={r}>{SAMPLE.join(" ")}</div>
          ))}
        </div>
        <span className="text-xs font-semibold uppercase tracking-wide">Rows</span>
      </button>
      <button type="button" onClick={() => onPick("columns")} className={choiceClass(layout === "columns")}>
        <div className="font-hieroglyph glyph-gold flex gap-3 text-xl leading-none">
          {[0, 1, 2].map((c) => (
            <div key={c} className="flex flex-col gap-1">
              {SAMPLE.map((g) => (
                <span key={g}>{g}</span>
              ))}
            </div>
          ))}
        </div>
        <span className="text-xs font-semibold uppercase tracking-wide">Columns</span>
      </button>
    </div>
    <p className="text-xs italic text-amber-200/70">
      Not sure? Temple and tomb walls are often written in columns; stelae and
      door lintels often in rows.
    </p>
  </div>
);

// Unicode hieroglyphs face left by default; mirrored they face right.
const CREATURES = "𓅓 𓄿 𓆑";

interface DirectionHintProps {
  direction: ReadingDirection | null;
  onPick: (d: ReadingDirection) => void;
}

export const DirectionHint = ({ direction, onPick }: DirectionHintProps) => (
  <div className="space-y-3">
    <p>
      Ancient Egyptians wrote and read <b className="text-amber-300">towards the
      faces</b> of living creatures. Birds, people, snakes and other animals all
      look towards the start of the line, so you read into their faces.
    </p>
    <p>
      Find the birds and animals in your image and see where they are looking.
    </p>
    <div className="flex flex-col gap-3">
      <button type="button" onClick={() => onPick("rtl")} className={choiceClass(direction === "rtl")}>
        <div className="font-hieroglyph glyph-gold -scale-x-100 text-2xl leading-none">
          {CREATURES}
        </div>
        <span className="text-xs">
          Faces look <b>right</b> → start at the right edge:{" "}
          <b className="uppercase">Right → Left</b>
        </span>
      </button>
      <button type="button" onClick={() => onPick("ltr")} className={choiceClass(direction === "ltr")}>
        <div className="font-hieroglyph glyph-gold text-2xl leading-none">
          {CREATURES}
        </div>
        <span className="text-xs">
          Faces look <b>left</b> → start at the left edge:{" "}
          <b className="uppercase">Left → Right</b>
        </span>
      </button>
    </div>
    <p className="font-semibold text-amber-300">
      Tell us: which direction does your image read?
    </p>
    <p className="text-xs italic text-amber-200/70">
      Not sure? Most inscriptions read right to left.
    </p>
  </div>
);
