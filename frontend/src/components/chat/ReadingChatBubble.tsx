// ============================================================
// <ReadingChatBubble /> — floating Sphinx button that toggles the reading chat
// ============================================================
// Appears only once the user has opened the reading chat for the current
// result (first "Ask Thot-Sphinx" click), then stays fixed bottom-left so the
// drawer can be reopened without scrolling back to the button.

import { useGlyphStore, useReadingChatStore } from "@/stores";
import { cn } from "@/utils/cn";
import sphinxIcon from "@/assets/sphinx_icon.svg";

export const ReadingChatBubble = () => {
  const result = useGlyphStore((s) => s.result);
  const readingId = useReadingChatStore((s) => s.readingId);
  const open = useReadingChatStore((s) => s.open);
  const isLoading = useReadingChatStore((s) => s.isLoading);
  const openFor = useReadingChatStore((s) => s.openFor);
  const close = useReadingChatStore((s) => s.close);

  // Not opened for this reading yet: stay hidden.
  if (!result || readingId !== result.id) return null;

  const label = open ? "Hide the Sphinx chat" : "Talk to Thot-Sphinx about this reading";

  return (
    <button
      type="button"
      onClick={() => (open ? close() : openFor(result))}
      aria-label={label}
      title={label}
      aria-expanded={open}
      className={cn(
        // Above the mobile tab bar below lg; above the drawer (z-60) so it can hide it.
        "fixed bottom-20 left-4 z-[70] h-14 w-14 items-center justify-center rounded-full",
        "border-2 border-amber-400 bg-gradient-to-br from-stone-800 to-stone-950 p-2",
        "shadow-[0_6px_20px_rgba(0,0,0,0.6),0_0_18px_rgba(251,191,36,0.35)]",
        "transition-transform hover:scale-105 active:scale-95 lg:bottom-6 lg:left-6",
        // On phones the drawer is a bottom sheet: hide the bubble there while open (✕ closes it).
        open ? "hidden sm:flex" : "flex",
      )}
    >
      <img src={sphinxIcon} alt="" className="h-full w-full" />
      {isLoading && (
        <span className="absolute -right-0.5 -top-0.5 h-3.5 w-3.5 animate-pulse rounded-full border-2 border-stone-950 bg-amber-400" />
      )}
    </button>
  );
};

export default ReadingChatBubble;
