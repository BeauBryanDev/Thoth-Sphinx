// ============================================================
// <Sidebar /> — left navigation panel with Egyptian theme
// ============================================================

import { NAV_ITEMS, APP_NAME, APP_TAGLINE } from "@/constants";
import { useApp } from "@/contexts/AppContext";
import { cn } from "@/utils/cn";
import type { PageKey } from "@/types";
import sphinxMaster from "@/assets/pharaoh_mask.webp";
import scarab from "@/assets/golden_scarab.webp";
import horus from "@/assets/horus.svg";
import lotus from "@/assets/lotus.svg";
import amphora from "@/assets/amphora.svg";
import leg from "@/assets/leg.svg";
import snake from "@/assets/snake.svg";
import owl from "@/assets/owl.svg";
import n26 from "@/assets/n26.svg";
import v8 from "@/assets/v8.svg";
import chatEye from "@/assets/chat_eye_icon.svg";

// Sidebar footer frieze, left to right.
const FRIEZE = [horus, lotus, amphora, leg, snake, owl, n26, v8];

interface SidebarProps {
  onNavigate?: (page: PageKey) => void;
}

export const Sidebar = ({ onNavigate }: SidebarProps) => {
  const { activePage, setActivePage, sidebarOpen, closeSidebar } = useApp();

  const handleClick = (key: PageKey) => {
    setActivePage(key);
    onNavigate?.(key);
    closeSidebar(); // dismiss the mobile drawer; no-op visually on lg+
  };

  return (
    <aside
      className={cn(
        // Mobile: off-canvas drawer that slides in over the content.
        "fixed inset-y-0 left-0 z-40 flex h-full w-64 flex-col overflow-hidden border-r border-amber-700/50 bg-gradient-to-b from-stone-900 via-amber-950 to-stone-900 text-amber-200 shadow-2xl transition-transform duration-300",
        // Desktop: static inline column, always visible.
        "lg:static lg:z-auto lg:shrink-0 lg:translate-x-0 lg:shadow-none",
        sidebarOpen ? "translate-x-0" : "-translate-x-full",
      )}
    >
      {/* Hieroglyph frame */}
      <div className="pointer-events-none absolute inset-y-0 left-0 w-64 border-r border-amber-700/20 opacity-20 text-[10px] leading-5 text-amber-500 select-none">
        <div className="h-full overflow-hidden px-2 pt-2">
          {"𓀀𓁿𓂀𓃭𓄿𓅓𓆣𓇋𓈖𓉐𓊪𓋹𓌻𓍯𓎼𓏏𓐍".repeat(40)}
        </div>
      </div>

      {/* Brand */}
      <div className="relative z-10 flex flex-col items-center px-6 pt-8 pb-6 text-center">
        <img
          src={sphinxMaster}
          alt="SphinxEyes pharaoh emblem"
          className="h-28 w-auto drop-shadow-[0_4px_14px_rgba(180,120,30,0.45)] select-none"
          draggable={false}
        />
        <h1 className="mt-3 text-3xl font-black tracking-wider text-amber-300 drop-shadow">
          {APP_NAME}
        </h1>
        <p className="mt-1 text-xs uppercase tracking-[0.3em] text-amber-400/80">
          {APP_TAGLINE}
        </p>
        <div className="mt-4 h-px w-full bg-gradient-to-r from-transparent via-amber-600/60 to-transparent" />
      </div>

      {/* Nav items */}
      <nav className="relative z-10 space-y-1 px-4">
        {NAV_ITEMS.map((item) => {
          const isActive = item.key === activePage;
          return (
            <button
              key={item.key}
              onClick={() => handleClick(item.key)}
              className={cn(
                "group flex w-full items-center gap-3 rounded-md px-3 py-2.5 text-left text-sm font-medium transition-colors",
                isActive
                  ? "bg-gradient-to-r from-amber-500/30 via-amber-600/20 to-transparent text-amber-200 shadow-inner"
                  : "text-amber-300/80 hover:bg-amber-900/40 hover:text-amber-100",
              )}
            >
              <span
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-sm text-base",
                  isActive
                    ? "bg-amber-500 text-amber-950"
                    : "bg-amber-900/50 text-amber-300 group-hover:bg-amber-800/60",
                )}
              >
                {item.icon}
              </span>
              <span className="tracking-wide">{item.label}</span>
              {isActive && (
                <span className="ml-auto text-amber-400">▸</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* The eye fills whatever height is left between the nav and the footer; it shrinks away on short screens. */}
      <div className="relative z-10 min-h-0 flex-1" aria-hidden="true">
        <img
          src={chatEye}
          alt=""
          className="absolute inset-0 h-full w-full object-contain px-8 py-4 opacity-80 drop-shadow-[0_0_18px_rgba(251,191,36,0.35)] select-none"
          draggable={false}
        />
      </div>

      {/* Footer emblem */}
      <div className="relative z-10 flex flex-col items-center px-6 py-6 text-center text-amber-500/80">
        <img
          src={scarab}
          alt=""
          aria-hidden="true"
          className="h-20 w-auto opacity-90 select-none"
          draggable={false}
        />
        <div className="mt-2 text-[10px] uppercase tracking-[0.3em]">
          Pro AI Vision
        </div>
        <div className="mt-3 flex w-full items-end justify-between gap-1" aria-hidden="true">
          {FRIEZE.map((src, i) => (
            <img
              key={i}
              src={src}
              alt=""
              className="h-5 w-5 object-contain opacity-75 drop-shadow-[0_1px_2px_rgba(0,0,0,0.6)] select-none"
              draggable={false}
            />
          ))}
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
