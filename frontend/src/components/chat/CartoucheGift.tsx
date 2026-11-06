// ============================================================
// CartoucheGift — the royal cartouche Thot-Sphinx carved for a seeker
// ============================================================
// The reward for passing the riddle trial: a WebP data URL from
// /chat/, shown under the reply with a client-side download.

export const CartoucheGift = ({ src }: { src: string }) => (
  <figure className="mt-3 flex flex-col items-center gap-2 border-t border-amber-800/50 pt-3">
    <img
      src={src}
      alt="Your name in a royal cartouche under the protection of Horus"
      className="w-full max-w-xs rounded-xl border-2 border-amber-500/70 shadow-[0_0_30px_rgba(180,120,30,0.3)]"
    />
    <a
      href={src}
      download="royal_cartouche.webp"
      className="rounded-full border border-amber-600/70 px-3 py-1 text-xs text-amber-200 hover:bg-amber-900/50"
    >
      𓍷 Download your cartouche
    </a>
  </figure>
);
