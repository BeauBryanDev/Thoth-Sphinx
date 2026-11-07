// ============================================================
// <SeekerMemory /> — what Thot-Sphinx remembers, and "Forget me"
// ============================================================
// Agent memory lives only in this browser (localStorage, keys "sphinx.*"):
// the seeker profile sent to /chat/, the last 10 readings, the last reading
// and both conversations. "Forget me" clears all of it and reloads.

import { useState } from "react";
import { Button, Card } from "@/components/common";
import { loadDecodeHistory, loadProfile, seekerMemoryText } from "@/utils/seekerMemory";
import { forgetEverything } from "@/utils/storage";

const day = (ms: number | null): string => (ms ? new Date(ms).toLocaleDateString() : "—");

export const SeekerMemory = () => {
  const [profile] = useState(loadProfile);
  const [history] = useState(loadDecodeHistory);
  const [line] = useState(seekerMemoryText);
  const [confirming, setConfirming] = useState(false);

  const forget = () => {
    forgetEverything();
    window.location.reload();
  };

  return (
    <Card className="lg:col-span-3" title="What the Sphinx remembers" icon="𓂀">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
        <div>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-sm">
            <dt className="text-amber-400/80">Name</dt>
            <dd className="text-amber-100">{profile.name ?? "—"}</dd>
            <dt className="text-amber-400/80">Walls read</dt>
            <dd className="text-amber-100">{profile.readings}</dd>
            <dt className="text-amber-400/80">Last reading</dt>
            <dd className="text-amber-100">{day(profile.lastReadingAt)}</dd>
            <dt className="text-amber-400/80">Riddle trials passed</dt>
            <dd className="text-amber-100">{profile.trialsPassed}</dd>
            <dt className="text-amber-400/80">Cartouche gifts</dt>
            <dd className="text-amber-100">{profile.gifts}</dd>
            <dt className="text-amber-400/80">Kings seen</dt>
            <dd className="text-amber-100">
              {Object.entries(profile.kings).map(([k, n]) => `${k}${n > 1 ? ` ×${n}` : ""}`).join(", ") || "—"}
            </dd>
          </dl>

          <p className="mt-4 text-[10px] uppercase tracking-[0.2em] text-amber-500/70">Sent to Thot-Sphinx each chat turn</p>
          <p className="mt-1 rounded border border-amber-800/50 bg-stone-950/60 p-2 font-mono text-xs text-amber-200/80">
            {line || "nothing yet — a first visit costs no extra tokens"}
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-2">
            {confirming ? (
              <>
                <span className="text-sm text-red-200">Erase every memory in this browser?</span>
                <Button size="sm" onClick={forget}>Yes, forget me</Button>
                <Button size="sm" variant="ghost" onClick={() => setConfirming(false)}>Cancel</Button>
              </>
            ) : (
              <Button size="sm" variant="secondary" onClick={() => setConfirming(true)}>Forget me</Button>
            )}
          </div>
          <p className="mt-2 text-[11px] text-amber-400/60">
            Stored only in this browser. Nothing is kept on the server.
          </p>
        </div>

        <div>
          <p className="mb-2 text-[10px] uppercase tracking-[0.2em] text-amber-500/70">Last readings</p>
          {history.length ? (
            <ol className="space-y-2">
              {history.map((h) => (
                <li key={`${h.id}-${h.at}`} className="rounded border border-amber-700/40 bg-amber-950/30 px-3 py-2">
                  <div className="flex flex-wrap items-baseline justify-between gap-2 text-xs">
                    <span className="font-semibold text-amber-200">{h.kings.join(", ") || "No royal name"}</span>
                    <span className="text-amber-400/70">{day(h.at)} · {h.lines} lines</span>
                  </div>
                  <p className="mt-1 truncate font-serif text-sm italic text-amber-100/80">“{h.translation}”</p>
                </li>
              ))}
            </ol>
          ) : (
            <p className="text-sm italic text-amber-300/60">No wall has been read in this browser yet.</p>
          )}
        </div>
      </div>
    </Card>
  );
};
