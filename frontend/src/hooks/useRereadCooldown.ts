// ============================================================
// useRereadCooldown — is Apply & re-read available right now?
// ============================================================
// Mirrors the server guardrail (one correction re-read per hour) so the
// buttons are disabled with a clear "next one at HH:MM", and switch back
// on by themselves when the hour is up. The server stays the authority.

import { useEffect, useState } from "react";
import { useReadingChatStore } from "@/stores";

export const useRereadCooldown = (): { coolingDown: boolean; availableAtLabel: string } => {
  const availableAt = useReadingChatStore((s) => s.rereadAvailableAt);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!availableAt || availableAt <= Date.now()) return;
    const timer = window.setInterval(() => setNow(Date.now()), 30_000);
    const done = window.setTimeout(() => setNow(Date.now()), availableAt - Date.now() + 500);
    return () => {
      window.clearInterval(timer);
      window.clearTimeout(done);
    };
  }, [availableAt]);

  const coolingDown = Boolean(availableAt && now < availableAt);
  const availableAtLabel = availableAt
    ? new Date(availableAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : "";
  return { coolingDown, availableAtLabel };
};

export default useRereadCooldown;
