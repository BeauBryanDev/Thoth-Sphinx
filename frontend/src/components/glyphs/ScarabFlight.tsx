// ============================================================
// <ScarabFlight /> — a golden scarab zigzags across the screen while the pipeline runs
// ============================================================
// Three nested layers, each with its own period, so the path never repeats exactly:
// X sweeps wall to wall, Y bounces floor to ceiling, and the body flutters on a short zigzag.
// Purely decorative: pointer-events none, aria-hidden, off under prefers-reduced-motion.

import scarab from "@/assets/scarab.svg";

export const ScarabFlight = ({ active }: { active: boolean }) => {
  if (!active) return null;
  return (
    <div className="scarab-sky" aria-hidden>
      <div className="scarab-x">
        <div className="scarab-y">
          <div className="scarab-heading">
            <img src={scarab} alt="" className="scarab-body" draggable={false} />
          </div>
        </div>
      </div>
    </div>
  );
};

export default ScarabFlight;
