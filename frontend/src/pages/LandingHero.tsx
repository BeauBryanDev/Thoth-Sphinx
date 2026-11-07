// ============================================================
// <LandingHero /> — public landing page (default "home")
// Hero banner + mission board. No chat / decoder panels here;
// those live on their own authenticated pages.
// ============================================================

import { Card } from "@/components/common";
import sphinxBackground from "@/assets/sphinx_background.jpg";
import pharaohMask from "@/assets/pharaoh_mask.webp";
import horusEye from "@/assets/horus_eye.webp";
import anubis from "@/assets/anubis.svg";
import owl from "@/assets/owl.svg";
import ankhSign from "@/assets/ankh_sign.svg";
import nefertiti from "@/assets/nefertiti.svg";
import ramsses from "@/assets/ramsses.svg";

/** A royal portrait beside the How It Works title. */
const Portrait = ({ src }: { src: string }) => (
  <img
    src={src}
    alt=""
    aria-hidden="true"
    className="mx-auto h-10 w-auto select-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.6)] sm:h-14"
    draggable={false}
  />
);

/** Anubis, owl, ankh beside the mission title; mirrored on the right so it reads ankh, owl, Anubis. Phones keep only the ankh. */
const MissionFrieze = ({ mirrored = false }: { mirrored?: boolean }) => (
  <div
    aria-hidden="true"
    className={`mx-auto flex items-end gap-2 sm:gap-4 ${mirrored ? "-scale-x-100" : ""}`}
  >
    {[anubis, owl, ankhSign].map((src, i) => (
      <img
        key={i}
        src={src}
        alt=""
        className={`h-9 w-auto select-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.6)] sm:h-12 ${i < 2 ? "hidden sm:block" : ""}`}
        draggable={false}
      />
    ))}
  </div>
);


// Mission pipeline — sourced from the project README (System Overview).
const PIPELINE_STAGES: { step: string; title: string; description: string }[] = [
  {
    step: "I",
    title: "Image Enhancement",
    description: "CLAHE, denoising and unsharp masking lift weathered carvings out of the stone.",
  },
  {
    step: "II",
    title: "Glyph Detection",
    description: "YOLOv11-Large reads 150 Gardiner classes natively, cartouche interiors included.",
  },
  {
    step: "III",
    title: "Layout & Reading Order",
    description: "Geometry decides rows vs. columns; quadrat clustering recovers the reading sequence.",
  },
  {
    step: "IV",
    title: "Semantic Correction",
    description: "A Viterbi pass over a lexicon trie and a bigram language model repairs misreads.",
  },
  {
    step: "V",
    title: "Cartouche Reading",
    description: "Needleman-Wunsch alignment matches royal names against an attested database.",
  },
  {
    step: "VI",
    title: "Transliteration",
    description: "The corrected sign sequence becomes structured, readable phonetic output.",
  },
];

const HorusDivider = () => (
  <div className="flex items-center justify-center gap-4">
    <span className="h-px w-16 bg-gradient-to-r from-transparent to-amber-600/60" />
    <img
      src={horusEye}
      alt=""
      aria-hidden="true"
      className="h-12 w-auto opacity-90 select-none"
      draggable={false}
    />
    <span className="h-px w-16 bg-gradient-to-l from-transparent to-amber-600/60" />
  </div>
);

export const LandingHero = () => {
  
  return (
    <div className="space-y-8">
      {/* Hero banner */}
      <section className="relative overflow-hidden rounded-2xl border-2 border-amber-500/60 bg-stone-950 shadow-[0_10px_40px_rgba(0,0,0,0.45)]">
        <img
          src={sphinxBackground}
          alt=""
          aria-hidden="true"
          className="absolute inset-0 h-full w-full object-cover object-center select-none"
          draggable={false}
        />
        <div className="absolute inset-0 bg-gradient-to-b from-stone-950/70 via-stone-950/45 to-stone-950/85" />

        <div className="relative z-10 flex flex-col items-center px-6 py-14 text-center md:py-20">
          <img
            src={pharaohMask}
            alt="SphinxEyes pharaoh emblem"
            className="h-32 w-auto drop-shadow-[0_4px_18px_rgba(180,120,30,0.55)] select-none md:h-44"
            draggable={false}
          />
          <h1 className="mt-4 text-4xl font-black tracking-[0.15em] text-amber-300 drop-shadow md:text-6xl">
            SphinxEyes
          </h1>
          <p className="mt-3 text-sm uppercase tracking-[0.35em] text-amber-200/80 md:text-base">
            Advanced Computer Vision · AI Expert Chatbot
          </p>
          <p className="mt-3 font-serif text-lg text-amber-100/90 md:text-xl">
            Unlock the Secrets of Ancient Egypt
          </p>
          <div className="mx-auto mt-4 flex items-center justify-center gap-2 text-amber-500/80">
            <span>—</span>
            <span>☥</span>
            <span>—</span>
          </div>
        </div>
      </section>

      {/* Mission board */}
      <Card
        title="Our Mission"
        headerBalance
        headerStart={<MissionFrieze />}
        headerEnd={<MissionFrieze mirrored />}
        subtitle="From weathered stone to readable word"
        icon={
          <img
            src={horusEye}
            alt=""
            aria-hidden="true"
            className="h-14 w-auto opacity-90 select-none"
            draggable={false}
          />
        }
      >
        <p className="mx-auto max-w-3xl text-center text-base leading-relaxed text-amber-100/90">
          SphinxEyes reads Middle Egyptian hieroglyphs straight from photographs.
          Dense, weathered, and arranged in rows or columns, the signs of antiquity
          resist easy reading — so the pipeline carries each inscription end to end,
          from raw image bytes to a structured, corrected transliteration.
        </p>

        <div className="my-8">
          <HorusDivider />
        </div>

        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {PIPELINE_STAGES.map((stage) => (
            <div
              key={stage.step}
              className="rounded-lg border border-amber-800/50 bg-stone-950/40 p-4 transition-colors hover:border-amber-600/70 hover:bg-stone-950/60"
            >
              <div className="flex items-center gap-3">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-amber-500/60 font-serif text-sm font-bold text-amber-300">
                  {stage.step}
                </span>
                <h4 className="text-sm font-bold uppercase tracking-[0.18em] text-amber-300">
                  {stage.title}
                </h4>
              </div>
              <p className="mt-2 text-sm leading-relaxed text-amber-200/80">
                {stage.description}
              </p>
            </div>
          ))}
        </div>
      </Card>

      {/* How it works: the whole pipeline in three paragraphs (vision, language, retrieval + agent). */}
      <Card
        title="How It Works"
        headerBalance
        headerStart={<Portrait src={nefertiti} />}
        headerEnd={<Portrait src={ramsses} />}
        subtitle="Vision · language · retrieval · agent"
        icon="𓂀"
      >
        <div className="mx-auto max-w-3xl space-y-5 text-base leading-relaxed text-amber-100/90">
          <p>
            <span className="font-bold uppercase tracking-[0.15em] text-amber-300">The eyes. </span>
            It begins with a photograph. A YOLO11 detector, trained on our own curated corpus of
            hieroglyphs, scans the whole image in a single pass at 1024 pixels. It finds every sign
            across 150 classes (148 Gardiner signs, plus royal cartouches and an &ldquo;unknown&rdquo; class)
            and gives each one a confidence score and its three most likely readings. Hieroglyphs have
            no spaces and can run left, right or downward. So a spatial layer groups the signs into
            quadrats, the little blocks a scribe carved together, then into lines, and puts them back
            in reading order using the layout and direction you choose.
          </p>
          <p>
            <span className="font-bold uppercase tracking-[0.15em] text-amber-300">The scribe&rsquo;s mind. </span>
            Raw detections are not yet words. A language layer can weigh each line against a lexicon
            of attested Egyptian spellings and a statistical model of which signs follow which. It
            also knows which signs the detector tends to mix up, so it can repair a likely misread
            instead of passing it on. Royal names get special care: the signs inside each cartouche
            are aligned against a register of 41 pharaohs, from the Third Dynasty to the Ptolemies.
            A matched king then anchors the rest of the reading.
          </p>
          <p>
            <span className="font-bold uppercase tracking-[0.15em] text-amber-300">The reader of stones. </span>
            Then an AI agent built on Claude reads the wall, looking at the photograph and the sign
            sequence together. It never answers from memory alone: it searches more than 100,000
            sentences from the Thesaurus Linguae Aegyptiae for parallel phrases. It looks up words,
            checks doubtful spellings and consults a knowledge graph of kings, gods and temples, all in
            parallel. It does this before writing a single line. If the evidence is thin, a
            rule-based router calls a stronger model for a second opinion. A second translation is then
            grounded in published translations of the same texts, found by hybrid search and reranking.
            Every claim arrives with its sources, its cost and a step-by-step trace. And you stay in
            the loop: tell Thot-Sphinx which sign looks wrong, and it reads the wall again with your
            correction in hand.
          </p>
        </div>
      </Card>
    </div>
  );
};

export default LandingHero;
