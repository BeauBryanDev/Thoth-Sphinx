// ============================================================
// <RosettaStonePage /> — the Rosetta Stone, rendered from rosetta_stone.md
// ============================================================

import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Card } from "@/components/common";
import rosettaStone from "@/assets/rosseta_stone.jpg";
import glyphs from "@/assets/glyphs.svg";
import source from "../../rosetta_stone.md?raw";

// The sections of rosetta_stone.md shown on the page, in order; the rest stay in the file.
const SECTIONS = [
  "Key Facts at a Glance",
  "Three Scripts, Two Languages",
  "The Decipherment: From Guesswork to System",
  "Significance and Legacy",
  "Timeline",
];

// The lead paragraph (before the first "## ") plus the chosen sections.
const pickSections = (md: string, titles: string[]): string => {
  const [lead, ...parts] = md.split(/^## /m);
  const byTitle = new Map(
    parts.map((p) => [p.slice(0, p.indexOf("\n")).trim(), `## ${p}`]),
  );
  const chosen = titles.map((t) => byTitle.get(t) ?? "");
  return [lead, ...chosen].join("\n");
};

const article = pickSections(source, SECTIONS);

// The page owns the only <h1>; the markdown's own title is dropped.
const MD: Components = {
  h1: () => null,
  h2: ({ children }) => (
    <h2 className="mt-10 border-b border-amber-700/40 pb-2 font-serif-egypt text-xl font-bold text-amber-300 sm:text-2xl">
      {children}
    </h2>
  ),
  h3: ({ children }) => (
    <h3 className="mt-6 text-lg font-semibold text-amber-200">{children}</h3>
  ),
  p: ({ children }) => (
    <p className="mt-3 leading-relaxed text-amber-100/90">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="mt-3 list-disc space-y-2 pl-6 text-amber-100/90">
      {children}
    </ul>
  ),
  ol: ({ children }) => (
    <ol className="mt-3 list-decimal space-y-2 pl-6 text-amber-100/90">
      {children}
    </ol>
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-amber-300">{children}</strong>
  ),
  em: ({ children }) => <em className="text-amber-200">{children}</em>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-amber-400 underline underline-offset-2 hover:text-amber-300"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="mt-4 border-l-4 border-amber-600/60 pl-4 italic text-amber-200/80">
      {children}
    </blockquote>
  ),
  table: ({ children }) => (
    <div className="mt-4 overflow-x-auto rounded-lg border border-amber-700/40">
      <table className="w-full border-collapse text-left text-sm">
        {children}
      </table>
    </div>
  ),
  thead: ({ children }) => (
    <thead className="bg-amber-900/40 text-amber-300">{children}</thead>
  ),
  th: ({ children }) => (
    <th className="px-3 py-2 font-semibold">{children}</th>
  ),
  td: ({ children }) => (
    <td className="border-t border-amber-800/30 px-3 py-2 align-top text-amber-100/90">
      {children}
    </td>
  ),
  img: () => null,
};

export const RosettaStonePage = () => {
  return (
    <Card bodyClassName="px-5 py-6 sm:px-10 sm:py-8">
      <article className="mx-auto max-w-4xl">
        <div className="flex items-center justify-center gap-3 sm:gap-5">
          <img
            src={glyphs}
            alt=""
            aria-hidden="true"
            className="h-10 w-10 shrink-0 select-none sm:h-14 sm:w-14"
            draggable={false}
          />
          <h1 className="text-center font-serif-egypt text-3xl font-black tracking-wide text-amber-300 sm:text-4xl">
            Discover Rosetta Stone
          </h1>
          <img
            src={glyphs}
            alt=""
            aria-hidden="true"
            className="h-10 w-10 shrink-0 -scale-x-100 select-none sm:h-14 sm:w-14"
            draggable={false}
          />
        </div>
        <img
          src={rosettaStone}
          alt="The Rosetta Stone in the British Museum"
          className="mx-auto mt-6 max-h-[32rem] w-auto rounded-xl border-2 border-amber-500/70 shadow-2xl"
        />
        <div className="mt-8">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>
            {article}
          </ReactMarkdown>
        </div>
      </article>
    </Card>
  );
};

export default RosettaStonePage;
