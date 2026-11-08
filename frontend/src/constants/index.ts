// ============================================================
// Application Constants
// ============================================================

import type { NavItem } from "@/types";

export const APP_NAME = "SphinxEyes";
export const APP_TAGLINE = "AI Vision. Ancient Wisdom.";
export const APP_DESCRIPTION =
  "Advanced Computer Vision | AI Expert Chatbot — Unlock the secrets of Ancient Egypt.";

export const FOOTER_LINES = [
  "Powered by Pro AI Vision",
  "Inspired by the wisdom of Osiris",
];

// ---------- Navigation ----------
export const NAV_ITEMS: NavItem[] = [
  { key: "home", label: "Home", icon: "🏠" },
  { key: "chat", label: "SphinxChat", icon: "💬" },
  { key: "glyphs", label: "Transliterate", icon: "👁" },
  { key: "transliteration", label: "Reverse", icon: "𓆓" },
  { key: "rosetta", label: "Rosetta Stone", icon: "𓉴" },
  { key: "learn", label: "Learn More", icon: "☥" },
  { key: "profile", label: "Profile", icon: "👤" },
];

// ---------- Chat suggestions ----------
export const CHAT_SUGGESTIONS: string[] = [
  "Who was Osiris in Egyptian mythology?",
  "Explain the purpose of the pyramids.",
  "How did hieroglyphs evolve over time?",
  "Tell me about the life of a pharaoh.",
];

// ---------- Feature cards ----------
export interface FeatureCard {
  title: string;
  description: string;
  icon: string;
}

export const FEATURES: FeatureCard[] = [
  {
    title: "SphinxChat",
    description: "Your AI expert on Ancient Egypt — history, gods, pharaohs, and mysteries.",
    icon: "𓁹",
  },
  {
    title: "Glyph Decoder",
    description: "Upload an image to transliterate hieroglyphs with computer vision.",
    icon: "𓂀",
  },
  {
    title: "Reverse Translation",
    description: "Write in English and see it composed in Middle Egyptian hieroglyphs.",
    icon: "𓋹",
  },
];
