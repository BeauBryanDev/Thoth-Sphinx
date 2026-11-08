// ============================================================
// <App /> — root component, composes layout + active page
// ============================================================

import { AppProvider, useApp } from "@/contexts/AppContext";
import { Footer, Header, Sidebar, MobileNav } from "@/components/layout";
import { ToastContainer } from "@/components/common";
import {
  ChatPage,
  GlyphsPage,
  LearnPage,
  ProfilePage,
  ReversePage,
  RosettaStonePage,
} from "@/pages";
import LandingHero from "@/pages/LandingHero";
import type { PageKey } from "@/types";
import React from "react";

const PAGE_REGISTRY: Record<PageKey, React.ComponentType> = {
  home: LandingHero,
  chat: ChatPage,
  glyphs: GlyphsPage,
  transliteration: ReversePage,
  learn: LearnPage,
  rosetta: RosettaStonePage,
  profile: ProfilePage,
};

const AppShell = () => {
  const { activePage, sidebarOpen, closeSidebar } = useApp();
  const ActivePage = PAGE_REGISTRY[activePage];

  return (
    <div className="flex min-h-screen bg-gradient-to-br from-stone-950 via-amber-950 to-stone-900 text-amber-100">
      {/* Ambient desert background */}
      <div
        className="pointer-events-none fixed inset-0 opacity-25"
        style={{
          backgroundImage:
            "radial-gradient(ellipse at top, rgba(251,191,36,0.25), transparent 60%), radial-gradient(ellipse at bottom, rgba(120,53,15,0.35), transparent 60%)",
        }}
      />

      {/* Mobile drawer backdrop */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={closeSidebar}
          aria-hidden="true"
        />
      )}

      <Sidebar />

      <div className="relative z-10 flex min-w-0 flex-1 flex-col">
        <Header />
        <main className="flex-1 overflow-y-auto px-5 py-6 pb-24 sm:px-8 sm:py-8 lg:px-12 lg:pb-8 xl:px-16">
          <ActivePage />
        </main>
        {/* Desktop footer; mobile uses the bottom tab bar instead */}
        <div className="hidden lg:block">
          <Footer />
        </div>
      </div>

      <MobileNav />
      <ToastContainer />
    </div>
  );
};

const App = () => (
  <AppProvider>
    <AppShell />
  </AppProvider>
);

export default App;
