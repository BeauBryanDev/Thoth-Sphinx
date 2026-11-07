// ============================================================
// Browser storage for agent memory — every access guarded
// ============================================================
// localStorage can be blocked (private windows), full, or cleared; memory
// then simply goes missing, never crashes the app. All keys start with
// "sphinx." so "Forget me" can clear them in one sweep.

import { createJSONStorage } from "zustand/middleware";

export const STORAGE_PREFIX = "sphinx.";

const guarded: Storage = {
  get length() {
    try {
      return localStorage.length;
    } catch {
      return 0;
    }
  },
  key: (i) => {
    try {
      return localStorage.key(i);
    } catch {
      return null;
    }
  },
  getItem: (k) => {
    try {
      return localStorage.getItem(k);
    } catch {
      return null;
    }
  },
  setItem: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      // full or blocked: keep running without memory
    }
  },
  removeItem: (k) => {
    try {
      localStorage.removeItem(k);
    } catch {
      // blocked: nothing to remove
    }
  },
  clear: () => {
    try {
      localStorage.clear();
    } catch {
      // blocked
    }
  },
};

/** The zustand persist storage: JSON over guarded localStorage. */
export const memoryStorage = createJSONStorage(() => guarded);

export const readJson = <T,>(key: string): T | null => {
  const raw = guarded.getItem(key);
  if (!raw) return null;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
};

export const writeJson = (key: string, value: unknown): void => guarded.setItem(key, JSON.stringify(value));

/** "Forget me": remove every key this app wrote. */
export const forgetEverything = (): void => {
  const keys: string[] = [];
  for (let i = 0; i < guarded.length; i++) {
    const k = guarded.key(i);
    if (k?.startsWith(STORAGE_PREFIX)) keys.push(k);
  }
  keys.forEach((k) => guarded.removeItem(k));
};
