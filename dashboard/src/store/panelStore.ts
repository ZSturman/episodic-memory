/**
 * Zustand store for dashboard panel visibility, verbosity, and UI state.
 * Persists layout preferences to localStorage.
 */
import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import type { PanelVisibility, VerbosityPreset } from "@/lib/types";
import { VERBOSITY_PRESETS } from "@/lib/types";

interface PanelState {
  // ─── Panel visibility ──────────────────────────────────────────────
  panels: PanelVisibility;
  togglePanel: (key: keyof PanelVisibility) => void;
  setAllPanels: (v: PanelVisibility) => void;

  // ─── Verbosity preset ──────────────────────────────────────────────
  verbosity: VerbosityPreset;
  setVerbosity: (v: VerbosityPreset) => void;

  // ─── Polling ───────────────────────────────────────────────────────
  pollingMs: number;
  setPollingMs: (ms: number) => void;
  paused: boolean;
  togglePause: () => void;

  // ─── Step selection ────────────────────────────────────────────────
  selectedStep: number | null;
  setSelectedStep: (step: number | null) => void;

  // ─── Memory detail ─────────────────────────────────────────────────
  selectedMemoryId: string | null;
  setSelectedMemoryId: (id: string | null) => void;

  // ─── Sidebar ───────────────────────────────────────────────────────
  sidebarOpen: boolean;
  toggleSidebar: () => void;
}

export const usePanelStore = create<PanelState>()(
  persist(
    (set) => ({
      // Defaults to "diagnose"
      panels: { ...VERBOSITY_PRESETS.diagnose },
      togglePanel: (key) =>
        set((s) => ({
          panels: { ...s.panels, [key]: !s.panels[key] },
        })),
      setAllPanels: (v) => set({ panels: v }),

      verbosity: "diagnose",
      setVerbosity: (v) =>
        set({
          verbosity: v,
          panels: { ...VERBOSITY_PRESETS[v] },
        }),

      pollingMs: 200,
      setPollingMs: (ms) => set({ pollingMs: ms }),
      paused: false,
      togglePause: () => set((s) => ({ paused: !s.paused })),

      selectedStep: null,
      setSelectedStep: (step) => set({ selectedStep: step }),

      selectedMemoryId: null,
      setSelectedMemoryId: (id) => set({ selectedMemoryId: id }),

      sidebarOpen: true,
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
    }),
    {
      name: "panorama-dashboard",
      storage: createJSONStorage(() => localStorage),
      partialize: (s) => ({
        panels: s.panels,
        verbosity: s.verbosity,
        pollingMs: s.pollingMs,
        sidebarOpen: s.sidebarOpen,
      }),
      merge: (persisted, current) => {
        const merged = { ...current, ...(persisted as object) };
        // Ensure new panel keys exist even if localStorage is stale
        if (merged.panels) {
          const defaults = VERBOSITY_PRESETS[merged.verbosity ?? "diagnose"];
          merged.panels = { ...defaults, ...merged.panels };
        }
        return merged;
      },
    },
  ),
);
