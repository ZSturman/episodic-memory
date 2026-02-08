"use client";

import type { AgentState } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

interface TopBarProps {
  state: AgentState | null;
  connected: boolean;
}

export function TopBar({ state, connected }: TopBarProps) {
  const toggleSidebar = usePanelStore((s) => s.toggleSidebar);
  const sidebarOpen = usePanelStore((s) => s.sidebarOpen);
  const paused = usePanelStore((s) => s.paused);
  const togglePause = usePanelStore((s) => s.togglePause);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const setPollingMs = usePanelStore((s) => s.setPollingMs);

  const agentState = state?.agent_state ?? "investigating_unknown";

  return (
    <header className="flex h-12 items-center justify-between border-b border-ctp-surface0 bg-ctp-base px-4">
      {/* Left: sidebar toggle + step info */}
      <div className="flex items-center gap-4">
        <button
          onClick={toggleSidebar}
          className="btn-ghost text-lg"
          title={sidebarOpen ? "Hide sidebar" : "Show sidebar"}
        >
          {sidebarOpen ? "◀" : "▶"}
        </button>

        {state && (
          <div className="flex items-center gap-3 text-sm">
            <span className="mono text-ctp-subtext0">
              Step {state.step}
            </span>
            <span className="text-ctp-overlay0">|</span>
            <span className="mono text-ctp-subtext0">
              {state.heading_deg.toFixed(0)}°
            </span>
            <span className="text-ctp-overlay0">|</span>
            <span className="text-ctp-subtext0">
              Viewport {state.viewport_index + 1}/{state.total_viewports}
            </span>
            <span className="text-ctp-overlay0">|</span>
            <span
              className="badge"
              style={{
                backgroundColor: STATE_COLORS[agentState] + "22",
                color: STATE_COLORS[agentState],
              }}
            >
              {STATE_LABELS[agentState]}
            </span>
          </div>
        )}
      </div>

      {/* Right: polling controls + connection */}
      <div className="flex items-center gap-3">
        {/* Polling rate control */}
        <div className="flex items-center gap-1.5 text-xs text-ctp-subtext0">
          <span>Poll</span>
          <select
            value={pollingMs}
            onChange={(e) => setPollingMs(Number(e.target.value))}
            className="rounded border border-ctp-surface0 bg-ctp-mantle px-1.5 py-0.5 text-xs text-ctp-text"
          >
            <option value={100}>100ms</option>
            <option value={200}>200ms</option>
            <option value={500}>500ms</option>
            <option value={1000}>1s</option>
            <option value={2000}>2s</option>
          </select>
        </div>

        {/* Pause button */}
        <button
          onClick={togglePause}
          className={`btn text-xs ${
            paused
              ? "bg-ctp-yellow text-ctp-crust"
              : "text-ctp-subtext0 hover:bg-ctp-surface0"
          }`}
        >
          {paused ? "▶ Resume" : "⏸ Pause"}
        </button>

        {/* Connection dot */}
        <div className="flex items-center gap-1 text-xs text-ctp-overlay0">
          <div
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: connected ? "#a6e3a1" : "#f38ba8" }}
          />
          {connected ? "Live" : "Offline"}
        </div>
      </div>
    </header>
  );
}
