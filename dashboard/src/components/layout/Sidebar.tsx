"use client";

import type { AgentState, PanelVisibility } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

const PANEL_LABELS: Record<keyof PanelVisibility, string> = {
  viewport: "Viewport",
  panoramaPreview: "Panorama Preview",
  recentViewports: "Recent Viewports",
  features: "Features",
  evidence: "Evidence",
  matches: "Matches",
  confidenceTimeline: "Confidence",
  memoryList: "Memory List",
  memoryDetail: "Memory Detail",
  eventLog: "Event Log",
  labelRequest: "Label Request",
  locationGraph: "Location Graph",
  similarityHeatmap: "Similarity Heatmap",
  embeddingVariance: "Embedding Variance",
  replayControls: "Replay Controls",
  hexGrid: "Hex Grid",
  hexControl: "Hex Control",
  agentControl: "Agent Control",
  reconstruction: "Reconstruction",
  labelConfirm: "Label Confirm",
};

interface SidebarProps {
  state: AgentState | null;
  connected: boolean;
}

export function Sidebar({ state, connected }: SidebarProps) {
  const panels = usePanelStore((s) => s.panels);
  const togglePanel = usePanelStore((s) => s.togglePanel);
  const verbosity = usePanelStore((s) => s.verbosity);
  const setVerbosity = usePanelStore((s) => s.setVerbosity);

  const agentState = state?.agent_state ?? "investigating_unknown";
  const stateColor = STATE_COLORS[agentState];
  const stateLabel = STATE_LABELS[agentState];

  return (
    <aside className="flex w-56 flex-col border-r border-ctp-surface0 bg-ctp-mantle">
      {/* Brand */}
      <div className="flex items-center gap-2 border-b border-ctp-surface0 px-4 py-3">
        <div
          className="h-2.5 w-2.5 rounded-full"
          style={{ backgroundColor: connected ? "#a6e3a1" : "#f38ba8" }}
        />
        <span className="text-sm font-semibold text-ctp-text">
          Panorama Agent
        </span>
      </div>

      {/* Agent state badge */}
      <div className="border-b border-ctp-surface0 px-4 py-3">
        <div className="text-[10px] uppercase tracking-widest text-ctp-overlay0">
          Agent State
        </div>
        <div
          className="badge mt-1"
          style={{ backgroundColor: stateColor + "22", color: stateColor }}
        >
          {stateLabel}
        </div>
        {state && (
          <div className="mt-2 space-y-0.5 text-xs text-ctp-subtext0">
            <div>Step {state.step}</div>
            <div>
              {state.location_label}{" "}
              <span className="text-ctp-overlay0">
                ({(state.location_confidence * 100).toFixed(0)}%)
              </span>
            </div>
            <div>{state.episode_count} memories</div>
          </div>
        )}
      </div>

      {/* Verbosity */}
      <div className="border-b border-ctp-surface0 px-4 py-3">
        <div className="mb-2 text-[10px] uppercase tracking-widest text-ctp-overlay0">
          Verbosity
        </div>
        <div className="flex gap-1">
          {(["skim", "diagnose", "deep"] as const).map((v) => (
            <button
              key={v}
              onClick={() => setVerbosity(v)}
              className={`btn flex-1 text-xs capitalize ${
                verbosity === v
                  ? "bg-ctp-blue text-ctp-crust"
                  : "text-ctp-subtext0 hover:bg-ctp-surface0"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Panel toggles */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        <div className="mb-2 text-[10px] uppercase tracking-widest text-ctp-overlay0">
          Panels
        </div>
        <div className="space-y-1">
          {(Object.keys(panels) as (keyof PanelVisibility)[]).map((key) => (
            <label
              key={key}
              className="flex cursor-pointer items-center gap-2 rounded px-1.5 py-1 text-xs hover:bg-ctp-surface0"
            >
              <input
                type="checkbox"
                checked={panels[key]}
                onChange={() => togglePanel(key)}
                className="h-3 w-3 rounded border-ctp-surface1 accent-ctp-blue"
              />
              <span className={panels[key] ? "text-ctp-text" : "text-ctp-overlay0"}>
                {PANEL_LABELS[key]}
              </span>
            </label>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className="border-t border-ctp-surface0 px-4 py-2 text-[10px] text-ctp-overlay0">
        polling @ {usePanelStore.getState().pollingMs}ms
      </div>
    </aside>
  );
}
