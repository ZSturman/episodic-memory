"use client";

import type { MatchCandidate, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";

interface MatchPanelProps {
  candidates: MatchCandidate[];
  agentState: PanoramaAgentState;
  currentLabel: string;
  currentConfidence: number;
}

export function MatchPanel({
  candidates,
  agentState,
  currentLabel,
  currentConfidence,
}: MatchPanelProps) {
  const sorted = [...candidates].sort((a, b) => b.confidence - a.confidence);
  const top = sorted[0];
  const rest = sorted.slice(1, 8);

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Match Candidates</span>
        <span className="mono text-ctp-overlay0">{candidates.length}</span>
      </div>

      {/* Current location */}
      <div className="mb-3 rounded-md bg-ctp-surface0 p-2.5">
        <div className="flex items-center justify-between">
          <div>
            <div className="text-[10px] text-ctp-overlay0">Current</div>
            <div className="text-sm font-medium text-ctp-text">
              {currentLabel || "unknown"}
            </div>
          </div>
          <div className="text-right">
            <div
              className="badge mb-0.5"
              style={{
                backgroundColor: STATE_COLORS[agentState] + "22",
                color: STATE_COLORS[agentState],
              }}
            >
              {STATE_LABELS[agentState]}
            </div>
            <div className="mono text-xs text-ctp-blue">
              {(currentConfidence * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      </div>

      {/* Top match */}
      {top && (
        <div className="mb-2 rounded-md border border-ctp-blue/20 bg-ctp-blue/5 p-2.5">
          <div className="flex items-center justify-between">
            <span className="text-sm text-ctp-text">{top.label}</span>
            <span className="mono text-ctp-blue">
              {(top.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-ctp-surface0">
            <div
              className="h-full rounded-full bg-ctp-blue transition-all"
              style={{ width: `${top.confidence * 100}%` }}
            />
          </div>
          <div className="mt-1 text-[10px] text-ctp-overlay0">
            distance: {top.distance.toFixed(4)} · id: {top.location_id}
          </div>
        </div>
      )}

      {/* Remaining candidates */}
      {rest.length > 0 && (
        <div className="space-y-1">
          {rest.map((c) => (
            <div
              key={c.location_id}
              className="flex items-center justify-between rounded bg-ctp-surface0 px-2 py-1.5"
            >
              <span className="truncate text-xs text-ctp-subtext0">
                {c.label}
              </span>
              <div className="ml-2 flex items-center gap-2">
                <div className="h-1 w-12 overflow-hidden rounded-full bg-ctp-crust">
                  <div
                    className="h-full rounded-full bg-ctp-subtext0/50"
                    style={{ width: `${c.confidence * 100}%` }}
                  />
                </div>
                <span className="mono w-10 text-right text-[10px] text-ctp-overlay0">
                  {(c.confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          ))}
        </div>
      )}

      {candidates.length === 0 && (
        <div className="text-sm text-ctp-overlay0">No match candidates</div>
      )}
    </div>
  );
}
