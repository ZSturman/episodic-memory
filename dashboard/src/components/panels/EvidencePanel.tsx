"use client";

import type { EvidenceBundle, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";

interface EvidencePanelProps {
  evidence: EvidenceBundle | null;
  agentState: PanoramaAgentState;
  investigationSteps: number;
}

export function EvidencePanel({
  evidence,
  agentState,
  investigationSteps,
}: EvidencePanelProps) {
  const isInvestigating =
    agentState === "investigating_unknown" ||
    agentState === "novel_location_candidate";

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Evidence Bundle</span>
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

      {isInvestigating && (
        <div className="mb-3">
          {/* Progress bar for investigation window */}
          <div className="mb-1 flex justify-between text-xs text-ctp-subtext0">
            <span>Investigation progress</span>
            <span>{investigationSteps} steps</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full bg-ctp-surface0">
            <div
              className="h-full rounded-full bg-ctp-mauve transition-all"
              style={{
                width: `${Math.min(100, (investigationSteps / 20) * 100)}%`,
              }}
            />
          </div>
        </div>
      )}

      {!evidence ? (
        <div className="text-sm text-ctp-overlay0">
          {isInvestigating ? "Accumulating evidence..." : "No active investigation"}
        </div>
      ) : (
        <div className="space-y-3">
          {/* Best candidate */}
          {evidence.best_candidate_label && (
            <div className="rounded-md bg-ctp-surface0 p-2.5">
              <div className="text-xs text-ctp-overlay0">Best Candidate</div>
              <div className="mt-0.5 flex items-center justify-between">
                <span className="text-sm font-medium text-ctp-text">
                  {evidence.best_candidate_label}
                </span>
                <span className="mono text-ctp-blue">
                  {(evidence.best_candidate_confidence * 100).toFixed(1)}%
                </span>
              </div>
            </div>
          )}

          {/* Stats row */}
          <div className="grid grid-cols-3 gap-2 text-center">
            <div className="rounded bg-ctp-surface0 px-2 py-1.5">
              <div className="text-[10px] text-ctp-overlay0">Steps</div>
              <div className="mono text-sm text-ctp-text">
                {evidence.investigation_steps}
              </div>
            </div>
            <div className="rounded bg-ctp-surface0 px-2 py-1.5">
              <div className="text-[10px] text-ctp-overlay0">Margin</div>
              <div className="mono text-sm text-ctp-text">
                {evidence.margin.toFixed(3)}
              </div>
            </div>
            <div className="rounded bg-ctp-surface0 px-2 py-1.5">
              <div className="text-[10px] text-ctp-overlay0">Images</div>
              <div className="mono text-sm text-ctp-text">
                {evidence.viewport_images_b64.length}
              </div>
            </div>
          </div>

          {/* Match scores */}
          {Object.keys(evidence.match_scores).length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
                Match Scores
              </div>
              <div className="space-y-1">
                {Object.entries(evidence.match_scores)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 5)
                  .map(([loc, score]) => (
                    <div
                      key={loc}
                      className="flex items-center justify-between rounded bg-ctp-surface0 px-2 py-1"
                    >
                      <span className="truncate text-xs text-ctp-text">
                        {loc}
                      </span>
                      <span className="mono ml-2 text-xs text-ctp-blue">
                        {(score * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}

          {/* Confidence history sparkline */}
          {evidence.confidence_history.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
                Confidence History ({evidence.confidence_history.length} steps)
              </div>
              <div className="flex h-8 items-end gap-px">
                {evidence.confidence_history.map((c, i) => (
                  <div
                    key={i}
                    className="flex-1 rounded-t-sm bg-ctp-blue/60"
                    style={{ height: `${c * 100}%` }}
                  />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
