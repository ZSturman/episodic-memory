"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { EvidenceBundle, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";

interface LabelRequestPanelProps {
  agentState: PanoramaAgentState;
  evidence: EvidenceBundle | null;
  cliLabelEvent?: { label: string; step: number; timestamp: string } | null;
}

export function LabelRequestPanel({
  agentState,
  evidence,
  cliLabelEvent,
}: LabelRequestPanelProps) {
  const [label, setLabel] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<string | null>(null);

  const isRequesting = agentState === "label_request";
  const cliProvided = !!cliLabelEvent;

  const handleSubmit = async () => {
    if (!label.trim()) return;
    setSubmitting(true);
    try {
      const res = await api.submitLabel(label.trim());
      setResult(`Labeled as "${res.label}"`);
      setLabel("");
    } catch (e) {
      setResult(`Error: ${e instanceof Error ? e.message : "Unknown"}`);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div
      className={`panel ${
        isRequesting
          ? "border-ctp-red/30 ring-1 ring-ctp-red/20"
          : cliProvided
          ? "border-ctp-green/30 ring-1 ring-ctp-green/20"
          : ""
      }`}
    >
      <div className="panel-header">
        <span>Label Request</span>
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

      {/* CLI provided a label — show it */}
      {cliProvided && !isRequesting && (
        <div className="space-y-2">
          <div className="rounded-md bg-ctp-green/10 p-2.5 text-sm text-ctp-green">
            Label provided via CLI at step {cliLabelEvent!.step}
          </div>
          <div className="rounded bg-ctp-surface0 p-2 text-sm font-medium text-ctp-text">
            &quot;{cliLabelEvent!.label}&quot;
          </div>
          {/* Still allow dashboard relabeling */}
          <div className="flex gap-2">
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="Relabel from dashboard..."
              className="flex-1 rounded-md border border-ctp-surface0 bg-ctp-crust px-3 py-1.5 text-xs text-ctp-text placeholder-ctp-overlay0 focus:border-ctp-blue focus:outline-none"
              disabled={submitting}
            />
            <button
              onClick={handleSubmit}
              disabled={!label.trim() || submitting}
              className="btn-primary text-xs disabled:opacity-40"
            >
              {submitting ? "..." : "Relabel"}
            </button>
          </div>
        </div>
      )}

      {/* Dashboard label request */}
      {isRequesting && (
        <div className="space-y-3">
          <div className="rounded-md bg-ctp-red/10 p-2.5 text-sm text-ctp-red">
            Agent is requesting a label for an unknown location.
          </div>

          {/* Evidence preview thumbnails */}
          {evidence && evidence.viewport_images_b64.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
                Evidence Frames ({evidence.viewport_images_b64.length})
              </div>
              <div className="grid grid-cols-4 gap-1">
                {evidence.viewport_images_b64.slice(0, 8).map((b64, i) => (
                  <div
                    key={i}
                    className="aspect-[16/9] overflow-hidden rounded-sm bg-ctp-surface0"
                  >
                    <img
                      src={`data:image/jpeg;base64,${b64}`}
                      alt={`Evidence ${i + 1}`}
                      className="h-full w-full object-cover"
                    />
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Best candidate suggestion */}
          {evidence?.best_candidate_label && (
            <div className="rounded bg-ctp-surface0 p-2 text-xs">
              <span className="text-ctp-overlay0">Suggestion: </span>
              <button
                onClick={() => setLabel(evidence.best_candidate_label!)}
                className="text-ctp-blue hover:underline"
              >
                {evidence.best_candidate_label} (
                {(evidence.best_candidate_confidence * 100).toFixed(0)}%)
              </button>
            </div>
          )}

          {/* Input */}
          <div className="flex gap-2">
            <input
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="Enter location label..."
              className="flex-1 rounded-md border border-ctp-surface0 bg-ctp-crust px-3 py-1.5 text-sm text-ctp-text placeholder-ctp-overlay0 focus:border-ctp-blue focus:outline-none"
              disabled={submitting}
            />
            <button
              onClick={handleSubmit}
              disabled={!label.trim() || submitting}
              className="btn-primary disabled:opacity-40"
            >
              {submitting ? "..." : "Label"}
            </button>
          </div>
        </div>
      )}

      {/* Neither requesting nor CLI provided */}
      {!isRequesting && !cliProvided && (
        <div className="text-sm text-ctp-overlay0">
          No label requested. Agent is{" "}
          <span style={{ color: STATE_COLORS[agentState] }}>
            {STATE_LABELS[agentState].toLowerCase()}
          </span>
          .
        </div>
      )}

      {/* Feedback */}
      {result && (
        <div className="mt-2 rounded bg-ctp-surface0 px-2 py-1 text-xs text-ctp-green">
          {result}
        </div>
      )}
    </div>
  );
}
