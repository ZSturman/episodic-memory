"use client";

import { useState } from "react";
import { api } from "@/lib/api";
import type { AgentControlStatus, UserResponseType } from "@/lib/types";

interface AgentControlPanelProps {
  status: AgentControlStatus;
  onTogglePause: () => void;
  onStep: () => void;
  onAdvance: () => void;
  onAutoFocus: (enabled: boolean) => void;
}

/**
 * Dashboard panel for controlling the agent:
 * - Pause / Resume
 * - Single step (while paused)
 * - Advance to next image
 * - Auto/Manual focus toggle
 * - Label submission (when awaiting user)
 */
export default function AgentControlPanel({
  status,
  onTogglePause,
  onStep,
  onAdvance,
  onAutoFocus,
}: AgentControlPanelProps) {
  const [labelInput, setLabelInput] = useState("");
  const [variantInput, setVariantInput] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const submitLabel = async (response: UserResponseType) => {
    setSubmitting(true);
    try {
      await api.submitHexLabel({
        response,
        parent_label: labelInput,
        variant_label: variantInput,
      });
      setLabelInput("");
      setVariantInput("");
    } catch {
      // ignore
    }
    setSubmitting(false);
  };

  return (
    <div className="space-y-3">
      {/* Playback controls */}
      <div className="flex items-center gap-2">
        <button
          onClick={onTogglePause}
          className={`px-3 py-1.5 rounded text-xs font-medium ${
            status.paused
              ? "bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
              : "bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
          }`}
        >
          {status.paused ? "▶ Resume" : "⏸ Pause"}
        </button>

        <button
          onClick={onStep}
          disabled={!status.paused}
          className="px-3 py-1.5 rounded text-xs font-medium bg-ctp-surface0 text-ctp-text border border-ctp-surface1 disabled:opacity-30"
        >
          ⏭ Step
        </button>

        <button
          onClick={onAdvance}
          className="px-3 py-1.5 rounded text-xs font-medium bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30"
        >
          ⏩ Next Image
        </button>
      </div>

      {/* Focus mode toggle */}
      <div className="flex items-center gap-2 text-xs">
        <span className="text-ctp-overlay0">Focus:</span>
        <button
          onClick={() => onAutoFocus(!status.auto_focus)}
          className={`px-2 py-1 rounded text-xs ${
            status.auto_focus
              ? "bg-ctp-mauve/20 text-ctp-mauve border border-ctp-mauve/30"
              : "bg-ctp-peach/20 text-ctp-peach border border-ctp-peach/30"
          }`}
        >
          {status.auto_focus ? "Auto" : "Manual"}
        </button>
      </div>

      {/* Status indicator */}
      <div className="text-xs text-ctp-overlay0 space-y-1">
        <div>
          Status:{" "}
          <span
            className={
              status.awaiting_user ? "text-ctp-pink font-medium" : "text-ctp-text"
            }
          >
            {status.awaiting_user
              ? "Awaiting label"
              : status.paused
                ? "Paused"
                : "Running"}
          </span>
        </div>
      </div>

      {/* Label submission (visible when awaiting user) */}
      {status.awaiting_user && (
        <div className="border border-ctp-pink/30 rounded p-3 space-y-2">
          <div className="text-xs font-medium text-ctp-pink">Label Required</div>

          <input
            type="text"
            value={labelInput}
            onChange={(e) => setLabelInput(e.target.value)}
            placeholder="Parent label (e.g. office)"
            className="w-full px-2 py-1 text-xs rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text placeholder:text-ctp-overlay0"
          />

          <input
            type="text"
            value={variantInput}
            onChange={(e) => setVariantInput(e.target.value)}
            placeholder="Variant (optional)"
            className="w-full px-2 py-1 text-xs rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text placeholder:text-ctp-overlay0"
          />

          <div className="flex gap-1 flex-wrap">
            <button
              onClick={() => submitLabel("confirm")}
              disabled={submitting}
              className="px-2 py-1 text-xs rounded bg-ctp-green/20 text-ctp-green border border-ctp-green/30 disabled:opacity-50"
            >
              ✓ Confirm
            </button>
            <button
              onClick={() => submitLabel("new_label")}
              disabled={submitting || !labelInput}
              className="px-2 py-1 text-xs rounded bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 disabled:opacity-50"
            >
              + New Label
            </button>
            <button
              onClick={() => submitLabel("same_place_different")}
              disabled={submitting || !variantInput}
              className="px-2 py-1 text-xs rounded bg-ctp-mauve/20 text-ctp-mauve border border-ctp-mauve/30 disabled:opacity-50"
            >
              ~ Variant
            </button>
            <button
              onClick={() => submitLabel("reject")}
              disabled={submitting}
              className="px-2 py-1 text-xs rounded bg-ctp-red/20 text-ctp-red border border-ctp-red/30 disabled:opacity-50"
            >
              ✗ Reject
            </button>
            <button
              onClick={() => submitLabel("skip")}
              disabled={submitting}
              className="px-2 py-1 text-xs rounded bg-ctp-surface0 text-ctp-overlay0 border border-ctp-surface1 disabled:opacity-50"
            >
              Skip
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
