"use client";

import { useState } from "react";
import type { HexScanData, AgentControlStatus } from "@/lib/types";

interface LabelConfirmPanelProps {
  scan: HexScanData | null;
  control: AgentControlStatus | null;
  onConfirm: () => void;
  onReject: () => void;
  onNewLabel: (parent: string, variant: string) => void;
  onVariant: (variant: string) => void;
  onSkip: () => void;
}

/**
 * Hypothesis display + user confirmation panel.
 * Shows the agent's best guess for the current image location
 * and lets the user confirm, reject, provide a new label, or skip.
 */
export default function LabelConfirmPanel({
  scan,
  control,
  onConfirm,
  onReject,
  onNewLabel,
  onVariant,
  onSkip,
}: LabelConfirmPanelProps) {
  const isAwaiting = control?.state === "awaiting_user";

  // Extract hypothesis from scan data if available
  const hypothesis = scan?.hypothesis;
  const matchedLabel = hypothesis?.parent_label || "unknown";
  const matchedVariant = hypothesis?.variant_label || "";
  const confidence = hypothesis?.confidence ?? 0;
  const locationId = hypothesis?.location_id;

  if (!isAwaiting) {
    return (
      <div className="text-xs text-ctp-overlay0 space-y-2">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-ctp-surface2" />
          <span>
            {control?.state === "scanning_image"
              ? "Scanning image..."
              : control?.state === "paused"
              ? "Agent paused"
              : "Waiting for scan..."}
          </span>
        </div>
        {scan && (
          <div className="text-[10px] text-ctp-surface2">
            Pass {scan.scan_pass} · {scan.total_cells} cells ·{" "}
            {scan.converged ? "Converged" : "In progress"}
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-3 text-xs">
      {/* Hypothesis display */}
      <div className="p-2 rounded bg-ctp-surface0 border border-ctp-surface1 space-y-1">
        <div className="text-ctp-subtext0 font-medium">Agent hypothesis</div>
        {locationId ? (
          <>
            <div className="text-ctp-text text-sm font-semibold">
              {matchedLabel}
              {matchedVariant && (
                <span className="text-ctp-subtext0 font-normal">
                  {" "}— {matchedVariant}
                </span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <div className="flex-1 h-1.5 rounded-full bg-ctp-surface1 overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    confidence > 0.7
                      ? "bg-ctp-green"
                      : confidence > 0.4
                      ? "bg-ctp-yellow"
                      : "bg-ctp-red"
                  }`}
                  style={{ width: `${confidence * 100}%` }}
                />
              </div>
              <span className="text-ctp-overlay0 w-10 text-right">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="text-[10px] text-ctp-surface2 truncate">
              ID: {locationId}
            </div>
          </>
        ) : (
          <div className="text-ctp-yellow">New / unknown location</div>
        )}
      </div>

      {/* Action buttons */}
      <div className="space-y-1.5">
        {locationId && (
          <>
            <button
              onClick={onConfirm}
              className="w-full py-1.5 rounded font-medium bg-ctp-green/20 text-ctp-green border border-ctp-green/30"
            >
              ✓ Confirm — this is {matchedLabel}
            </button>

            {/* Same place, different variant */}
            <VariantInput
              parentLabel={matchedLabel}
              onSubmit={onVariant}
            />

            <button
              onClick={onReject}
              className="w-full py-1.5 rounded font-medium bg-ctp-red/20 text-ctp-red border border-ctp-red/30"
            >
              ✗ Wrong — this isn't {matchedLabel}
            </button>
          </>
        )}

        {/* New label */}
        <NewLabelInput onSubmit={onNewLabel} />

        <button
          onClick={onSkip}
          className="w-full py-1 rounded text-ctp-overlay0 bg-ctp-surface0 border border-ctp-surface1"
        >
          Skip image
        </button>
      </div>
    </div>
  );
}

/* ---- Sub-components ---- */

function VariantInput({
  parentLabel,
  onSubmit,
}: {
  parentLabel: string;
  onSubmit: (variant: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [variant, setVariant] = useState("");

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="w-full py-1.5 rounded font-medium bg-ctp-mauve/20 text-ctp-mauve border border-ctp-mauve/30"
      >
        Same place, different view
      </button>
    );
  }

  return (
    <div className="flex gap-1">
      <input
        type="text"
        value={variant}
        onChange={(e) => setVariant(e.target.value)}
        placeholder={`${parentLabel} — variant...`}
        className="flex-1 px-2 py-1 rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text"
        autoFocus
      />
      <button
        onClick={() => {
          if (variant.trim()) {
            onSubmit(variant.trim());
            setVariant("");
            setOpen(false);
          }
        }}
        className="px-2 py-1 rounded bg-ctp-mauve/20 text-ctp-mauve border border-ctp-mauve/30"
      >
        Set
      </button>
    </div>
  );
}

function NewLabelInput({
  onSubmit,
}: {
  onSubmit: (parent: string, variant: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [parent, setParent] = useState("");
  const [variant, setVariant] = useState("");

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="w-full py-1.5 rounded font-medium bg-ctp-peach/20 text-ctp-peach border border-ctp-peach/30"
      >
        + New label
      </button>
    );
  }

  return (
    <div className="space-y-1 p-2 rounded bg-ctp-surface0 border border-ctp-surface1">
      <input
        type="text"
        value={parent}
        onChange={(e) => setParent(e.target.value)}
        placeholder="Location name (e.g. kitchen)"
        className="w-full px-2 py-1 rounded bg-ctp-mantle border border-ctp-surface1 text-ctp-text"
        autoFocus
      />
      <input
        type="text"
        value={variant}
        onChange={(e) => setVariant(e.target.value)}
        placeholder="Variant (optional, e.g. near sink)"
        className="w-full px-2 py-1 rounded bg-ctp-mantle border border-ctp-surface1 text-ctp-text"
      />
      <div className="flex gap-1">
        <button
          onClick={() => {
            if (parent.trim()) {
              onSubmit(parent.trim(), variant.trim());
              setParent("");
              setVariant("");
              setOpen(false);
            }
          }}
          className="flex-1 py-1 rounded bg-ctp-peach/20 text-ctp-peach border border-ctp-peach/30"
        >
          Create
        </button>
        <button
          onClick={() => setOpen(false)}
          className="px-2 py-1 rounded bg-ctp-surface0 text-ctp-overlay0 border border-ctp-surface1"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}
