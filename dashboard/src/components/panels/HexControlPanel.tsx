"use client";

import { useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { HexScanData } from "@/lib/types";

interface HexControlPanelProps {
  scan: HexScanData | null;
  autoFocus: boolean;
  onAutoFocusChange: (enabled: boolean) => void;
}

/**
 * Controls for hex grid parameters:
 * - Focus center (click on grid to set)
 * - Fovea / mid / outer radii (sliders)
 * - Auto/manual toggle
 * - View mode selection (color, interest, detail, edges)
 */
export default function HexControlPanel({
  scan,
  autoFocus,
  onAutoFocusChange,
}: HexControlPanelProps) {
  const [foveaRadius, setFoveaRadius] = useState(1);
  const [midRadius, setMidRadius] = useState(3);
  const [outerRadius, setOuterRadius] = useState(6);
  const [centerQ, setCenterQ] = useState(0);
  const [centerR, setCenterR] = useState(0);

  const applyFocus = useCallback(async () => {
    try {
      await api.updateFocusProfile({
        center_q: centerQ,
        center_r: centerR,
        fovea_radius: foveaRadius,
        mid_radius: midRadius,
        outer_radius: outerRadius,
      });
    } catch {
      // ignore
    }
  }, [centerQ, centerR, foveaRadius, midRadius, outerRadius]);

  if (!scan) {
    return (
      <div className="text-xs text-ctp-overlay0">No scan data available</div>
    );
  }

  return (
    <div className="space-y-3 text-xs">
      {/* Auto/Manual toggle */}
      <div className="flex items-center justify-between">
        <span className="text-ctp-subtext0">Focus Mode</span>
        <button
          onClick={() => onAutoFocusChange(!autoFocus)}
          className={`px-2 py-1 rounded ${
            autoFocus
              ? "bg-ctp-mauve/20 text-ctp-mauve border border-ctp-mauve/30"
              : "bg-ctp-peach/20 text-ctp-peach border border-ctp-peach/30"
          }`}
        >
          {autoFocus ? "Auto" : "Manual"}
        </button>
      </div>

      {/* Scan info */}
      <div className="text-ctp-overlay0 space-y-0.5">
        <div>Cells: {scan.total_cells}</div>
        <div>Pass: {scan.scan_pass}</div>
        <div>
          Size: {scan.image_width}×{scan.image_height}
        </div>
        <div>
          Status:{" "}
          <span className={scan.converged ? "text-ctp-green" : "text-ctp-yellow"}>
            {scan.converged ? "Converged" : "Scanning"}
          </span>
        </div>
      </div>

      {/* Manual focus controls */}
      {!autoFocus && (
        <div className="space-y-2 border-t border-ctp-surface0 pt-2">
          <div>
            <label className="text-ctp-subtext0">Center (q, r)</label>
            <div className="flex gap-1 mt-1">
              <input
                type="number"
                value={centerQ}
                onChange={(e) => setCenterQ(parseInt(e.target.value) || 0)}
                className="w-16 px-1 py-0.5 rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text"
              />
              <input
                type="number"
                value={centerR}
                onChange={(e) => setCenterR(parseInt(e.target.value) || 0)}
                className="w-16 px-1 py-0.5 rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text"
              />
            </div>
          </div>

          <div>
            <label className="text-ctp-subtext0">
              Fovea radius: {foveaRadius}
            </label>
            <input
              type="range"
              min={0}
              max={5}
              value={foveaRadius}
              onChange={(e) => setFoveaRadius(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-ctp-subtext0">
              Mid radius: {midRadius}
            </label>
            <input
              type="range"
              min={1}
              max={8}
              value={midRadius}
              onChange={(e) => setMidRadius(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <div>
            <label className="text-ctp-subtext0">
              Outer radius: {outerRadius}
            </label>
            <input
              type="range"
              min={2}
              max={12}
              value={outerRadius}
              onChange={(e) => setOuterRadius(parseInt(e.target.value))}
              className="w-full"
            />
          </div>

          <button
            onClick={applyFocus}
            className="w-full py-1.5 rounded text-xs font-medium bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30"
          >
            Apply Focus
          </button>
        </div>
      )}
    </div>
  );
}
