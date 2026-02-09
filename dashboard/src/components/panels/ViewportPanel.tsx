"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { FeatureSummary } from "@/lib/types";

interface ViewportPanelProps {
  viewportB64: string | null;
  step: number;
  heading: number;
  viewportIndex: number;
  totalViewports: number;
  featureSummary?: FeatureSummary | null;
  sourceFile?: string;
}

export function ViewportPanel({
  viewportB64,
  step,
  heading,
  viewportIndex,
  totalViewports,
  featureSummary,
  sourceFile,
}: ViewportPanelProps) {
  const [sourceB64, setSourceB64] = useState<string | null>(null);
  const [sourceMime, setSourceMime] = useState("image/jpeg");

  // Fetch full source image when sourceFile changes
  useEffect(() => {
    if (!sourceFile) {
      setSourceB64(null);
      return;
    }
    let cancelled = false;
    fetch("/api/source-image")
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled && data.image_b64) {
          setSourceB64(data.image_b64);
          setSourceMime(data.mime_type || "image/jpeg");
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [sourceFile, step]);

  // Compute viewport position as percentage of a 360° panorama
  const viewportPct = totalViewports > 0 ? ((viewportIndex) / totalViewports) * 100 : 0;
  const viewportWidthPct = totalViewports > 0 ? (1 / totalViewports) * 100 : 100;

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Current Viewport</span>
        <span className="mono text-ctp-overlay0">
          {viewportIndex + 1}/{totalViewports} · {heading.toFixed(0)}°
        </span>
      </div>

      {/* Full source image with viewport overlay */}
      {sourceB64 && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wide text-ctp-overlay0 mb-1">
            Source Image
          </div>
          <div className="relative aspect-[2/1] overflow-hidden rounded-md bg-ctp-surface0">
            <img
              src={`data:${sourceMime};base64,${sourceB64}`}
              alt="Full panorama source"
              className="h-full w-full object-cover"
            />
            {/* Viewport position indicator */}
            <div
              className="absolute top-0 bottom-0 border-2 border-ctp-blue/70 bg-ctp-blue/15 transition-all duration-200"
              style={{
                left: `${viewportPct}%`,
                width: `${viewportWidthPct}%`,
              }}
            />
            {/* Source file name */}
            <div className="absolute bottom-1 right-1 rounded bg-ctp-crust/80 px-1.5 py-0.5 text-[9px] text-ctp-subtext0 truncate max-w-[60%]">
              {sourceFile?.split("/").pop()}
            </div>
          </div>
        </div>
      )}

      {/* Current viewport crop */}
      <div className="relative aspect-[16/9] overflow-hidden rounded-md bg-ctp-surface0">
        {viewportB64 ? (
          <img
            src={`data:image/jpeg;base64,${viewportB64}`}
            alt={`Viewport at step ${step}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-ctp-overlay0">
            No viewport data
          </div>
        )}

        {/* Step overlay */}
        <div className="absolute bottom-2 left-2 rounded bg-ctp-crust/80 px-2 py-0.5 text-xs text-ctp-subtext0">
          Step {step}
        </div>
      </div>

      {/* Feature summary visualization */}
      {featureSummary && (
        <div className="mt-2 space-y-1.5">
          <div className="text-[10px] uppercase tracking-wide text-ctp-overlay0">
            Sensor Features
          </div>
          <div className="flex gap-2 items-center">
            {/* Brightness indicator */}
            {featureSummary.global_brightness !== undefined && (
              <div className="flex items-center gap-1">
                <div
                  className="h-4 w-4 rounded-sm border border-ctp-surface1"
                  style={{
                    backgroundColor: `hsl(0, 0%, ${Math.round(featureSummary.global_brightness * 100 / 255)}%)`,
                  }}
                />
                <span className="text-[9px] text-ctp-overlay0">
                  Bright: {featureSummary.global_brightness?.toFixed(0)}
                </span>
              </div>
            )}
            {/* Edge density indicator */}
            {featureSummary.global_edge_density !== undefined && (
              <span className="text-[9px] text-ctp-overlay0">
                Edges: {(featureSummary.global_edge_density * 100).toFixed(1)}%
              </span>
            )}
            {/* Scene signature */}
            {featureSummary.scene_signature && (
              <span className="mono text-[8px] text-ctp-overlay0 truncate max-w-[100px]">
                {featureSummary.scene_signature}
              </span>
            )}
          </div>
          {/* Dominant colors */}
          {featureSummary.dominant_colors && featureSummary.dominant_colors.length > 0 && (
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-ctp-overlay0">Colors:</span>
              {featureSummary.dominant_colors.slice(0, 5).map((c, i) => (
                <div
                  key={i}
                  className="h-3 w-3 rounded-sm border border-ctp-surface1"
                  style={{
                    backgroundColor: `rgb(${c[0]}, ${c[1]}, ${c[2]})`,
                  }}
                  title={`RGB(${c[0]}, ${c[1]}, ${c[2]})`}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
