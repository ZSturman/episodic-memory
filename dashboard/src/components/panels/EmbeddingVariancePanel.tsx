"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { EmbeddingVarianceResponse } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

/**
 * EmbeddingVariancePanel — per-dimension bar chart of embedding
 * variance for the currently selected memory.
 *
 * Shows which dimensions are stable (low variance) vs noisy (high).
 */
export function EmbeddingVariancePanel() {
  const selectedId = usePanelStore((s) => s.selectedMemoryId);
  const [data, setData] = useState<EmbeddingVarianceResponse | null>(null);

  useEffect(() => {
    if (!selectedId) {
      setData(null);
      return;
    }
    let cancelled = false;
    api.getEmbeddingVariance(selectedId).then((d) => {
      if (!cancelled) setData(d);
    }).catch(() => {
      if (!cancelled) setData(null);
    });
    return () => { cancelled = true; };
  }, [selectedId]);

  if (!selectedId) {
    return (
      <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
        <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
          Embedding Variance
        </h3>
        <p className="text-xs text-ctp-overlay0">
          Select a memory to inspect its embedding variance.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
        <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
          Embedding Variance
        </h3>
        <p className="text-xs text-ctp-overlay0">Loading…</p>
      </div>
    );
  }

  const maxVar = Math.max(...data.variance_per_dim, 0.001);
  // Group dimensions into buckets for visual clarity
  const dims = data.variance_per_dim;
  const barWidth = Math.max(2, Math.min(6, 400 / dims.length));

  return (
    <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
      <h3 className="mb-1 text-sm font-semibold text-ctp-subtext1">
        Embedding Variance
        <span className="ml-2 text-xs font-normal text-ctp-overlay0">
          {data.dimensions} dims · Σ = {data.total_variance.toFixed(4)} · n = {data.observation_count}
        </span>
      </h3>

      {/* Bar chart */}
      <div className="mt-2 flex items-end gap-px overflow-x-auto" style={{ height: 80 }}>
        {dims.map((v, i) => {
          const h = Math.max(1, (v / maxVar) * 72);
          // Color: low variance → green, high → red
          const hue = Math.round((1 - v / maxVar) * 120);
          return (
            <div
              key={i}
              className="flex-shrink-0"
              style={{
                width: barWidth,
                height: h,
                backgroundColor: `hsl(${hue}, 60%, 50%)`,
                alignSelf: "flex-end",
              }}
              title={`dim ${i}: ${v.toFixed(6)}`}
            />
          );
        })}
      </div>

      {/* Axis labels */}
      <div className="mt-1 flex justify-between text-[8px] text-ctp-overlay0">
        <span>dim 0</span>
        <span>dim {dims.length - 1}</span>
      </div>
    </div>
  );
}
