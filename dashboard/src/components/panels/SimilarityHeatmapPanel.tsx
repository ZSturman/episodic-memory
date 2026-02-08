"use client";

import type { SimilarityMatrixResponse } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

interface Props {
  matrix: SimilarityMatrixResponse | null;
}

/**
 * SimilarityHeatmapPanel — NxN heatmap of pairwise cosine similarity
 * between all discovered location embeddings.
 *
 * Uses a green→yellow→red gradient.  Click a cell to select one of
 * the involved locations for detail.
 */
export function SimilarityHeatmapPanel({ matrix }: Props) {
  const setSelectedMemoryId = usePanelStore((s) => s.setSelectedMemoryId);

  if (!matrix || matrix.locations.length === 0) {
    return (
      <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
        <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
          Similarity Heatmap
        </h3>
        <p className="text-xs text-ctp-overlay0">
          No locations available for comparison.
        </p>
      </div>
    );
  }

  const n = matrix.locations.length;
  const cellSize = Math.max(20, Math.min(48, 300 / n));

  /** Map similarity ∈ [0,1] → hsl color. */
  const simColor = (v: number) => {
    // 0 → red (0°), 0.5 → yellow (60°), 1.0 → green (120°)
    const hue = Math.round(v * 120);
    return `hsl(${hue}, 70%, 45%)`;
  };

  return (
    <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
      <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
        Similarity Heatmap
        <span className="ml-2 text-xs font-normal text-ctp-overlay0">
          {n}×{n} locations
        </span>
      </h3>

      <div className="overflow-x-auto">
        <div
          className="inline-grid gap-px"
          style={{
            gridTemplateColumns: `80px repeat(${n}, ${cellSize}px)`,
          }}
        >
          {/* Header row */}
          <div /> {/* empty corner */}
          {matrix.labels.map((lbl, j) => (
            <div
              key={`hdr-${j}`}
              className="truncate text-center text-[9px] text-ctp-subtext0"
              style={{ width: cellSize }}
              title={lbl}
            >
              {lbl.length > 6 ? lbl.slice(0, 6) + "…" : lbl}
            </div>
          ))}

          {/* Data rows */}
          {matrix.matrix.map((row, i) => (
            <>
              {/* Row label */}
              <div
                key={`rl-${i}`}
                className="flex items-center truncate pr-1 text-right text-[9px] text-ctp-subtext0"
                style={{ width: 80 }}
                title={matrix.labels[i]}
              >
                {matrix.labels[i].length > 10
                  ? matrix.labels[i].slice(0, 10) + "…"
                  : matrix.labels[i]}
              </div>
              {row.map((sim, j) => (
                <div
                  key={`c-${i}-${j}`}
                  className="flex cursor-pointer items-center justify-center border border-ctp-crust text-[8px] font-mono text-ctp-base"
                  style={{
                    width: cellSize,
                    height: cellSize,
                    backgroundColor: simColor(sim),
                  }}
                  title={`${matrix.labels[i]} ↔ ${matrix.labels[j]}: ${sim.toFixed(3)}`}
                  onClick={() =>
                    setSelectedMemoryId(matrix.locations[j])
                  }
                >
                  {n <= 8 ? sim.toFixed(2) : ""}
                </div>
              ))}
            </>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="mt-2 flex items-center gap-2 text-[9px] text-ctp-overlay0">
        <span>Low</span>
        <div
          className="h-2 w-24 rounded"
          style={{
            background:
              "linear-gradient(to right, hsl(0,70%,45%), hsl(60,70%,45%), hsl(120,70%,45%))",
          }}
        />
        <span>High</span>
      </div>
    </div>
  );
}
