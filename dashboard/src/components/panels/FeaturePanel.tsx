"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { FeatureArrays } from "@/lib/types";

interface FeaturePanelProps {
  features: FeatureArrays | null;
}

/** Re-usable mini bar chart for histogram data. */
function HistogramChart({
  data,
  color,
  label,
}: {
  data: number[];
  color: string;
  label: string;
}) {
  const chartData = data.map((v, i) => ({ bin: i, value: v }));

  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
        {label}
      </div>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: 0 }}>
          <Bar dataKey="value" radius={[1, 1, 0, 0]}>
            {chartData.map((_, i) => (
              <Cell key={i} fill={color} fillOpacity={0.7} />
            ))}
          </Bar>
          <Tooltip
            contentStyle={{
              backgroundColor: "#1e1e2e",
              border: "1px solid #313244",
              borderRadius: 6,
              fontSize: 11,
            }}
            labelStyle={{ color: "#a6adc8" }}
            formatter={(v: number) => [v.toFixed(4), "Value"]}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/** Color swatch row for dominant colors. */
function DominantColors({ colors }: { colors: number[][] }) {
  return (
    <div>
      <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
        Dominant Colors
      </div>
      <div className="flex gap-1">
        {colors.map((c, i) => (
          <div
            key={i}
            className="h-6 flex-1 rounded-sm"
            style={{
              backgroundColor: `rgb(${c[0] ?? 0}, ${c[1] ?? 0}, ${c[2] ?? 0})`,
            }}
            title={`rgb(${c.join(", ")})`}
          />
        ))}
      </div>
    </div>
  );
}

export function FeaturePanel({ features }: FeaturePanelProps) {
  if (!features) {
    return (
      <div className="panel">
        <div className="panel-header">Features</div>
        <div className="text-sm text-ctp-overlay0">No feature data</div>
      </div>
    );
  }

  return (
    <div className="panel space-y-3">
      <div className="panel-header">
        <span>Features</span>
        {features.scene_signature && (
          <span className="mono text-ctp-overlay0 truncate max-w-[120px]">
            {features.scene_signature}
          </span>
        )}
      </div>

      {features.color_histogram && features.color_histogram.length > 0 && (
        <HistogramChart
          data={features.color_histogram}
          color="#89b4fa"
          label="Color Histogram"
        />
      )}

      {features.edge_histogram && features.edge_histogram.length > 0 && (
        <HistogramChart
          data={features.edge_histogram}
          color="#a6e3a1"
          label="Edge Histogram"
        />
      )}

      {features.cell_brightness && features.cell_brightness.length > 0 && (
        <HistogramChart
          data={features.cell_brightness}
          color="#f9e2af"
          label="Cell Brightness"
        />
      )}

      {features.dominant_colors && features.dominant_colors.length > 0 && (
        <DominantColors colors={features.dominant_colors} />
      )}

      {features.cell_count !== undefined && (
        <div className="text-xs text-ctp-subtext0">
          Grid: {features.cell_count} cells
        </div>
      )}
    </div>
  );
}
