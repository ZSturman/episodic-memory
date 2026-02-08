"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
} from "recharts";
import type { TimelinePoint, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS } from "@/lib/types";

interface ConfidenceTimelineProps {
  timeline: TimelinePoint[];
}

export function ConfidenceTimeline({ timeline }: ConfidenceTimelineProps) {
  // Transform for Recharts
  const data = timeline.map((p) => ({
    step: p.step,
    confidence: p.confidence,
    state: p.state,
    label: p.label,
    fill: STATE_COLORS[p.state as PanoramaAgentState] ?? "#89b4fa",
  }));

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Confidence Timeline</span>
        <span className="mono text-ctp-overlay0">
          {timeline.length} pts
        </span>
      </div>

      {data.length < 2 ? (
        <div className="flex h-32 items-center justify-center text-sm text-ctp-overlay0">
          Waiting for data...
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={160}>
          <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 0 }}>
            <XAxis
              dataKey="step"
              tick={{ fontSize: 10, fill: "#6c7086" }}
              tickLine={false}
              axisLine={{ stroke: "#313244" }}
            />
            <YAxis
              domain={[0, 1]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              tick={{ fontSize: 10, fill: "#6c7086" }}
              tickLine={false}
              axisLine={false}
              width={30}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />

            {/* Threshold lines */}
            <ReferenceLine
              y={0.7}
              stroke="#a6e3a1"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
            />
            <ReferenceLine
              y={0.4}
              stroke="#f38ba8"
              strokeDasharray="4 4"
              strokeOpacity={0.4}
            />

            {/* Filled area */}
            <Area
              type="monotone"
              dataKey="confidence"
              fill="#89b4fa"
              fillOpacity={0.1}
              stroke="none"
            />

            {/* Main line */}
            <Line
              type="monotone"
              dataKey="confidence"
              stroke="#89b4fa"
              strokeWidth={1.5}
              dot={false}
              activeDot={{
                r: 3,
                fill: "#89b4fa",
                stroke: "#1e1e2e",
                strokeWidth: 1,
              }}
            />

            <Tooltip
              contentStyle={{
                backgroundColor: "#1e1e2e",
                border: "1px solid #313244",
                borderRadius: 6,
                fontSize: 11,
              }}
              labelStyle={{ color: "#a6adc8" }}
              formatter={(v: number, _: string, entry: { payload?: { label?: string; state?: string } }) => [
                `${(v * 100).toFixed(1)}% — ${entry.payload?.label ?? ""}`,
                entry.payload?.state ?? "",
              ]}
              labelFormatter={(s: number) => `Step ${s}`}
            />
          </ComposedChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}
