"use client";

import {
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Area,
  ComposedChart,
  Line,
  Scatter,
} from "recharts";
import type { TimelinePoint, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS, STATE_LABELS } from "@/lib/types";

interface ConfidenceTimelineProps {
  timeline: TimelinePoint[];
}

export function ConfidenceTimeline({ timeline }: ConfidenceTimelineProps) {
  // Transform for Recharts — add fill color per state
  const data = timeline.map((p, i) => ({
    step: p.step,
    confidence: p.confidence,
    state: p.state,
    label: p.label,
    fill: STATE_COLORS[p.state as PanoramaAgentState] ?? "#89b4fa",
    stroke: STATE_COLORS[p.state as PanoramaAgentState] ?? "#89b4fa",
    // Mark state transitions
    isTransition:
      i > 0 && timeline[i - 1].state !== p.state ? p.confidence : undefined,
  }));

  // Find state transition steps for vertical markers
  const transitions: { step: number; state: string; label: string }[] = [];
  for (let i = 1; i < timeline.length; i++) {
    if (timeline[i].state !== timeline[i - 1].state) {
      transitions.push({
        step: timeline[i].step,
        state: timeline[i].state,
        label: STATE_LABELS[timeline[i].state as PanoramaAgentState] ?? timeline[i].state,
      });
    }
  }

  // Current state info
  const currentState = timeline.length > 0 ? timeline[timeline.length - 1].state : null;
  const currentColor = currentState
    ? STATE_COLORS[currentState as PanoramaAgentState] ?? "#89b4fa"
    : "#89b4fa";

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Confidence Timeline</span>
        <div className="flex items-center gap-2">
          {currentState && (
            <span
              className="badge text-[9px]"
              style={{
                backgroundColor: currentColor + "22",
                color: currentColor,
              }}
            >
              {STATE_LABELS[currentState as PanoramaAgentState] ?? currentState}
            </span>
          )}
          <span className="mono text-ctp-overlay0">
            {timeline.length} pts
          </span>
        </div>
      </div>

      {data.length < 2 ? (
        <div className="flex h-32 items-center justify-center text-sm text-ctp-overlay0">
          Waiting for data...
        </div>
      ) : (
        <>
          <ResponsiveContainer width="100%" height={180}>
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
                strokeOpacity={0.5}
                label={{
                  value: "confident",
                  position: "right",
                  fill: "#a6e3a1",
                  fontSize: 8,
                }}
              />
              <ReferenceLine
                y={0.4}
                stroke="#f38ba8"
                strokeDasharray="4 4"
                strokeOpacity={0.5}
                label={{
                  value: "label req",
                  position: "right",
                  fill: "#f38ba8",
                  fontSize: 8,
                }}
              />

              {/* State transition vertical markers */}
              {transitions.map((t, i) => (
                <ReferenceLine
                  key={`t-${i}`}
                  x={t.step}
                  stroke={
                    STATE_COLORS[t.state as PanoramaAgentState] ?? "#585b70"
                  }
                  strokeDasharray="2 3"
                  strokeOpacity={0.5}
                />
              ))}

              {/* Filled area under curve */}
              <Area
                type="monotone"
                dataKey="confidence"
                fill={currentColor}
                fillOpacity={0.08}
                stroke="none"
              />

              {/* Main confidence line */}
              <Line
                type="monotone"
                dataKey="confidence"
                stroke={currentColor}
                strokeWidth={1.5}
                dot={false}
                activeDot={{
                  r: 3,
                  fill: currentColor,
                  stroke: "#1e1e2e",
                  strokeWidth: 1,
                }}
              />

              {/* State transition dots */}
              <Scatter
                dataKey="isTransition"
                fill="#cdd6f4"
                stroke="#1e1e2e"
                strokeWidth={1}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e1e2e",
                  border: "1px solid #313244",
                  borderRadius: 6,
                  fontSize: 11,
                }}
                labelStyle={{ color: "#a6adc8" }}
                formatter={(
                  v: number,
                  name: string,
                  entry: { payload?: { label?: string; state?: string } },
                ) => {
                  if (name === "isTransition") return [null, null];
                  const st = entry.payload?.state ?? "";
                  const lbl = entry.payload?.label ?? "";
                  const stateLabel =
                    STATE_LABELS[st as PanoramaAgentState] ?? st;
                  return [
                    `${(v * 100).toFixed(1)}%  ·  ${lbl}`,
                    stateLabel,
                  ];
                }}
                labelFormatter={(s: number) => `Step ${s}`}
              />
            </ComposedChart>
          </ResponsiveContainer>

          {/* State legend */}
          <div className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 px-1">
            {Object.entries(STATE_COLORS).map(([state, color]) => (
              <div key={state} className="flex items-center gap-1 text-[9px]">
                <div
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: color }}
                />
                <span className="text-ctp-overlay0">
                  {STATE_LABELS[state as PanoramaAgentState]}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
