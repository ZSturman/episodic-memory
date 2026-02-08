"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { MemoryCard } from "@/lib/types";

interface MemoryCardDetailProps {
  detail: MemoryCard | null;
}

export function MemoryCardDetail({ detail }: MemoryCardDetailProps) {
  if (!detail) {
    return (
      <div className="panel">
        <div className="panel-header">Memory Detail</div>
        <div className="text-sm text-ctp-overlay0">
          Select a memory from the list
        </div>
      </div>
    );
  }

  const histData = detail.match_confidence_history.map((h) => ({
    step: h.step,
    confidence: h.confidence,
  }));

  return (
    <div className="panel space-y-3">
      <div className="panel-header">
        <span>Memory Card</span>
        <span className="mono text-ctp-overlay0 truncate max-w-[140px]">
          {detail.location_id}
        </span>
      </div>

      {/* Title & label */}
      <div className="rounded-md bg-ctp-surface0 p-3">
        <div className="text-lg font-semibold text-ctp-text">
          {detail.label}
        </div>
        {detail.aliases.length > 0 && (
          <div className="mt-0.5 text-xs text-ctp-overlay0">
            aliases: {detail.aliases.join(", ")}
          </div>
        )}
      </div>

      {/* Stats grid */}
      <div className="grid grid-cols-2 gap-2">
        {[
          ["Observations", detail.observation_count],
          ["Variance", detail.variance.toFixed(4)],
          ["Stability", (detail.stability_score * 100).toFixed(1) + "%"],
          [
            "First/Last",
            `${detail.first_seen_step}–${detail.last_seen_step}`,
          ],
        ].map(([label, value]) => (
          <div key={label as string} className="rounded bg-ctp-surface0 px-2 py-1.5 text-center">
            <div className="text-[10px] text-ctp-overlay0">{label}</div>
            <div className="mono text-sm text-ctp-text">{value}</div>
          </div>
        ))}
      </div>

      {/* Confidence history chart */}
      {histData.length > 1 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
            Match Confidence Over Time
          </div>
          <ResponsiveContainer width="100%" height={100}>
            <LineChart
              data={histData}
              margin={{ top: 4, right: 4, bottom: 4, left: 0 }}
            >
              <XAxis
                dataKey="step"
                tick={{ fontSize: 9, fill: "#6c7086" }}
                tickLine={false}
                axisLine={{ stroke: "#313244" }}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fontSize: 9, fill: "#6c7086" }}
                tickLine={false}
                axisLine={false}
                width={28}
              />
              <Line
                type="monotone"
                dataKey="confidence"
                stroke="#a6e3a1"
                strokeWidth={1.5}
                dot={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e1e2e",
                  border: "1px solid #313244",
                  borderRadius: 6,
                  fontSize: 11,
                }}
                formatter={(v: number) => [
                  (v * 100).toFixed(1) + "%",
                  "Confidence",
                ]}
                labelFormatter={(s: number) => `Step ${s}`}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Co-occurring entities */}
      {detail.co_occurring_entities.length > 0 && (
        <div>
          <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
            Co-occurring Entities
          </div>
          <div className="flex flex-wrap gap-1">
            {detail.co_occurring_entities.map((e) => (
              <span
                key={e}
                className="badge bg-ctp-surface0 text-ctp-subtext0"
              >
                {e}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Embedding centroid preview (norm) */}
      <div className="text-xs text-ctp-overlay0">
        Embedding dim: {detail.embedding_centroid.length} ·{" "}
        Norm: {Math.sqrt(
          detail.embedding_centroid.reduce((s, v) => s + v * v, 0),
        ).toFixed(4)}
      </div>
    </div>
  );
}
