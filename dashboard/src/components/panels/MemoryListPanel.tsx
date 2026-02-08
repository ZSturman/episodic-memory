"use client";

import type { MemorySummary } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

interface MemoryListPanelProps {
  memories: MemorySummary[];
}

export function MemoryListPanel({ memories }: MemoryListPanelProps) {
  const selectedId = usePanelStore((s) => s.selectedMemoryId);
  const setSelectedId = usePanelStore((s) => s.setSelectedMemoryId);
  const sorted = [...memories].sort(
    (a, b) => b.confidence_vs_current - a.confidence_vs_current,
  );

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Memory Locations</span>
        <span className="mono text-ctp-overlay0">{memories.length}</span>
      </div>

      {memories.length === 0 ? (
        <div className="text-sm text-ctp-overlay0">No memories yet</div>
      ) : (
        <div className="space-y-1.5 max-h-64 overflow-y-auto">
          {sorted.map((m) => {
            const active = m.location_id === selectedId;
            return (
              <button
                key={m.location_id}
                onClick={() =>
                  setSelectedId(active ? null : m.location_id)
                }
                className={`w-full rounded-md p-2.5 text-left transition-colors ${
                  active
                    ? "border border-ctp-blue/30 bg-ctp-blue/10"
                    : "bg-ctp-surface0 hover:bg-ctp-surface1"
                }`}
              >
                <div className="flex items-center justify-between">
                  <span className="text-sm font-medium text-ctp-text">
                    {m.label}
                  </span>
                  <span className="mono text-xs text-ctp-blue">
                    {(m.confidence_vs_current * 100).toFixed(0)}%
                  </span>
                </div>

                <div className="mt-1 flex items-center gap-3 text-[10px] text-ctp-overlay0">
                  <span>{m.observation_count} obs</span>
                  <span>var: {m.variance.toFixed(3)}</span>
                  <span>
                    steps {m.first_seen_step}–{m.last_seen_step}
                  </span>
                </div>

                {/* Stability bar */}
                <div className="mt-1.5 h-1 overflow-hidden rounded-full bg-ctp-crust">
                  <div
                    className="h-full rounded-full bg-ctp-green/60"
                    style={{ width: `${m.stability_score * 100}%` }}
                  />
                </div>

                {m.aliases.length > 0 && (
                  <div className="mt-1 text-[10px] text-ctp-overlay0">
                    aliases: {m.aliases.join(", ")}
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
