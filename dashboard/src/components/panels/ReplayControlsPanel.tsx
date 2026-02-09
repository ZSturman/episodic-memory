"use client";

import { useState, useEffect } from "react";
import { api } from "@/lib/api";
import type { ReplayState } from "@/lib/types";

interface RunEntry {
  name: string;
  file: string;
  size_bytes: number;
  modified: string;
}

interface Props {
  replay: ReplayState | null;
  onLoad: (filePath: string) => void;
  onControl: (action: string, params?: Record<string, unknown>) => void;
}

/**
 * ReplayControlsPanel — play/pause/seek controls for JSONL replay.
 *
 * Auto-lists available run directories and allows loading their events
 * for frame-by-frame or configurable-speed playback.
 */
export function ReplayControlsPanel({ replay, onLoad, onControl }: Props) {
  const [filePath, setFilePath] = useState("");
  const [runs, setRuns] = useState<RunEntry[]>([]);
  const [runsLoading, setRunsLoading] = useState(false);

  // Auto-fetch available runs on mount
  useEffect(() => {
    setRunsLoading(true);
    api.getReplayRuns()
      .then((data) => setRuns(data.runs))
      .catch(() => setRuns([]))
      .finally(() => setRunsLoading(false));
  }, []);

  const loaded = replay?.loaded ?? false;
  const playing = replay?.playing ?? false;
  const cursor = replay?.cursor ?? 0;
  const total = replay?.total_events ?? 0;
  const speed = replay?.speed ?? 1.0;
  const pct = total > 0 ? Math.round((cursor / total) * 100) : 0;

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes}B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)}MB`;
  };

  return (
    <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
      <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
        Replay Controls
        {loaded && (
          <span className="ml-2 text-xs font-normal text-ctp-green">
            ● Loaded
          </span>
        )}
      </h3>

      {/* Run selection */}
      {!loaded && (
        <div className="mb-3 space-y-2">
          {/* Available runs dropdown */}
          {runs.length > 0 && (
            <div>
              <div className="mb-1 text-[10px] uppercase tracking-wide text-ctp-overlay0">
                Available Runs ({runs.length})
              </div>
              <div className="max-h-40 space-y-1 overflow-y-auto">
                {runs.map((run) => (
                  <button
                    key={run.file}
                    onClick={() => {
                      setFilePath(run.file);
                      onLoad(run.file);
                    }}
                    className={`w-full rounded-md px-2.5 py-1.5 text-left transition-colors ${
                      filePath === run.file
                        ? "bg-ctp-blue/15 border border-ctp-blue/30"
                        : "bg-ctp-surface0 hover:bg-ctp-surface1"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-medium text-ctp-text">
                        {run.name}
                      </span>
                      <span className="text-[10px] text-ctp-overlay0">
                        {formatSize(run.size_bytes)}
                      </span>
                    </div>
                    <div className="text-[9px] text-ctp-overlay0 truncate">
                      {run.modified}
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
          {runsLoading && (
            <div className="text-xs text-ctp-overlay0">Loading runs...</div>
          )}
          {!runsLoading && runs.length === 0 && (
            <div className="text-xs text-ctp-overlay0">
              No previous runs found in runs/ directory
            </div>
          )}

          {/* Manual path input fallback */}
          <div className="flex gap-2">
            <input
              type="text"
              value={filePath}
              onChange={(e) => setFilePath(e.target.value)}
              placeholder="Or enter path to events.jsonl…"
              className="flex-1 rounded border border-ctp-surface1 bg-ctp-crust px-2 py-1 text-xs text-ctp-text placeholder:text-ctp-overlay0 focus:border-ctp-blue focus:outline-none"
            />
            <button
              onClick={() => filePath.trim() && onLoad(filePath.trim())}
              className="rounded bg-ctp-blue px-3 py-1 text-xs font-semibold text-ctp-base hover:bg-ctp-sapphire"
            >
              Load
            </button>
          </div>
        </div>
      )}

      {/* Playback controls */}
      {loaded && (
        <>
          {/* Progress bar */}
          <div className="mb-2">
            <div className="flex items-center justify-between text-[10px] text-ctp-overlay0">
              <span>Event {cursor} / {total}</span>
              <span>{pct}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={total}
              value={cursor}
              onChange={(e) =>
                onControl("seek", { position: parseInt(e.target.value) })
              }
              className="mt-1 w-full accent-ctp-blue"
            />
          </div>

          {/* Buttons */}
          <div className="mb-2 flex flex-wrap gap-1.5">
            <button
              onClick={() => onControl(playing ? "pause" : "play")}
              className="rounded bg-ctp-surface1 px-2 py-1 text-xs text-ctp-text hover:bg-ctp-surface2"
            >
              {playing ? "⏸ Pause" : "▶ Play"}
            </button>
            <button
              onClick={() => onControl("step_back")}
              className="rounded bg-ctp-surface1 px-2 py-1 text-xs text-ctp-text hover:bg-ctp-surface2"
            >
              ⏮ Step ←
            </button>
            <button
              onClick={() => onControl("step_forward")}
              className="rounded bg-ctp-surface1 px-2 py-1 text-xs text-ctp-text hover:bg-ctp-surface2"
            >
              Step → ⏭
            </button>
            <button
              onClick={() => onControl("stop")}
              className="rounded bg-ctp-surface1 px-2 py-1 text-xs text-ctp-text hover:bg-ctp-surface2"
            >
              ⏹ Reset
            </button>
          </div>

          {/* Speed control */}
          <div className="flex items-center gap-2 text-xs text-ctp-overlay0">
            <span>Speed:</span>
            {[0.25, 0.5, 1, 2, 5, 10].map((s) => (
              <button
                key={s}
                onClick={() => onControl("set_speed", { speed: s })}
                className={`rounded px-1.5 py-0.5 text-[10px] ${
                  Math.abs(speed - s) < 0.01
                    ? "bg-ctp-blue text-ctp-base"
                    : "bg-ctp-surface0 text-ctp-subtext0 hover:bg-ctp-surface1"
                }`}
              >
                {s}×
              </button>
            ))}
          </div>

          {/* File info */}
          <div className="mt-2 truncate text-[9px] text-ctp-overlay0">
            {replay?.file}
          </div>
        </>
      )}
    </div>
  );
}
