"use client";

import { useState } from "react";
import type { ReplayState } from "@/lib/types";

interface Props {
  replay: ReplayState | null;
  onLoad: (filePath: string) => void;
  onControl: (action: string, params?: Record<string, unknown>) => void;
}

/**
 * ReplayControlsPanel — play/pause/seek controls for JSONL replay.
 *
 * Allows loading a historical events.jsonl file and scrubbing through
 * it frame-by-frame or at configurable speed.
 */
export function ReplayControlsPanel({ replay, onLoad, onControl }: Props) {
  const [filePath, setFilePath] = useState("");

  const loaded = replay?.loaded ?? false;
  const playing = replay?.playing ?? false;
  const cursor = replay?.cursor ?? 0;
  const total = replay?.total_events ?? 0;
  const speed = replay?.speed ?? 1.0;
  const pct = total > 0 ? Math.round((cursor / total) * 100) : 0;

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

      {/* Load file input */}
      {!loaded && (
        <div className="mb-3 flex gap-2">
          <input
            type="text"
            value={filePath}
            onChange={(e) => setFilePath(e.target.value)}
            placeholder="Path to events.jsonl…"
            className="flex-1 rounded border border-ctp-surface1 bg-ctp-crust px-2 py-1 text-xs text-ctp-text placeholder:text-ctp-overlay0 focus:border-ctp-blue focus:outline-none"
          />
          <button
            onClick={() => filePath.trim() && onLoad(filePath.trim())}
            className="rounded bg-ctp-blue px-3 py-1 text-xs font-semibold text-ctp-base hover:bg-ctp-sapphire"
          >
            Load
          </button>
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
