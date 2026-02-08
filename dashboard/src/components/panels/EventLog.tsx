"use client";

import { useState, useMemo } from "react";
import type { PanoramaEvent, PanoramaEventType, PanoramaAgentState } from "@/lib/types";
import { STATE_COLORS } from "@/lib/types";

const EVENT_TYPE_COLORS: Record<PanoramaEventType, string> = {
  perception_update: "#89dceb",
  match_evaluation: "#89b4fa",
  state_transition: "#cba6f7",
  investigation_window: "#f9e2af",
  label_request: "#f38ba8",
  memory_write: "#a6e3a1",
};

interface EventLogProps {
  events: PanoramaEvent[];
}

export function EventLog({ events }: EventLogProps) {
  const [typeFilter, setTypeFilter] = useState<PanoramaEventType | "all">("all");
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const filtered = useMemo(() => {
    const base =
      typeFilter === "all"
        ? events
        : events.filter((e) => e.event_type === typeFilter);
    return base.slice(-200).reverse(); // newest first, cap at 200
  }, [events, typeFilter]);

  const toggle = (step: number) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(step) ? next.delete(step) : next.add(step);
      return next;
    });
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Event Log</span>
        <span className="mono text-ctp-overlay0">{events.length} total</span>
      </div>

      {/* Filter tabs */}
      <div className="mb-3 flex flex-wrap gap-1">
        <button
          onClick={() => setTypeFilter("all")}
          className={`btn text-[10px] ${
            typeFilter === "all"
              ? "bg-ctp-surface1 text-ctp-text"
              : "text-ctp-overlay0 hover:bg-ctp-surface0"
          }`}
        >
          All
        </button>
        {(Object.keys(EVENT_TYPE_COLORS) as PanoramaEventType[]).map((t) => (
          <button
            key={t}
            onClick={() => setTypeFilter(t)}
            className={`btn text-[10px] ${
              typeFilter === t
                ? "text-ctp-text"
                : "text-ctp-overlay0 hover:bg-ctp-surface0"
            }`}
            style={
              typeFilter === t
                ? {
                    backgroundColor: EVENT_TYPE_COLORS[t] + "22",
                    color: EVENT_TYPE_COLORS[t],
                  }
                : undefined
            }
          >
            {t.replace("_", " ")}
          </button>
        ))}
      </div>

      {/* Event list */}
      <div className="max-h-80 space-y-1 overflow-y-auto">
        {filtered.length === 0 ? (
          <div className="text-sm text-ctp-overlay0">No events</div>
        ) : (
          filtered.map((e, i) => {
            const isOpen = expanded.has(e.step * 1000 + i);
            return (
              <div
                key={`${e.step}-${i}`}
                className="rounded bg-ctp-surface0 text-xs"
              >
                <button
                  onClick={() => toggle(e.step * 1000 + i)}
                  className="flex w-full items-center gap-2 px-2.5 py-1.5 text-left"
                >
                  {/* Type dot */}
                  <div
                    className="h-1.5 w-1.5 flex-shrink-0 rounded-full"
                    style={{
                      backgroundColor:
                        EVENT_TYPE_COLORS[e.event_type] ?? "#6c7086",
                    }}
                  />

                  {/* Step */}
                  <span className="mono w-10 flex-shrink-0 text-ctp-overlay0">
                    #{e.step}
                  </span>

                  {/* Type */}
                  <span
                    className="flex-shrink-0"
                    style={{
                      color: EVENT_TYPE_COLORS[e.event_type] ?? "#6c7086",
                    }}
                  >
                    {e.event_type}
                  </span>

                  {/* State badge */}
                  <span
                    className="badge ml-auto flex-shrink-0"
                    style={{
                      backgroundColor:
                        STATE_COLORS[e.state as PanoramaAgentState] + "22",
                      color: STATE_COLORS[e.state as PanoramaAgentState],
                    }}
                  >
                    {e.state}
                  </span>

                  <span className="text-ctp-overlay0">{isOpen ? "▾" : "▸"}</span>
                </button>

                {isOpen && (
                  <div className="border-t border-ctp-crust px-2.5 py-2">
                    <pre className="max-h-48 overflow-auto whitespace-pre-wrap text-[10px] text-ctp-subtext0">
                      {JSON.stringify(e.payload, null, 2)}
                    </pre>
                    {e.timestamp && (
                      <div className="mt-1 text-[10px] text-ctp-overlay0">
                        {e.timestamp}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
