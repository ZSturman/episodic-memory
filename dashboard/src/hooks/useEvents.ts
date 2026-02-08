/**
 * Polls GET /api/events?since_step=N, accumulating events incrementally.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { PanoramaEvent } from "@/lib/types";

const MAX_BUFFER = 2000;

export function useEvents() {
  const [events, setEvents] = useState<PanoramaEvent[]>([]);
  const [latestStep, setLatestStep] = useState(0);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const sinceRef = useRef(0);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await api.getEvents(sinceRef.current);
      if (res.events.length > 0) {
        sinceRef.current = res.latest_step + 1;
        setLatestStep(res.latest_step);
        setEvents((prev) => {
          const next = [...prev, ...res.events];
          return next.length > MAX_BUFFER ? next.slice(-MAX_BUFFER) : next;
        });
      }
    } catch {
      // silently retry on next tick
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    poll();
    timerRef.current = setInterval(poll, pollingMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs, paused]);

  const clear = useCallback(() => {
    setEvents([]);
    sinceRef.current = 0;
    setLatestStep(0);
  }, []);

  return { events, latestStep, clear };
}
