/**
 * Polls GET /api/matches for current match candidates.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { MatchesResponse } from "@/lib/types";

export function useMatches() {
  const [data, setData] = useState<MatchesResponse | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      setData(await api.getMatches());
    } catch {
      // retry
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    poll();
    timerRef.current = setInterval(poll, pollingMs * 2); // slower cadence
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs, paused]);

  return data;
}
