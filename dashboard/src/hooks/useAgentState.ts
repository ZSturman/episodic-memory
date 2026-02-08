/**
 * Polls GET /api/state at the configured interval.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { AgentState } from "@/lib/types";

export function useAgentState() {
  const [state, setState] = useState<AgentState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [connected, setConnected] = useState(false);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const s = await api.getState();
      setState(s);
      setConnected(true);
      setError(null);
    } catch (e) {
      setConnected(false);
      setError(e instanceof Error ? e.message : "Connection lost");
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    poll(); // immediate first fetch
    timerRef.current = setInterval(poll, pollingMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs, paused]);

  return { state, error, connected };
}
