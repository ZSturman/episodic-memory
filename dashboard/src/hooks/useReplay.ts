"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { ReplayState } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

/** Polls replay controller state. */
export function useReplay() {
  const [replay, setReplay] = useState<ReplayState | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getReplayState();
      setReplay(data);
    } catch {
      /* agent may not be running */
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    refresh();
    const id = setInterval(refresh, pollingMs * 2);
    return () => clearInterval(id);
  }, [pollingMs, paused, refresh]);

  const loadFile = useCallback(async (filePath: string) => {
    try {
      await api.loadReplay(filePath);
      await refresh();
    } catch (e) {
      console.error("Failed to load replay:", e);
    }
  }, [refresh]);

  const control = useCallback(
    async (action: string, params?: Record<string, unknown>) => {
      try {
        await api.replayControl(action, params);
        await refresh();
      } catch (e) {
        console.error("Replay control failed:", e);
      }
    },
    [refresh],
  );

  return { replay, loadFile, control };
}
