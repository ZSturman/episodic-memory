"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { GraphTopologyResponse } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

/** Polls location graph topology at a slow cadence (5× pollingMs). */
export function useGraph() {
  const [graph, setGraph] = useState<GraphTopologyResponse | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getGraphTopology();
      setGraph(data);
    } catch {
      /* agent may not be running */
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    refresh();
    const id = setInterval(refresh, pollingMs * 5);
    return () => clearInterval(id);
  }, [pollingMs, paused, refresh]);

  return graph;
}
