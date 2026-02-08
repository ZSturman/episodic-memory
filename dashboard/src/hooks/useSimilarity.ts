"use client";

import { useEffect, useState, useCallback } from "react";
import { api } from "@/lib/api";
import type { SimilarityMatrixResponse } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

/** Polls similarity matrix at a slow cadence (5× pollingMs). */
export function useSimilarity() {
  const [matrix, setMatrix] = useState<SimilarityMatrixResponse | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);

  const refresh = useCallback(async () => {
    try {
      const data = await api.getSimilarityMatrix();
      setMatrix(data);
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

  return matrix;
}
