/**
 * Polls GET /api/features for raw feature arrays.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { FeatureArrays } from "@/lib/types";

export function useFeatures() {
  const [features, setFeatures] = useState<FeatureArrays | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      setFeatures(await api.getFeatures());
    } catch {
      // retry
    }
  }, []);

  useEffect(() => {
    if (paused) return;
    poll();
    timerRef.current = setInterval(poll, pollingMs * 2);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs, paused]);

  return features;
}
