/**
 * Polls GET /api/hex/scan for hex grid scan data.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { HexScanData } from "@/lib/types";

export function useHexScan() {
  const [scan, setScan] = useState<HexScanData | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await api.getHexScan();
      setScan(res.scan);
    } catch {
      // retry
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

  const submitLabel = useCallback(async (payload: {
    response: string;
    parent_label?: string;
    variant_label?: string;
  }) => {
    try {
      await api.submitHexLabel(payload);
    } catch {
      // ignore
    }
  }, []);

  return { scan, submitLabel };
}
