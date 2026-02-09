/**
 * Polls GET /api/memories for memory summaries.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { MemorySummary, MemoryCard } from "@/lib/types";

export function useMemories() {
  const [memories, setMemories] = useState<MemorySummary[]>([]);
  const [detail, setDetail] = useState<MemoryCard | null>(null);
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const paused = usePanelStore((s) => s.paused);
  const selectedId = usePanelStore((s) => s.selectedMemoryId);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const poll = useCallback(async () => {
    try {
      const res = await api.getMemories();
      setMemories(res.memories);
    } catch {
      // retry
    }
  }, []);

  // Fetch detail when selectedId changes or memories list updates
  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    let cancelled = false;
    const fetchDetail = () => {
      api.getMemoryCard(selectedId).then((d) => {
        if (!cancelled) setDetail(d);
      }).catch(() => {
        if (!cancelled) setDetail(null);
      });
    };
    fetchDetail();
    // Re-fetch detail on same cadence as the memory list
    const detailTimer = setInterval(fetchDetail, pollingMs * 5);
    return () => { cancelled = true; clearInterval(detailTimer); };
  }, [selectedId, pollingMs]);

  useEffect(() => {
    if (paused) return;
    poll();
    timerRef.current = setInterval(poll, pollingMs * 5); // slower cadence for list
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs, paused]);

  return { memories, detail };
}
