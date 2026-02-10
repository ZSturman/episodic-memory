/**
 * Polls GET /api/control/status for agent control state.
 * Provides methods to pause, step, and advance.
 */
"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import { usePanelStore } from "@/store/panelStore";
import type { AgentControlStatus } from "@/lib/types";

export function useAgentControl() {
  const [status, setStatus] = useState<AgentControlStatus>({
    paused: false,
    awaiting_user: false,
    auto_focus: true,
    user_response: "pending",
    state: "idle",
  });
  const pollingMs = usePanelStore((s) => s.pollingMs);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const deriveState = (raw: Omit<AgentControlStatus, "state">): AgentControlStatus => {
    let state: AgentControlStatus["state"] = "idle";
    if (raw.paused) state = "paused";
    else if (raw.awaiting_user) state = "awaiting_user";
    else state = "scanning_image";
    return { ...raw, state };
  };

  const poll = useCallback(async () => {
    try {
      const raw = await api.getControlStatus();
      setStatus(deriveState(raw));
    } catch {
      // retry
    }
  }, []);

  useEffect(() => {
    poll();
    timerRef.current = setInterval(poll, pollingMs);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [poll, pollingMs]);

  const togglePause = useCallback(async () => {
    try {
      const res = await api.togglePause();
      setStatus((prev) => ({ ...prev, paused: res.paused }));
    } catch {
      // ignore
    }
  }, []);

  const requestStep = useCallback(async () => {
    try {
      await api.requestStep();
    } catch {
      // ignore
    }
  }, []);

  const advanceImage = useCallback(async () => {
    try {
      await api.advanceImage();
    } catch {
      // ignore
    }
  }, []);

  const setAutoFocus = useCallback(async (enabled: boolean) => {
    try {
      await api.setAutoFocus(enabled);
      setStatus((prev) => ({ ...prev, auto_focus: enabled }));
    } catch {
      // ignore
    }
  }, []);

  return {
    status,
    togglePause,
    requestStep,
    advanceImage,
    setAutoFocus,
  };
}
