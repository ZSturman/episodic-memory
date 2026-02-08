"use client";

import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { ViewportPanel } from "@/components/panels/ViewportPanel";
import { RecentViewports } from "@/components/panels/RecentViewports";
import { FeaturePanel } from "@/components/panels/FeaturePanel";
import { EvidencePanel } from "@/components/panels/EvidencePanel";
import { MatchPanel } from "@/components/panels/MatchPanel";
import { ConfidenceTimeline } from "@/components/panels/ConfidenceTimeline";
import { MemoryListPanel } from "@/components/panels/MemoryListPanel";
import { MemoryCardDetail } from "@/components/panels/MemoryCardDetail";
import { EventLog } from "@/components/panels/EventLog";
import { LabelRequestPanel } from "@/components/panels/LabelRequestPanel";
import { LocationGraphPanel } from "@/components/panels/LocationGraphPanel";
import { SimilarityHeatmapPanel } from "@/components/panels/SimilarityHeatmapPanel";
import { EmbeddingVariancePanel } from "@/components/panels/EmbeddingVariancePanel";
import { ReplayControlsPanel } from "@/components/panels/ReplayControlsPanel";

import { useAgentState } from "@/hooks/useAgentState";
import { useEvents } from "@/hooks/useEvents";
import { useMatches } from "@/hooks/useMatches";
import { useMemories } from "@/hooks/useMemories";
import { useFeatures } from "@/hooks/useFeatures";
import { useGraph } from "@/hooks/useGraph";
import { useSimilarity } from "@/hooks/useSimilarity";
import { useReplay } from "@/hooks/useReplay";
import { usePanelStore } from "@/store/panelStore";

export default function DashboardPage() {
  const { state, connected } = useAgentState();
  const { events } = useEvents();
  const matches = useMatches();
  const { memories, detail } = useMemories();
  const features = useFeatures();
  const graph = useGraph();
  const similarity = useSimilarity();
  const { replay, loadFile, control } = useReplay();
  const panels = usePanelStore((s) => s.panels);
  const sidebarOpen = usePanelStore((s) => s.sidebarOpen);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      {sidebarOpen && <Sidebar state={state} connected={connected} />}

      {/* Main area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar state={state} connected={connected} />

        {/* Panel grid */}
        <main className="flex-1 overflow-y-auto p-4">
          {!connected && (
            <div className="mb-4 rounded-lg border border-ctp-red/30 bg-ctp-red/10 p-4 text-center text-ctp-red">
              Agent not connected. Start the panorama harness with{" "}
              <code className="rounded bg-ctp-surface0 px-1.5 py-0.5 text-xs">
                --debug-ui
              </code>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2 xl:grid-cols-3">
            {/* Row 1: Viewport + Matches + Timeline */}
            {panels.viewport && (
              <ViewportPanel
                viewportB64={state?.viewport_image_b64 ?? null}
                step={state?.step ?? 0}
                heading={state?.heading_deg ?? 0}
                viewportIndex={state?.viewport_index ?? 0}
                totalViewports={state?.total_viewports ?? 0}
              />
            )}

            {panels.matches && (
              <MatchPanel
                candidates={matches?.candidates ?? []}
                agentState={state?.agent_state ?? "investigating_unknown"}
                currentLabel={state?.location_label ?? "unknown"}
                currentConfidence={state?.location_confidence ?? 0}
              />
            )}

            {panels.confidenceTimeline && (
              <ConfidenceTimeline
                timeline={state?.confidence_timeline ?? []}
              />
            )}

            {/* Row 2: Recent viewports + Features + Evidence */}
            {panels.recentViewports && (
              <RecentViewports
                viewports={state?.recent_viewports ?? []}
              />
            )}

            {panels.features && <FeaturePanel features={features} />}

            {panels.evidence && (
              <EvidencePanel
                evidence={state?.evidence_bundle ?? null}
                agentState={state?.agent_state ?? "investigating_unknown"}
                investigationSteps={state?.investigation_steps ?? 0}
              />
            )}

            {/* Row 3: Memory list + Memory detail + Label request */}
            {panels.memoryList && (
              <MemoryListPanel memories={memories} />
            )}

            {panels.memoryDetail && (
              <MemoryCardDetail detail={detail} />
            )}

            {panels.labelRequest && (
              <LabelRequestPanel
                agentState={state?.agent_state ?? "investigating_unknown"}
                evidence={state?.evidence_bundle ?? null}
              />
            )}

            {/* Row 4: Graph + Similarity + Variance */}
            {panels.locationGraph && (
              <LocationGraphPanel graph={graph} />
            )}

            {panels.similarityHeatmap && (
              <SimilarityHeatmapPanel matrix={similarity} />
            )}

            {panels.embeddingVariance && (
              <EmbeddingVariancePanel />
            )}

            {/* Replay controls: full width */}
            {panels.replayControls && (
              <div className="col-span-full">
                <ReplayControlsPanel replay={replay} onLoad={loadFile} onControl={control} />
              </div>
            )}

            {/* Full width: Event log */}
            {panels.eventLog && (
              <div className="col-span-full">
                <EventLog events={events} />
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
