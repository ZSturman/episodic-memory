"use client";

import type { GraphTopologyResponse } from "@/lib/types";
import { usePanelStore } from "@/store/panelStore";

interface Props {
  graph: GraphTopologyResponse | null;
}

/**
 * LocationGraphPanel — node-link display of discovered locations.
 *
 * Renders a simple topological view using SVG.  Each node is a circle
 * sized by observation_count; edges are lines colored by type.
 */
export function LocationGraphPanel({ graph }: Props) {
  const setSelectedMemoryId = usePanelStore((s) => s.setSelectedMemoryId);

  if (!graph || graph.nodes.length === 0) {
    return (
      <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
        <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
          Location Graph
        </h3>
        <p className="text-xs text-ctp-overlay0">No locations discovered yet.</p>
      </div>
    );
  }

  // Layout: arrange nodes in a circle
  const W = 400;
  const H = 300;
  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(cx, cy) - 40;
  const n = graph.nodes.length;

  const positions = graph.nodes.map((_, i) => ({
    x: cx + radius * Math.cos((2 * Math.PI * i) / n - Math.PI / 2),
    y: cy + radius * Math.sin((2 * Math.PI * i) / n - Math.PI / 2),
  }));

  const nodeIdToIdx: Record<string, number> = {};
  graph.nodes.forEach((nd, i) => {
    nodeIdToIdx[nd.location_id] = i;
  });

  // Edge colors by type
  const edgeColor = (type: string) => {
    switch (type) {
      case "similar_to":
        return "#89b4fa";
      case "temporal":
        return "#a6e3a1";
      case "spatial":
        return "#f9e2af";
      default:
        return "#585b70";
    }
  };

  return (
    <div className="rounded-lg border border-ctp-surface0 bg-ctp-mantle p-4">
      <h3 className="mb-2 text-sm font-semibold text-ctp-subtext1">
        Location Graph
        <span className="ml-2 text-xs font-normal text-ctp-overlay0">
          {graph.nodes.length} nodes · {graph.edges.length} edges
        </span>
      </h3>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full rounded border border-ctp-surface0 bg-ctp-crust"
      >
        {/* Edges */}
        {graph.edges.map((edge) => {
          const si = nodeIdToIdx[edge.source];
          const ti = nodeIdToIdx[edge.target];
          if (si === undefined || ti === undefined) return null;
          return (
            <line
              key={edge.edge_id}
              x1={positions[si].x}
              y1={positions[si].y}
              x2={positions[ti].x}
              y2={positions[ti].y}
              stroke={edgeColor(edge.edge_type)}
              strokeWidth={Math.max(1, edge.weight * 2)}
              strokeOpacity={0.5}
            />
          );
        })}

        {/* Nodes */}
        {graph.nodes.map((nd, i) => {
          const r = Math.max(8, Math.min(20, 6 + nd.observation_count * 0.5));
          return (
            <g
              key={nd.location_id}
              className="cursor-pointer"
              onClick={() => setSelectedMemoryId(nd.location_id)}
            >
              <circle
                cx={positions[i].x}
                cy={positions[i].y}
                r={r}
                fill="#89b4fa"
                fillOpacity={0.7 + nd.activation * 0.3}
                stroke="#cdd6f4"
                strokeWidth={1}
              />
              <text
                x={positions[i].x}
                y={positions[i].y + r + 12}
                textAnchor="middle"
                fill="#cdd6f4"
                fontSize={9}
                className="pointer-events-none"
              >
                {nd.label.length > 12
                  ? nd.label.slice(0, 12) + "…"
                  : nd.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
