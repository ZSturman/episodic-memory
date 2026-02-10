"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "@/lib/api";
import type { ReconstructionData, ReconstructionLayer } from "@/lib/types";

const LAYER_LABELS: Record<ReconstructionLayer, string> = {
  color: "Color",
  edges: "Edges",
  texture: "Texture",
  interest: "Interest",
  detail_level: "Detail Level",
};

const LAYER_COLORS: Record<ReconstructionLayer, string> = {
  color: "ctp-blue",
  edges: "ctp-green",
  texture: "ctp-mauve",
  interest: "ctp-peach",
  detail_level: "ctp-yellow",
};

interface ReconstructionPanelProps {
  locationId?: string;
}

export default function ReconstructionPanel({
  locationId,
}: ReconstructionPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [data, setData] = useState<ReconstructionData | null>(null);
  const [enabledLayers, setEnabledLayers] = useState<Set<ReconstructionLayer>>(
    new Set(["color"])
  );
  const [loading, setLoading] = useState(false);
  const [inputId, setInputId] = useState(locationId || "");

  const fetchReconstruction = useCallback(async (id: string) => {
    if (!id) return;
    setLoading(true);
    try {
      const resp = await api.getReconstruction(id);
      if (resp.reconstruction) {
        setData(resp.reconstruction);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (locationId) {
      setInputId(locationId);
      fetchReconstruction(locationId);
    }
  }, [locationId, fetchReconstruction]);

  const toggleLayer = (layer: ReconstructionLayer) => {
    setEnabledLayers((prev) => {
      const next = new Set(prev);
      if (next.has(layer)) {
        next.delete(layer);
      } else {
        next.add(layer);
      }
      return next;
    });
  };

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const maxW = 600;
    const scale = Math.min(1, maxW / data.image_width);
    canvas.width = data.image_width * scale;
    canvas.height = data.image_height * scale;

    ctx.fillStyle = "#1e1e2e";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    const hexSize = data.hex_size * scale;
    const sqrt3 = Math.sqrt(3);

    for (const cell of data.cells) {
      const cx = cell.center_x * scale;
      const cy = cell.center_y * scale;

      // Draw hex path
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 180) * (60 * i - 30);
        const px = cx + hexSize * Math.cos(angle);
        const py = cy + hexSize * Math.sin(angle);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.closePath();

      // Determine fill based on enabled layers
      let r = 30,
        g = 30,
        b = 46; // base dark
      let alpha = 1;

      if (enabledLayers.has("color") && cell.avg_rgb) {
        r = cell.avg_rgb[0];
        g = cell.avg_rgb[1];
        b = cell.avg_rgb[2];
      }

      if (enabledLayers.has("interest") && cell.interest_score !== undefined) {
        const t = Math.min(1, cell.interest_score);
        // Blend toward orange for high interest
        r = Math.round(r * (1 - t * 0.5) + 250 * t * 0.5);
        g = Math.round(g * (1 - t * 0.6) + 160 * t * 0.4);
      }

      if (enabledLayers.has("detail_level") && cell.detail_level !== undefined) {
        const t = cell.detail_level / 3;
        // Brighten for higher detail
        r = Math.round(r + (255 - r) * t * 0.3);
        g = Math.round(g + (255 - g) * t * 0.3);
        b = Math.round(b + (255 - b) * t * 0.3);
      }

      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
      ctx.fill();

      // Edge overlay
      if (enabledLayers.has("edges") && cell.edge_energy !== undefined) {
        const eAlpha = Math.min(0.8, cell.edge_energy);
        ctx.strokeStyle = `rgba(166, 227, 161, ${eAlpha})`; // ctp-green
        ctx.lineWidth = 1.5;
        ctx.stroke();
      } else {
        ctx.strokeStyle = "rgba(88, 91, 112, 0.3)";
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    }
  }, [data, enabledLayers]);

  return (
    <div className="space-y-3 text-xs">
      {/* Location ID input */}
      <div className="flex gap-1">
        <input
          type="text"
          value={inputId}
          onChange={(e) => setInputId(e.target.value)}
          placeholder="Location ID..."
          className="flex-1 px-2 py-1 rounded bg-ctp-surface0 border border-ctp-surface1 text-ctp-text"
        />
        <button
          onClick={() => fetchReconstruction(inputId)}
          disabled={loading || !inputId}
          className="px-2 py-1 rounded bg-ctp-blue/20 text-ctp-blue border border-ctp-blue/30 disabled:opacity-40"
        >
          {loading ? "..." : "Load"}
        </button>
      </div>

      {/* Layer toggles */}
      <div className="flex flex-wrap gap-1">
        {(Object.keys(LAYER_LABELS) as ReconstructionLayer[]).map((layer) => (
          <button
            key={layer}
            onClick={() => toggleLayer(layer)}
            className={`px-2 py-0.5 rounded text-[10px] border ${
              enabledLayers.has(layer)
                ? `bg-${LAYER_COLORS[layer]}/20 text-${LAYER_COLORS[layer]} border-${LAYER_COLORS[layer]}/30`
                : "bg-ctp-surface0 text-ctp-overlay0 border-ctp-surface1"
            }`}
          >
            {LAYER_LABELS[layer]}
          </button>
        ))}
      </div>

      {/* Canvas */}
      {data ? (
        <div className="border border-ctp-surface0 rounded overflow-hidden">
          <canvas ref={canvasRef} className="w-full" />
          <div className="px-2 py-1 bg-ctp-mantle text-ctp-overlay0">
            {data.parent_label && (
              <span>
                {data.parent_label}
                {data.variant_label && ` — ${data.variant_label}`}
              </span>
            )}
            {" · "}
            {data.cells.length} cells · {data.image_width}×{data.image_height}
          </div>
        </div>
      ) : (
        <div className="text-ctp-overlay0">
          Enter a location ID to view reconstruction
        </div>
      )}
    </div>
  );
}
