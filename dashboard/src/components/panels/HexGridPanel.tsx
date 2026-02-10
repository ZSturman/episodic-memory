"use client";

import { useCallback, useRef, useEffect, useMemo } from "react";
import type { HexScanData, HexCellData } from "@/lib/types";

interface HexGridPanelProps {
  scan: HexScanData | null;
  colorMode?: "color" | "interest" | "detail" | "edges";
  onCellClick?: (cell: HexCellData) => void;
}

const DETAIL_COLORS = [
  "rgba(100, 100, 100, 0.3)", // 0 = skip
  "rgba(137, 180, 250, 0.5)", // 1 = coarse
  "rgba(166, 227, 161, 0.6)", // 2 = standard
  "rgba(250, 179, 135, 0.8)", // 3 = fine
];

/**
 * Renders the hex grid as a canvas overlay.
 *
 * Each hex cell is drawn as a hexagon colored by the selected mode:
 * - color: average RGB from scan
 * - interest: yellow→red heat map
 * - detail: blue/green/orange by detail level
 * - edges: gray→white by edge energy
 */
export default function HexGridPanel({
  scan,
  colorMode = "color",
  onCellClick,
}: HexGridPanelProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Scale factor to fit the canvas
  const scale = useMemo(() => {
    if (!scan) return 1;
    const maxDim = Math.max(scan.image_width, scan.image_height);
    return Math.min(600 / maxDim, 1);
  }, [scan]);

  const drawHex = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      cx: number,
      cy: number,
      size: number,
      fillColor: string,
      strokeColor: string,
    ) => {
      ctx.beginPath();
      for (let i = 0; i < 6; i++) {
        const angle = (Math.PI / 180) * (60 * i + 30); // pointy-top
        const x = cx + size * Math.cos(angle);
        const y = cy + size * Math.sin(angle);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    },
    [],
  );

  const getCellColor = useCallback(
    (cell: HexCellData): string => {
      switch (colorMode) {
        case "color": {
          if (cell.avg_rgb.length >= 3) {
            const [r, g, b] = cell.avg_rgb.map((v) => Math.round(v * 255));
            return `rgb(${r}, ${g}, ${b})`;
          }
          return "rgba(80, 80, 80, 0.3)";
        }
        case "interest": {
          const t = Math.min(cell.interest_score, 1);
          const r = Math.round(255 * t);
          const g = Math.round(255 * (1 - t) * 0.8);
          return `rgba(${r}, ${g}, 40, ${0.3 + t * 0.6})`;
        }
        case "detail":
          return DETAIL_COLORS[cell.detail_level] ?? DETAIL_COLORS[0];
        case "edges": {
          const e = Math.min(cell.edge_energy, 1);
          const v = Math.round(60 + 195 * e);
          return `rgba(${v}, ${v}, ${v}, ${0.3 + e * 0.6})`;
        }
        default:
          return "rgba(80, 80, 80, 0.3)";
      }
    },
    [colorMode],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !scan || !scan.cells.length) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = Math.round(scan.image_width * scale);
    const h = Math.round(scan.image_height * scale);
    canvas.width = w;
    canvas.height = h;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#181825";
    ctx.fillRect(0, 0, w, h);

    const hexSize = scan.hex_size * scale;

    for (const cell of scan.cells) {
      const cx = cell.center_x * scale;
      const cy = cell.center_y * scale;
      const fillColor = getCellColor(cell);
      const strokeColor = "rgba(205, 214, 244, 0.15)";
      drawHex(ctx, cx, cy, hexSize * 0.95, fillColor, strokeColor);
    }

    // Highlight focus center
    const focusCell = scan.cells.find(
      (c) => c.q === scan.focus_center_q && c.r === scan.focus_center_r,
    );
    if (focusCell) {
      const cx = focusCell.center_x * scale;
      const cy = focusCell.center_y * scale;
      ctx.beginPath();
      ctx.arc(cx, cy, hexSize * 0.3, 0, Math.PI * 2);
      ctx.strokeStyle = "#f38ba8";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }, [scan, scale, getCellColor, drawHex]);

  const handleClick = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>) => {
      if (!scan || !onCellClick) return;
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / scale;
      const y = (e.clientY - rect.top) / scale;

      // Find closest cell
      let closest: HexCellData | null = null;
      let minDist = Infinity;
      for (const cell of scan.cells) {
        const dx = cell.center_x - x;
        const dy = cell.center_y - y;
        const dist = dx * dx + dy * dy;
        if (dist < minDist) {
          minDist = dist;
          closest = cell;
        }
      }
      if (closest && minDist < scan.hex_size * scan.hex_size * 2) {
        onCellClick(closest);
      }
    },
    [scan, scale, onCellClick],
  );

  if (!scan) {
    return (
      <div className="flex items-center justify-center h-48 text-ctp-overlay0 text-sm">
        No hex scan data
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs text-ctp-overlay0">
          {scan.total_cells} cells · Pass {scan.scan_pass}
          {scan.converged && (
            <span className="ml-2 text-ctp-green">Converged</span>
          )}
        </div>
      </div>
      <canvas
        ref={canvasRef}
        onClick={handleClick}
        className="rounded border border-ctp-surface0 cursor-crosshair"
        style={{ maxWidth: "100%" }}
      />
    </div>
  );
}
