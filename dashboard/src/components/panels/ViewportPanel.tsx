"use client";

interface ViewportPanelProps {
  viewportB64: string | null;
  step: number;
  heading: number;
  viewportIndex: number;
  totalViewports: number;
}

export function ViewportPanel({
  viewportB64,
  step,
  heading,
  viewportIndex,
  totalViewports,
}: ViewportPanelProps) {
  return (
    <div className="panel">
      <div className="panel-header">
        <span>Current Viewport</span>
        <span className="mono text-ctp-overlay0">
          {viewportIndex + 1}/{totalViewports} · {heading.toFixed(0)}°
        </span>
      </div>

      <div className="relative aspect-[16/9] overflow-hidden rounded-md bg-ctp-surface0">
        {viewportB64 ? (
          <img
            src={`data:image/jpeg;base64,${viewportB64}`}
            alt={`Viewport at step ${step}`}
            className="h-full w-full object-cover"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-ctp-overlay0">
            No viewport data
          </div>
        )}

        {/* Step overlay */}
        <div className="absolute bottom-2 left-2 rounded bg-ctp-crust/80 px-2 py-0.5 text-xs text-ctp-subtext0">
          Step {step}
        </div>
      </div>
    </div>
  );
}
