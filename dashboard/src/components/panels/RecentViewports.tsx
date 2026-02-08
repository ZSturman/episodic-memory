"use client";

interface RecentViewportsProps {
  viewports: string[];
}

export function RecentViewports({ viewports }: RecentViewportsProps) {
  if (viewports.length === 0) {
    return (
      <div className="panel">
        <div className="panel-header">Recent Viewports</div>
        <div className="text-sm text-ctp-overlay0">No viewports yet</div>
      </div>
    );
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <span>Recent Viewports</span>
        <span className="mono text-ctp-overlay0">{viewports.length}</span>
      </div>

      <div className="grid grid-cols-4 gap-1.5">
        {viewports.map((b64, i) => (
          <div
            key={i}
            className="aspect-[16/9] overflow-hidden rounded-sm bg-ctp-surface0"
          >
            <img
              src={`data:image/jpeg;base64,${b64}`}
              alt={`Recent viewport ${i + 1}`}
              className="h-full w-full object-cover"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
