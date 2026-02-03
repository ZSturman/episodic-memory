"""Report generation tool for run analysis.

Generates:
- Human-readable text summary
- HTML report with visualizations
- Episode timeline
- Metrics overview
- Question log
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from run data.
    
    Creates both text and HTML reports from:
    - run.jsonl (step logs)
    - episodes.jsonl (frozen episodes)
    - metrics.json (computed metrics)
    - nodes.jsonl / edges.jsonl (graph data)
    """

    def __init__(self, run_dir: Path) -> None:
        """Initialize report generator.
        
        Args:
            run_dir: Path to run directory.
        """
        self.run_dir = run_dir
        self._data: dict[str, Any] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load all available data from run directory."""
        # Load metrics
        metrics_path = self.run_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                self._data["metrics"] = json.load(f)
        
        # Load episodes
        episodes_path = self.run_dir / "episodes.jsonl"
        if episodes_path.exists():
            self._data["episodes"] = []
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._data["episodes"].append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        
        # Load step logs (sample for large files)
        log_path = self.run_dir / "run.jsonl"
        if log_path.exists():
            self._data["steps"] = []
            self._data["step_count"] = 0
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    self._data["step_count"] += 1
                    # Sample every 10th step for summary
                    if self._data["step_count"] % 10 == 0:
                        try:
                            self._data["steps"].append(json.loads(line.strip()))
                        except json.JSONDecodeError:
                            continue
        
        # Load scenario result if available
        scenario_path = self.run_dir / "scenario_result.json"
        if scenario_path.exists():
            with open(scenario_path, "r", encoding="utf-8") as f:
                self._data["scenario"] = json.load(f)

    def generate_text_report(self) -> str:
        """Generate human-readable text report.
        
        Returns:
            Text report string.
        """
        lines = []
        
        # Header
        lines.append("=" * 70)
        lines.append("EPISODIC MEMORY AGENT - RUN REPORT")
        lines.append("=" * 70)
        lines.append(f"Run Directory: {self.run_dir}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Scenario info if available
        if "scenario" in self._data:
            scenario = self._data["scenario"]
            lines.append("-" * 70)
            lines.append("SCENARIO")
            lines.append("-" * 70)
            lines.append(f"Name: {scenario.get('scenario_name', 'N/A')}")
            lines.append(f"Run ID: {scenario.get('run_id', 'N/A')}")
            lines.append(f"Success: {'✓' if scenario.get('success') else '✗'}")
            if scenario.get("error"):
                lines.append(f"Error: {scenario.get('error')}")
            lines.append("")
        
        # Summary metrics
        if "metrics" in self._data:
            metrics = self._data["metrics"]
            summary = metrics.get("summary", {})
            
            lines.append("-" * 70)
            lines.append("SUMMARY")
            lines.append("-" * 70)
            lines.append(f"Total Steps: {summary.get('total_steps', 0)}")
            lines.append(f"Total Episodes: {summary.get('total_episodes', 0)}")
            lines.append(f"Duration: {summary.get('total_duration_seconds', 0):.1f} seconds")
            lines.append(f"Unique Locations: {summary.get('unique_locations', 0)}")
            lines.append(f"Events Detected: {summary.get('events_detected', 0)}")
            lines.append(f"Events Labeled: {summary.get('events_labeled', 0)}")
            lines.append(f"Final Nodes: {summary.get('final_node_count', 0)}")
            lines.append(f"Final Edges: {summary.get('final_edge_count', 0)}")
            lines.append("")
            
            # Rates
            rates = metrics.get("rates", {})
            lines.append("-" * 70)
            lines.append("RATES")
            lines.append("-" * 70)
            lines.append(f"Steps/minute: {rates.get('steps_per_minute', 0):.1f}")
            lines.append(f"Episodes/minute: {rates.get('episodes_per_minute', 0):.2f}")
            lines.append(f"Questions/minute: {rates.get('questions_per_minute', 0):.2f}")
            lines.append(f"Avg Episode Duration: {rates.get('avg_episode_duration', 0):.1f}s")
            lines.append("")
            
            # Location metrics
            location = metrics.get("location", {})
            lines.append("-" * 70)
            lines.append("LOCATION RECOGNITION")
            lines.append("-" * 70)
            lines.append(f"Coverage: {location.get('coverage', 0):.1%}")
            lines.append(f"Accuracy: {location.get('accuracy', 0):.1%}")
            lines.append(f"Location Changes: {location.get('location_changes', 0)}")
            lines.append("")
            
            # Event metrics
            event = metrics.get("event", {})
            lines.append("-" * 70)
            lines.append("EVENT DETECTION")
            lines.append("-" * 70)
            lines.append(f"Deltas Detected: {event.get('total_deltas_detected', 0)}")
            lines.append(f"Events Detected: {event.get('total_events_detected', 0)}")
            lines.append(f"Events Labeled: {event.get('total_events_labeled', 0)}")
            lines.append(f"Events Recognized: {event.get('total_events_recognized', 0)}")
            lines.append(f"Labeling Rate: {event.get('labeling_rate', 0):.1%}")
            lines.append(f"Recognition Rate: {event.get('recognition_rate', 0):.1%}")
            lines.append("")
            
            # Memory growth
            memory = metrics.get("memory", {})
            lines.append("-" * 70)
            lines.append("MEMORY GROWTH")
            lines.append("-" * 70)
            lines.append(f"Final Nodes: {memory.get('final_node_count', 0)}")
            lines.append(f"Final Edges: {memory.get('final_edge_count', 0)}")
            lines.append(f"Final Episodes: {memory.get('final_episode_count', 0)}")
            
            nodes_by_type = memory.get("nodes_by_type", {})
            if nodes_by_type:
                lines.append("Nodes by Type:")
                for node_type, count in sorted(nodes_by_type.items()):
                    lines.append(f"  - {node_type}: {count}")
            lines.append("")
        
        # Episode timeline
        if "episodes" in self._data and self._data["episodes"]:
            lines.append("-" * 70)
            lines.append("EPISODE TIMELINE")
            lines.append("-" * 70)
            
            for i, ep in enumerate(self._data["episodes"][:20]):  # Limit to 20
                location = ep.get("location_label", "?")
                entity_count = len(ep.get("entities", []))
                event_count = len(ep.get("events", []))
                boundary = ep.get("boundary_reason", "?")
                
                lines.append(
                    f"[{i+1:02d}] {location:20s} | "
                    f"entities={entity_count:2d} events={event_count:2d} | "
                    f"boundary={boundary}"
                )
            
            if len(self._data["episodes"]) > 20:
                lines.append(f"... and {len(self._data['episodes']) - 20} more episodes")
            lines.append("")
        
        # Footer
        lines.append("=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)
        
        return "\n".join(lines)

    def generate_html_report(self) -> str:
        """Generate HTML report with visualizations.
        
        Returns:
            HTML report string.
        """
        html_parts = []
        
        # HTML header
        html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Episodic Memory Agent - Run Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
            color: #333;
        }
        h1, h2, h3 { color: #2c3e50; }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .header h1 { color: white; margin: 0; }
        .header p { margin: 10px 0 0 0; opacity: 0.9; }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .metric {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-top: 5px;
        }
        .success { color: #27ae60; }
        .failure { color: #e74c3c; }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
        }
        .timeline {
            position: relative;
            padding-left: 30px;
        }
        .timeline::before {
            content: '';
            position: absolute;
            left: 10px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: #ddd;
        }
        .timeline-item {
            position: relative;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        .timeline-item::before {
            content: '';
            position: absolute;
            left: -24px;
            top: 15px;
            width: 10px;
            height: 10px;
            background: #667eea;
            border-radius: 50%;
        }
        .bar-chart {
            display: flex;
            align-items: flex-end;
            height: 150px;
            gap: 10px;
            margin-top: 15px;
        }
        .bar {
            flex: 1;
            background: linear-gradient(to top, #667eea, #764ba2);
            border-radius: 4px 4px 0 0;
            min-width: 40px;
            text-align: center;
            color: white;
            font-size: 0.8em;
            padding-top: 5px;
        }
        .bar-label {
            margin-top: 5px;
            font-size: 0.75em;
            color: #666;
        }
    </style>
</head>
<body>
""")
        
        # Header
        run_id = self.run_dir.name
        html_parts.append(f"""
<div class="header">
    <h1>🧠 Episodic Memory Agent Report</h1>
    <p>Run: {run_id} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
</div>
""")
        
        # Scenario status if available
        if "scenario" in self._data:
            scenario = self._data["scenario"]
            success_class = "success" if scenario.get("success") else "failure"
            success_icon = "✓" if scenario.get("success") else "✗"
            
            html_parts.append(f"""
<div class="card">
    <h2>📋 Scenario Result</h2>
    <div class="metric-grid">
        <div class="metric">
            <div class="metric-value {success_class}">{success_icon}</div>
            <div class="metric-label">{scenario.get('scenario_name', 'N/A')}</div>
        </div>
        <div class="metric">
            <div class="metric-value">{scenario.get('steps_completed', 0)}</div>
            <div class="metric-label">Steps Completed</div>
        </div>
        <div class="metric">
            <div class="metric-value">{scenario.get('episodes_created', 0)}</div>
            <div class="metric-label">Episodes Created</div>
        </div>
        <div class="metric">
            <div class="metric-value">{scenario.get('duration_seconds', 0):.1f}s</div>
            <div class="metric-label">Duration</div>
        </div>
    </div>
    {f'<p style="color: #e74c3c; margin-top: 15px;">Error: {scenario.get("error")}</p>' if scenario.get("error") else ''}
</div>
""")
        
        # Summary metrics
        if "metrics" in self._data:
            metrics = self._data["metrics"]
            summary = metrics.get("summary", {})
            
            html_parts.append(f"""
<div class="card">
    <h2>📊 Summary Metrics</h2>
    <div class="metric-grid">
        <div class="metric">
            <div class="metric-value">{summary.get('total_steps', 0)}</div>
            <div class="metric-label">Total Steps</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('total_episodes', 0)}</div>
            <div class="metric-label">Episodes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('unique_locations', 0)}</div>
            <div class="metric-label">Locations</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('events_detected', 0)}</div>
            <div class="metric-label">Events Detected</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('final_node_count', 0)}</div>
            <div class="metric-label">Graph Nodes</div>
        </div>
        <div class="metric">
            <div class="metric-value">{summary.get('final_edge_count', 0)}</div>
            <div class="metric-label">Graph Edges</div>
        </div>
    </div>
</div>
""")
            
            # Rates
            rates = metrics.get("rates", {})
            html_parts.append(f"""
<div class="card">
    <h2>⏱️ Performance Rates</h2>
    <div class="metric-grid">
        <div class="metric">
            <div class="metric-value">{rates.get('steps_per_minute', 0):.1f}</div>
            <div class="metric-label">Steps/min</div>
        </div>
        <div class="metric">
            <div class="metric-value">{rates.get('episodes_per_minute', 0):.2f}</div>
            <div class="metric-label">Episodes/min</div>
        </div>
        <div class="metric">
            <div class="metric-value">{rates.get('questions_per_minute', 0):.2f}</div>
            <div class="metric-label">Questions/min</div>
        </div>
        <div class="metric">
            <div class="metric-value">{rates.get('avg_episode_duration', 0):.1f}s</div>
            <div class="metric-label">Avg Episode Duration</div>
        </div>
    </div>
</div>
""")
            
            # Memory growth by type
            memory = metrics.get("memory", {})
            nodes_by_type = memory.get("nodes_by_type", {})
            if nodes_by_type:
                max_count = max(nodes_by_type.values()) if nodes_by_type else 1
                bars_html = ""
                for node_type, count in sorted(nodes_by_type.items(), key=lambda x: -x[1]):
                    height = int((count / max_count) * 100)
                    bars_html += f"""
                    <div>
                        <div class="bar" style="height: {max(height, 10)}%">{count}</div>
                        <div class="bar-label">{node_type}</div>
                    </div>
                    """
                
                html_parts.append(f"""
<div class="card">
    <h2>🧩 Graph Nodes by Type</h2>
    <div class="bar-chart">
        {bars_html}
    </div>
</div>
""")
        
        # Episode timeline
        if "episodes" in self._data and self._data["episodes"]:
            timeline_html = ""
            for i, ep in enumerate(self._data["episodes"][:15]):
                location = ep.get("location_label", "?")
                entity_count = len(ep.get("entities", []))
                event_count = len(ep.get("events", []))
                boundary = ep.get("boundary_reason", "?")
                
                # Entity labels
                entity_labels = [e.get("label", "?") for e in ep.get("entities", [])[:5]]
                entity_str = ", ".join(entity_labels) if entity_labels else "none"
                
                timeline_html += f"""
                <div class="timeline-item">
                    <strong>Episode {i+1}</strong> - {location}
                    <br><small>
                        Entities: {entity_count} ({entity_str}) |
                        Events: {event_count} |
                        Boundary: {boundary}
                    </small>
                </div>
                """
            
            remaining = len(self._data["episodes"]) - 15
            if remaining > 0:
                timeline_html += f'<div class="timeline-item"><em>... and {remaining} more episodes</em></div>'
            
            html_parts.append(f"""
<div class="card">
    <h2>📜 Episode Timeline</h2>
    <div class="timeline">
        {timeline_html}
    </div>
</div>
""")
        
        # Footer
        html_parts.append("""
<div class="card" style="text-align: center; color: #666;">
    <p>Generated by Episodic Memory Agent Report Tool</p>
    <p><small>No external services required - runs locally</small></p>
</div>
</body>
</html>
""")
        
        return "".join(html_parts)

    def save_reports(self) -> tuple[Path, Path]:
        """Generate and save both reports.
        
        Returns:
            Tuple of (text_path, html_path).
        """
        # Generate reports
        text_report = self.generate_text_report()
        html_report = self.generate_html_report()
        
        # Save text report
        text_path = self.run_dir / "report.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text_report)
        
        # Save HTML report
        html_path = self.run_dir / "report.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_report)
        
        return text_path, html_path


def generate_report(run_dir: Path) -> tuple[str, Path]:
    """Generate reports for a run.
    
    Args:
        run_dir: Path to run directory.
        
    Returns:
        Tuple of (text_report, html_path).
    """
    generator = ReportGenerator(run_dir)
    text_path, html_path = generator.save_reports()
    
    text_report = generator.generate_text_report()
    
    logger.info(f"Reports saved to {text_path} and {html_path}")
    
    return text_report, html_path
