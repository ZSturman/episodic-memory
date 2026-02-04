"""Visualization utilities for episodic memory agent runs.

This module provides functions to generate visualizations from run data.
Supports both matplotlib for static plots and simple HTML for interactive viewing.

Usage:
    # From command line
    python -m episodic_agent.visualize runs/20260202_174416
    
    # From code
    from episodic_agent.visualize import RunVisualizer
    viz = RunVisualizer("runs/20260202_174416")
    viz.plot_memory_growth()
    viz.generate_html_report()
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from episodic_agent.metrics.logging import LogAnalyzer


# =============================================================================
# Run Visualizer
# =============================================================================

class RunVisualizer:
    """Visualizer for episodic memory agent run data.
    
    Generates plots and reports from JSONL log files.
    """

    def __init__(self, run_path: str | Path) -> None:
        """Initialize visualizer with run directory.
        
        Args:
            run_path: Path to run directory containing run.jsonl.
        """
        self._run_path = Path(run_path)
        self._log_path = self._run_path / "run.jsonl"
        self._viz_path = self._run_path / "visualization_data.json"
        
        if not self._log_path.exists():
            raise FileNotFoundError(f"Run log not found: {self._log_path}")
        
        self._analyzer = LogAnalyzer(self._log_path)
        self._data: dict[str, Any] = {}
        self._loaded = False

    def load(self) -> "RunVisualizer":
        """Load and analyze run data.
        
        Returns:
            Self for method chaining.
        """
        self._analyzer.load()
        self._data = self._analyzer.generate_plot_data()
        
        # Also load visualization data if available
        if self._viz_path.exists():
            with open(self._viz_path) as f:
                self._viz_data = json.load(f)
        else:
            self._viz_data = {}
        
        self._loaded = True
        return self

    @property
    def step_count(self) -> int:
        """Total number of steps in the run."""
        if not self._loaded:
            self.load()
        return len(self._data.get("steps", []))

    @property
    def episode_count(self) -> int:
        """Total number of episodes in the run."""
        if not self._loaded:
            self.load()
        return len(self._data.get("boundaries", []))

    def print_summary(self) -> None:
        """Print run summary to console."""
        if not self._loaded:
            self.load()
        
        print(f"\n{'='*60}")
        print(f"Run Summary: {self._run_path.name}")
        print(f"{'='*60}")
        print(f"Total Steps: {self.step_count}")
        print(f"Total Episodes: {self.episode_count}")
        
        loc_dist = self._data.get("location_distribution", {})
        print(f"\nLocation Distribution:")
        for loc, count in sorted(loc_dist.items(), key=lambda x: -x[1]):
            pct = count / self.step_count * 100 if self.step_count > 0 else 0
            print(f"  {loc}: {count} steps ({pct:.1f}%)")
        
        boundaries = self._data.get("boundaries", [])
        if boundaries:
            print(f"\nBoundary Triggers:")
            reasons = {}
            for b in boundaries:
                reason = b.get("reason", "unknown")
                reasons[reason] = reasons.get(reason, 0) + 1
            for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
                print(f"  {reason}: {count}")
        
        print(f"{'='*60}\n")

    def export_csv(self, output_path: Path | None = None) -> Path:
        """Export run data to CSV for external analysis.
        
        Args:
            output_path: Output path. Defaults to run_dir/analysis.csv.
            
        Returns:
            Path to exported CSV file.
        """
        if output_path is None:
            output_path = self._run_path / "analysis.csv"
        
        self._analyzer.export_csv(output_path)
        print(f"Exported CSV to: {output_path}")
        return output_path

    def generate_html_report(self, output_path: Path | None = None) -> Path:
        """Generate interactive HTML report with embedded charts.
        
        Uses simple JavaScript charting for visualization without
        requiring matplotlib or other heavy dependencies.
        
        Args:
            output_path: Output path. Defaults to run_dir/report.html.
            
        Returns:
            Path to generated HTML file.
        """
        if not self._loaded:
            self.load()
        
        if output_path is None:
            output_path = self._run_path / "report.html"
        
        html = self._generate_html_content()
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"Generated HTML report: {output_path}")
        return output_path

    def _generate_html_content(self) -> str:
        """Generate HTML report content."""
        # Prepare data for charts
        steps = self._data.get("steps", [])
        entity_counts = [e[1] for e in self._data.get("entity_counts", [])]
        event_counts = [e[1] for e in self._data.get("event_counts", [])]
        location_confidence = [e[1] for e in self._data.get("location_confidence", [])]
        
        memory_nodes = [e[1] for e in self._data.get("memory_nodes", [])]
        memory_edges = [e[1] for e in self._data.get("memory_edges", [])]
        memory_episodes = [e[1] for e in self._data.get("memory_episodes", [])]
        
        loc_dist = self._data.get("location_distribution", {})
        boundaries = self._data.get("boundaries", [])
        
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Episodic Memory Agent - Run Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --text-primary: #eaeaea;
            --text-secondary: #b8b8b8;
            --accent-blue: #4fc3f7;
            --accent-green: #81c784;
            --accent-orange: #ffb74d;
            --accent-purple: #ba68c8;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        h1 {{
            text-align: center;
            margin-bottom: 10px;
            color: var(--accent-blue);
        }}
        
        .subtitle {{
            text-align: center;
            color: var(--text-secondary);
            margin-bottom: 30px;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: var(--accent-blue);
        }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.9em;
            margin-top: 5px;
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .chart-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 10px;
        }}
        
        .chart-title {{
            margin-bottom: 15px;
            color: var(--accent-green);
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
        }}
        
        .table-container {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        
        th {{
            color: var(--accent-purple);
        }}
        
        tr:hover {{
            background: rgba(255,255,255,0.05);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Episodic Memory Agent</h1>
        <p class="subtitle">Run: {self._run_path.name}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{self.step_count}</div>
                <div class="stat-label">Total Steps</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.episode_count}</div>
                <div class="stat-label">Episodes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(loc_dist)}</div>
                <div class="stat-label">Unique Locations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{sum(event_counts) if event_counts else 0}</div>
                <div class="stat-label">Total Events</div>
            </div>
        </div>
        
        <div class="chart-grid">
            <div class="chart-card">
                <h3 class="chart-title">Memory Growth Over Time</h3>
                <div class="chart-container">
                    <canvas id="memoryChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3 class="chart-title">Entity & Event Counts</h3>
                <div class="chart-container">
                    <canvas id="countsChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3 class="chart-title">Location Confidence</h3>
                <div class="chart-container">
                    <canvas id="confidenceChart"></canvas>
                </div>
            </div>
            
            <div class="chart-card">
                <h3 class="chart-title">Location Distribution</h3>
                <div class="chart-container">
                    <canvas id="locationChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="table-container">
            <h3 class="chart-title">Episode Boundaries</h3>
            <table>
                <thead>
                    <tr>
                        <th>Step</th>
                        <th>Location</th>
                        <th>Reason</th>
                        <th>Episode #</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td>{b.get("step", "")}</td>
                        <td>{b.get("location", "")}</td>
                        <td>{b.get("reason", "")}</td>
                        <td>{b.get("episode_count", "")}</td>
                    </tr>
                    ''' for b in boundaries)}
                </tbody>
            </table>
        </div>
    </div>
    
    <script>
        const chartColors = {{
            blue: '#4fc3f7',
            green: '#81c784',
            orange: '#ffb74d',
            purple: '#ba68c8',
            red: '#ef5350',
        }};
        
        // Memory Growth Chart
        new Chart(document.getElementById('memoryChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(list(range(1, len(memory_nodes) + 1)))},
                datasets: [
                    {{
                        label: 'Nodes',
                        data: {json.dumps(memory_nodes)},
                        borderColor: chartColors.blue,
                        tension: 0.1,
                        fill: false,
                    }},
                    {{
                        label: 'Edges',
                        data: {json.dumps(memory_edges)},
                        borderColor: chartColors.green,
                        tension: 0.1,
                        fill: false,
                    }},
                    {{
                        label: 'Episodes',
                        data: {json.dumps(memory_episodes)},
                        borderColor: chartColors.purple,
                        tension: 0.1,
                        fill: false,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
        
        // Counts Chart
        new Chart(document.getElementById('countsChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(steps)},
                datasets: [
                    {{
                        label: 'Entities',
                        data: {json.dumps(entity_counts)},
                        borderColor: chartColors.orange,
                        tension: 0.1,
                        fill: false,
                    }},
                    {{
                        label: 'Events',
                        data: {json.dumps(event_counts)},
                        borderColor: chartColors.red,
                        tension: 0.1,
                        fill: false,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
        
        // Confidence Chart
        new Chart(document.getElementById('confidenceChart'), {{
            type: 'line',
            data: {{
                labels: {json.dumps(steps)},
                datasets: [
                    {{
                        label: 'Location Confidence',
                        data: {json.dumps(location_confidence)},
                        borderColor: chartColors.green,
                        backgroundColor: 'rgba(129, 199, 132, 0.2)',
                        tension: 0.1,
                        fill: true,
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 1 }}
                }}
            }}
        }});
        
        // Location Distribution Chart
        new Chart(document.getElementById('locationChart'), {{
            type: 'doughnut',
            data: {{
                labels: {json.dumps(list(loc_dist.keys()))},
                datasets: [{{
                    data: {json.dumps(list(loc_dist.values()))},
                    backgroundColor: [
                        chartColors.blue,
                        chartColors.green,
                        chartColors.orange,
                        chartColors.purple,
                        chartColors.red,
                    ]
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
            }}
        }});
    </script>
</body>
</html>'''


# =============================================================================
# Matplotlib Plotting (Optional)
# =============================================================================

def plot_with_matplotlib(run_path: str | Path, output_path: Path | None = None) -> None:
    """Generate matplotlib plots if available.
    
    Args:
        run_path: Path to run directory.
        output_path: Optional output path for PNG file.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Use 'pip install matplotlib' for static plots.")
        return
    
    viz = RunVisualizer(run_path)
    viz.load()
    data = viz._data
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Episodic Memory Agent - {Path(run_path).name}", fontsize=14)
    
    # Memory growth
    ax1 = axes[0, 0]
    memory_nodes = data.get("memory_nodes", [])
    memory_edges = data.get("memory_edges", [])
    memory_episodes = data.get("memory_episodes", [])
    if memory_nodes:
        steps_m = [x[0] for x in memory_nodes]
        ax1.plot(steps_m, [x[1] for x in memory_nodes], label="Nodes", color="#4fc3f7")
        ax1.plot(steps_m, [x[1] for x in memory_edges], label="Edges", color="#81c784")
        ax1.plot(steps_m, [x[1] for x in memory_episodes], label="Episodes", color="#ba68c8")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Count")
    ax1.set_title("Memory Growth")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Entity/Event counts
    ax2 = axes[0, 1]
    entity_counts = data.get("entity_counts", [])
    event_counts = data.get("event_counts", [])
    if entity_counts:
        steps_e = [x[0] for x in entity_counts]
        ax2.plot(steps_e, [x[1] for x in entity_counts], label="Entities", color="#ffb74d")
        ax2.plot(steps_e, [x[1] for x in event_counts], label="Events", color="#ef5350")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Count")
    ax2.set_title("Entity & Event Counts")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Location confidence
    ax3 = axes[1, 0]
    loc_conf = data.get("location_confidence", [])
    if loc_conf:
        steps_l = [x[0] for x in loc_conf]
        ax3.fill_between(steps_l, [x[1] for x in loc_conf], alpha=0.3, color="#81c784")
        ax3.plot(steps_l, [x[1] for x in loc_conf], color="#81c784")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Confidence")
    ax3.set_title("Location Confidence")
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Location distribution
    ax4 = axes[1, 1]
    loc_dist = data.get("location_distribution", {})
    if loc_dist:
        labels = list(loc_dist.keys())
        values = list(loc_dist.values())
        colors = ["#4fc3f7", "#81c784", "#ffb74d", "#ba68c8", "#ef5350"]
        ax4.pie(values, labels=labels, autopct="%1.1f%%", colors=colors[:len(labels)])
    ax4.set_title("Location Distribution")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {output_path}")
    else:
        plt.show()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main() -> None:
    """CLI entry point for visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Visualize episodic memory agent runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Print run summary
    python -m episodic_agent.visualize runs/20260202_174416
    
    # Generate HTML report
    python -m episodic_agent.visualize runs/20260202_174416 --html
    
    # Export CSV for external analysis
    python -m episodic_agent.visualize runs/20260202_174416 --csv
    
    # Generate matplotlib plots
    python -m episodic_agent.visualize runs/20260202_174416 --plot
        """
    )
    
    parser.add_argument(
        "run_path",
        help="Path to run directory containing run.jsonl",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate interactive HTML report",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Export data to CSV",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate matplotlib plots (requires matplotlib)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Custom output path",
    )
    
    args = parser.parse_args()
    
    viz = RunVisualizer(args.run_path)
    viz.load()
    
    # Always print summary
    viz.print_summary()
    
    if args.html:
        output = Path(args.output) if args.output else None
        viz.generate_html_report(output)
    
    if args.csv:
        output = Path(args.output) if args.output else None
        viz.export_csv(output)
    
    if args.plot:
        output = Path(args.output) if args.output else None
        plot_with_matplotlib(args.run_path, output)


if __name__ == "__main__":
    main()
