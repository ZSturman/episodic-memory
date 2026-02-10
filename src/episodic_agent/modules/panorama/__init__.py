"""Panorama / video harness for perception-driven localization.

This module provides a standalone test environment that allows the agent
to infer location purely from visual input — panoramic images or video.
No Unity dependency. No hidden ground-truth identifiers.

Components:
- PanoramaSensorProvider: Loads images from a folder, simulates "looking around"
- PanoramaFeatureExtractor: Hand-crafted feature extraction (color, edge, brightness)
- PanoramaPerception: Converts viewport crops into Percept with scene embeddings
- SaccadePolicy: Controls where the agent looks next (scan, investigate, confirm)
- TerminalDebugger: Rich terminal output showing evidence and hypotheses
- DebugServer: Web-based debug UI for deep inspection
- PanoramaEventBus: Structured event backbone for observability
- InvestigationStateMachine: Adaptive label-gating logic
- PanoramaAPIServer: JSON API for the Next.js dashboard
- HexGrid / HexCell: Axial-coordinate hex overlay for structured image scanning
- HexFeatureExtractor / HexScanner: Multi-pass adaptive hex-grid scanning
- AgentControl: Thread-safe pause/step/advance control interface
"""

from episodic_agent.modules.panorama.sensor_provider import PanoramaSensorProvider
from episodic_agent.modules.panorama.feature_extractor import PanoramaFeatureExtractor
from episodic_agent.modules.panorama.perception import PanoramaPerception
from episodic_agent.modules.panorama.saccade import SaccadePolicy
from episodic_agent.modules.panorama.event_bus import PanoramaEventBus
from episodic_agent.modules.panorama.investigation import InvestigationStateMachine
from episodic_agent.modules.panorama.hex_grid import HexCoord, build_hex_grid, hex_size_from_image
from episodic_agent.modules.panorama.hex_cell import HexCell, FocusProfile
from episodic_agent.modules.panorama.hex_feature_extractor import HexFeatureExtractor, HexScanResult
from episodic_agent.modules.panorama.hex_scanner import HexScanner, ScanState
from episodic_agent.modules.panorama.agent_control import AgentControl, UserResponse

__all__ = [
    "PanoramaSensorProvider",
    "PanoramaFeatureExtractor",
    "PanoramaPerception",
    "SaccadePolicy",
    "PanoramaEventBus",
    "InvestigationStateMachine",
    "HexCoord",
    "build_hex_grid",
    "hex_size_from_image",
    "HexCell",
    "FocusProfile",
    "HexFeatureExtractor",
    "HexScanResult",
    "HexScanner",
    "ScanState",
    "AgentControl",
    "UserResponse",
]
