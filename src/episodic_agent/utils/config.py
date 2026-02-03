"""Configuration constants for the episodic memory system."""

# Default embedding dimensionality for perception modules
DEFAULT_EMBEDDING_DIM: int = 128

# Default number of steps between episode freezes
DEFAULT_FREEZE_INTERVAL: int = 50

# Log format version for JSONL records
LOG_VERSION: str = "v1"

# =============================================================================
# Confidence Thresholds
# =============================================================================

# Low confidence threshold - below this, consider "unknown"
CONFIDENCE_T_LOW: float = 0.3

# High confidence threshold - above this, consider "confident"
CONFIDENCE_T_HIGH: float = 0.7

# Minimum confidence delta to trigger location change boundary
LOCATION_CHANGE_CONFIDENCE_DELTA: float = 0.2

# =============================================================================
# Boundary Detection (Hysteresis)
# =============================================================================

# Minimum frames to persist before allowing boundary on location change
BOUNDARY_HYSTERESIS_MIN_FRAMES: int = 10

# Maximum frames before forcing a boundary (timeout)
BOUNDARY_TIMEOUT_FRAMES: int = 200

# =============================================================================
# Label Management
# =============================================================================

# Maximum number of alternative labels per node
MAX_LABELS_PER_NODE: int = 10

# =============================================================================
# Delta Detection (Phase 5)
# =============================================================================

# Movement threshold in units for detecting entity movement
DEFAULT_MOVE_THRESHOLD: float = 0.5

# Steps entity must be absent before considered "missing"
DEFAULT_MISSING_WINDOW: int = 3

# =============================================================================
# Event Detection (Phase 5)
# =============================================================================

# Confidence threshold for event recognition
EVENT_RECOGNITION_THRESHOLD: float = 0.7

# Confidence boost for learned event patterns
LEARNED_EVENT_CONFIDENCE_BOOST: float = 0.15
