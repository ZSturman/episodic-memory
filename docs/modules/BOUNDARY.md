# Boundary Detection

Documentation for episode boundary detection algorithms.

## Overview

Episode boundaries determine when to "freeze" the Active Context Frame into a stored Episode. The boundary detector balances:

- **Temporal continuity** - Episodes shouldn't be too short
- **Event salience** - Important events should trigger boundaries
- **Location coherence** - Location changes are natural boundaries
- **Prediction error** - Unexpected events may indicate boundaries

---

## Boundary Reasons

| Reason | Description | Priority |
|--------|-------------|----------|
| `location_change` | Entered new location | High |
| `time_interval` | Elapsed step threshold | Low |
| `salient_event` | Important event occurred | Medium |
| `prediction_error` | High prediction error | Medium |
| `manual` | User-triggered | High |

---

## Implementations

### StubBoundaryDetector

Simple time-based detector for testing.

```python
class StubBoundaryDetector(BoundaryDetector):
    def __init__(self, freeze_interval: int = 50):
        self._interval = freeze_interval
    
    def check(self, acf: ActiveContextFrame) -> tuple[bool, str | None]:
        if acf.step_count >= self._interval:
            return True, "time_interval"
        return False, None
```

**Use case**: Testing, development

---

### HysteresisBoundaryDetector

Production detector with multiple factors and hysteresis.

```python
class HysteresisBoundaryDetector(BoundaryDetector):
    def __init__(
        self,
        high_threshold: float = 0.8,
        low_threshold: float = 0.3,
        time_interval: int = 100,
        location_weight: float = 1.0,
        event_weight: float = 0.5,
        prediction_weight: float = 0.3,
    ):
        ...
```

#### Hysteresis Behavior

```
                    High threshold (0.8)
                         ─────────────────────────
Boundary score    ▲      /                       \
                  │     /                         \
                  │    /                           \
                  │   /                             \
                  │──/─────────────────────────────────────
                         Low threshold (0.3)
                  └─────────────────────────────────────────▶
                                  Time

        Zone A: Below low threshold (stable)
        Zone B: Between thresholds (hysteresis)
        Zone C: Above high threshold (trigger)
```

**Hysteresis prevents oscillation**:
- Score must exceed `high_threshold` to trigger
- Must drop below `low_threshold` before triggering again
- Prevents rapid freeze/unfreeze cycles

#### Scoring Algorithm

```python
def _compute_score(self, acf: ActiveContextFrame) -> float:
    score = 0.0
    
    # Location change (binary, highest impact)
    if self._location_changed(acf):
        score += self.location_weight
    
    # Salient events (weighted by count)
    salient_count = self._count_salient_events(acf)
    score += min(salient_count * 0.2, 1.0) * self.event_weight
    
    # Prediction error (continuous)
    pred_error = self._get_prediction_error(acf)
    score += pred_error * self.prediction_weight
    
    # Time factor (gradual increase)
    time_factor = min(acf.step_count / self.time_interval, 1.0)
    score += time_factor * 0.2
    
    return min(score, 1.0)
```

---

## Configuration

### CLI Options

```bash
python -m episodic_agent run \
    --freeze-interval 100 \
    --profile unity_full
```

### Profile Parameters

```python
UNITY_FULL_PROFILE = ProfileConfig(
    ...
    boundary_detector="HysteresisBoundaryDetector",
    parameters={
        "boundary_high_threshold": 0.8,
        "boundary_low_threshold": 0.3,
        "boundary_time_interval": 100,
    },
)
```

---

## Factor Details

### Location Change

**Weight**: 1.0 (highest)

Immediate boundary when entering a new location:

```python
def _location_changed(self, acf: ActiveContextFrame) -> bool:
    current = acf.location_label
    previous = self._previous_location
    self._previous_location = current
    
    if previous is None:
        return False
    
    return current != previous
```

**Why high weight**: Locations provide natural episode structure (like chapters in a story).

---

### Time Interval

**Weight**: 0.2 (background)

Gradual increase over time:

```python
time_factor = min(acf.step_count / self.time_interval, 1.0)
score += time_factor * 0.2
```

**Behavior**:
- Steps 0-50: Factor 0.0-0.5 → Score contribution 0.0-0.1
- Steps 50-100: Factor 0.5-1.0 → Score contribution 0.1-0.2

**Why low weight**: Time alone shouldn't trigger boundaries, but prevents infinitely long episodes.

---

### Salient Events

**Weight**: 0.5 (medium)

Events like state changes can trigger boundaries:

```python
def _count_salient_events(self, acf: ActiveContextFrame) -> int:
    salient_types = {"state_change", "appearance", "disappearance"}
    return sum(1 for e in acf.events if e.get("type") in salient_types)
```

**Scoring**:
- 1 event: +0.1 (0.5 × 0.2)
- 5 events: +0.5 (0.5 × 1.0, capped)

**Why medium weight**: Events are meaningful but shouldn't cause constant boundaries.

---

### Prediction Error

**Weight**: 0.3 (medium)

High prediction error suggests unexpected situation:

```python
def _get_prediction_error(self, acf: ActiveContextFrame) -> float:
    return acf.extras.get("prediction_error", 0.0)
```

**What increases prediction error**:
- Missing expected entities
- Unexpected state changes
- Novel entities at familiar location

**Why medium weight**: Unexpected situations may indicate new episode, but could also be noise.

---

## Tuning Guidelines

### For Shorter Episodes

```python
parameters={
    "boundary_high_threshold": 0.6,
    "boundary_low_threshold": 0.2,
    "boundary_time_interval": 50,
}
```

### For Longer Episodes

```python
parameters={
    "boundary_high_threshold": 0.9,
    "boundary_low_threshold": 0.4,
    "boundary_time_interval": 200,
}
```

### For Event-Heavy Scenarios

```python
parameters={
    "boundary_high_threshold": 0.8,
    "event_weight": 0.7,
    "prediction_weight": 0.5,
}
```

---

## Debugging

### Enable Boundary Logging

```bash
python -m episodic_agent run --verbose
```

Outputs:
```
[DEBUG] Boundary score: 0.45 (loc=0.0, evt=0.2, pred=0.1, time=0.15)
[DEBUG] Boundary score: 0.85 (loc=1.0, evt=0.0, pred=0.0, time=0.05) -> TRIGGER
```

### Inspect ACF Extras

```python
# In orchestrator or custom module
print(f"Prediction error: {acf.extras.get('prediction_error', 0)}")
print(f"Recent events: {len(acf.events)}")
```

---

## Custom Boundary Detector

```python
from episodic_agent.core.interfaces import BoundaryDetector
from episodic_agent.schemas import ActiveContextFrame

class MyBoundaryDetector(BoundaryDetector):
    def __init__(self, custom_threshold: float = 0.7):
        self._threshold = custom_threshold
        self._event_count = 0
    
    def check(
        self,
        acf: ActiveContextFrame,
    ) -> tuple[bool, str | None]:
        # Custom logic
        self._event_count += len(acf.events)
        
        if self._event_count >= 10:
            self._event_count = 0
            return True, "event_accumulation"
        
        return False, None
```

Register in profiles to use.
