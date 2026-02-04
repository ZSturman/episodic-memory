# Architectural Invariants

This document specifies the **non-negotiable constraints** that define the episodic memory architecture. These invariants MUST be preserved in all implementations—Python backend, Unity sensor, and any future sensor integrations.

## Purpose

These invariants exist to ensure the system:
1. Learns from experience rather than encoding assumptions
2. Treats sensors as interchangeable input sources
3. Preserves entity identity through consolidation
4. Maintains reproducible, auditable behavior

**If you're making changes that might violate an invariant, STOP and reconsider.**

---

## Table of Contents

1. [No Pre-Wired Semantics](#invariant-1-no-pre-wired-semantics)
2. [Labels Come From Users](#invariant-2-labels-come-from-users)
3. [Protocol is Sensor-Agnostic](#invariant-3-protocol-is-sensor-agnostic)
4. [Relative Position Over Categories](#invariant-4-relative-position-over-categories)
5. [ACF Stability Over Perceptual Variation](#invariant-5-acf-stability-over-perceptual-variation)
6. [Motion Advisory, Perception Authoritative](#invariant-6-motion-advisory-perception-authoritative)
7. [Consolidation Preserves Identity](#invariant-7-consolidation-preserves-identity)
8. [Relabeling is Additive](#invariant-8-relabeling-is-additive)
9. [Salience is Learned](#invariant-9-salience-is-learned)
10. [All Decisions Are Logged](#invariant-10-all-decisions-are-logged)

---

## Invariant 1: No Pre-Wired Semantics

### Statement
The system MUST NOT contain hardcoded semantic categories, room types, object names, or domain-specific knowledge.

### Rationale
Pre-wired semantics create assumptions that fail when reality differs. A "kitchen" might have a couch; a "chair" might be used as a shelf. The system must discover what things are through experience, not assume.

### What This Means
- **NO** `ROOM_KITCHEN = "kitchen"` constants with special handling
- **NO** `if object_type == "door":` branches
- **NO** predefined event types like `DOOR_OPENED`
- **YES** structural edge types: `LOCATED_IN`, `RELATED_TO`, `SIMILAR_TO`
- **YES** event types: `appeared`, `disappeared`, `state_changed`, `moved`

### Implementation References
- [schemas/graph.py](../src/episodic_agent/schemas/graph.py) - Edge types are structural only
- [schemas/events.py](../src/episodic_agent/schemas/events.py) - Event types are structural
- [modules/event_pipeline.py](../src/episodic_agent/modules/event_pipeline.py) - No predefined semantic events

### Verification Questions
- [ ] Can I grep the codebase and find no hardcoded room/object names?
- [ ] Do all classifications emerge from user labels, not constants?
- [ ] Would this work for a warehouse as well as a home?

---

## Invariant 2: Labels Come From Users

### Statement
The ONLY way semantic labels (names, categories, descriptions) enter the system is through explicit user input via the `label_request`/`label_response` protocol.

### Rationale
An agent cannot know what you call things. Your "office" is your "office" because you said so, not because it contains a desk. This ensures the system's knowledge reflects the user's mental model.

### What This Means
- **NO** auto-labeling based on detected objects
- **NO** inferring "this must be a kitchen" from appliances
- **NO** labels from sensor metadata
- **YES** `LabelRequest` sent to user with visual context
- **YES** `LabelResponse` with user's chosen label
- **YES** confidence-based flows: high→auto-accept, medium→confirm, low→request

### Implementation References
- [schemas/protocol.py](../src/episodic_agent/schemas/protocol.py) - `LabelRequest`, `LabelResponse` messages
- [schemas/labels.py](../src/episodic_agent/schemas/labels.py) - Label storage schema
- [modules/event_pipeline.py](../src/episodic_agent/modules/event_pipeline.py) - Confidence-based label requests

### Verification Questions
- [ ] Is every semantic label traceable to a user interaction?
- [ ] Does the code ever assign a label without user input?
- [ ] Can the system function with all entities labeled "unknown"?

---

## Invariant 3: Protocol is Sensor-Agnostic

### Statement
The wire protocol between sensor and backend MUST work with ANY sensor type—Unity simulation, physical robot, smartphone camera, or future devices.

### Rationale
The architecture must not be "Unity-dependent." Today's Unity sim becomes tomorrow's robot becomes next year's AR glasses. The protocol must be portable.

### What This Means
- **NO** Unity-specific shortcuts (`transform.position`, `Physics.Raycast`)
- **NO** absolute world coordinates crossing the wire
- **NO** reliance on Unity scene hierarchy
- **YES** relative positions (meters from agent)
- **YES** normalized feature vectors
- **YES** standard message formats (JSON, protobuf-ready)

### Implementation References
- [schemas/protocol.py](../src/episodic_agent/schemas/protocol.py) - All protocol messages
- [schemas/spatial.py](../src/episodic_agent/schemas/spatial.py) - Relative position schemas
- [modules/sensor_gateway/](../src/episodic_agent/modules/sensor_gateway/) - Sensor abstraction

### Verification Questions
- [ ] Could a physical robot sensor implement this protocol?
- [ ] Are all coordinates relative to the agent, not world origin?
- [ ] Does the protocol assume any Unity-specific features?

---

## Invariant 4: Relative Position Over Categories

### Statement
Entity identity hypothesis MUST use relative spatial position, NOT object categories or visual similarity alone.

### Rationale
A blue mug that becomes a red mug (same spot) is likely the SAME mug with changed state. Two identical red mugs in different spots are DIFFERENT mugs. Position is a stronger identity signal than appearance.

### What This Means
- **NO** "this looks like a mug, so it IS the same mug"
- **NO** category-based identity matching
- **YES** "entity at position (1.2m, -0.3m) matches prior entity at (1.1m, -0.4m)"
- **YES** visual similarity as supporting evidence, not primary signal

### Implementation References
- [memory/cued_recall.py](../src/episodic_agent/memory/cued_recall.py) - `EntityHypothesisTracker`
- [schemas/salience.py](../src/episodic_agent/schemas/salience.py) - `EntityHypothesis`
- [schemas/spatial.py](../src/episodic_agent/schemas/spatial.py) - Position calculations

### Verification Questions
- [ ] Does same-entity detection prioritize position over appearance?
- [ ] Would changing an object's color break identity tracking?
- [ ] Is visual similarity used as evidence, not as the primary matcher?

---

## Invariant 5: ACF Stability Over Perceptual Variation

### Statement
Perceptual variation (lighting changes, shadows, clutter) MUST NOT fragment location identity. The Active Context Frame (ACF) must be stable against visual noise.

### Rationale
Walking from window to corner changes lighting dramatically. Dropping a bag adds clutter. These are NOT location changes. The system must distinguish real location transitions from perceptual noise.

### What This Means
- **NO** new location when lighting changes
- **NO** fragmentation from shadow movement
- **NO** splitting locations because furniture moved
- **YES** fingerprint-based stability checks
- **YES** "same location, lighting variation" as stable state

### Implementation References
- [modules/acf_stability.py](../src/episodic_agent/modules/acf_stability.py) - `ACFStabilityGuard`, `ACFFingerprint`
- [docs/modules/BOUNDARY.md](../docs/modules/BOUNDARY.md) - Stability rules

### Verification Questions
- [ ] Does walking past a window create a new location? (Should NOT)
- [ ] Does dramatic lighting change fragment the location? (Should NOT)
- [ ] Is the location fingerprint immune to transient visual changes?

---

## Invariant 6: Motion Advisory, Perception Authoritative

### Statement
Motion sensors ADVISE on location transitions; visual perception is AUTHORITATIVE. When they conflict, enter uncertainty state—don't auto-relabel.

### Rationale
Odometry drifts. Elevators move without visual change. Teleportation (in games) breaks motion assumptions. The visual scene is ground truth; motion is a hint.

### What This Means
- **NO** auto-labeling location based on motion alone
- **NO** trusting odometry over visual evidence
- **YES** motion suggests new location → check visual confirmation
- **YES** visual match + motion discontinuity → uncertainty state
- **YES** require additional evidence before location change

### Implementation References
- [modules/arbitrator.py](../src/episodic_agent/modules/arbitrator.py) - `MotionPerceptionArbitrator`
- [modules/acf_stability.py](../src/episodic_agent/modules/acf_stability.py) - Uncertainty states

### Verification Questions
- [ ] Does elevator travel auto-relabel location? (Should NOT)
- [ ] Does visual match + motion conflict trigger uncertainty? (Should YES)
- [ ] Is additional evidence required before confirming location change?

---

## Invariant 7: Consolidation Preserves Identity

### Statement
Graph consolidation (merging nodes) MUST preserve all identity information. Merged nodes retain labels as aliases.

### Rationale
The system might later discover two "separate" entities were the same thing. When merging, we can't throw away information—the old labels become aliases for the merged entity.

### What This Means
- **NO** dropping labels during merge
- **NO** "lossy" consolidation that discards history
- **YES** merged node has `aliases: ["old_label_1", "old_label_2"]`
- **YES** all edges from merged nodes transfer to survivor

### Implementation References
- [modules/consolidation.py](../src/episodic_agent/modules/consolidation.py) - `ConsolidationModule`, `MergeOperation`
- [memory/stubs/graph_store.py](../src/episodic_agent/memory/stubs/graph_store.py) - `update_node()`, `remove_node()`

### Verification Questions
- [ ] Does merging two nodes preserve both labels as aliases?
- [ ] Are all edges from the merged-away node transferred?
- [ ] Is merge operation reversible in principle (no data loss)?

---

## Invariant 8: Relabeling is Additive

### Statement
When a user provides a new label, the old label becomes an alias—it is NOT deleted.

### Rationale
Users change their minds. "The chair" becomes "Dad's chair" becomes "the old chair." All names are valid; the new one is just primary. Historical references must still work.

### What This Means
- **NO** overwriting labels
- **NO** deleting old label on rename
- **YES** `primary_label` + `aliases` list
- **YES** old labels searchable for recall

### Implementation References
- [modules/consolidation.py](../src/episodic_agent/modules/consolidation.py) - `RelabelOperation`
- [schemas/labels.py](../src/episodic_agent/schemas/labels.py) - Label assignment with history

### Verification Questions
- [ ] Does renaming "mug" to "coffee mug" keep "mug" as alias?
- [ ] Can you recall memories using an old/alias label?
- [ ] Is label history preserved for audit?

---

## Invariant 9: Salience is Learned

### Statement
Salience weights start at 0.0 and are boosted through experience (user labels, prediction errors, novelty). Nothing is pre-important.

### Rationale
The system cannot know what's important to you. Your coffee mug might be more salient than the expensive vase. Importance is personal and learned.

### What This Means
- **NO** predefined "important" categories
- **NO** boosting salience based on object type
- **YES** `salience_weights` start at 0.0 for all entities
- **YES** user labeling boosts salience
- **YES** prediction errors (surprises) boost salience
- **YES** novelty detection boosts salience

### Implementation References
- [schemas/salience.py](../src/episodic_agent/schemas/salience.py) - `SalienceWeights`, `DEFAULT_CUE_WEIGHTS`
- [schemas/graph.py](../src/episodic_agent/schemas/graph.py) - `GraphEdge.salience`
- [modules/event_pipeline.py](../src/episodic_agent/modules/event_pipeline.py) - Salience computation

### Verification Questions
- [ ] Do all entities start with 0.0 salience?
- [ ] Does user labeling boost salience?
- [ ] Is there NO predefined "important" category?

---

## Invariant 10: All Decisions Are Logged

### Statement
Every significant decision (label assignment, location change, consolidation, arbitration) MUST be logged with timestamp, category, and context for post-run analysis.

### Rationale
The system must be debuggable, auditable, and reproducible. If you can't see why a decision was made, you can't fix bugs or verify correctness.

### What This Means
- **NO** silent state changes
- **NO** decisions without log entries
- **YES** structured logging with categories: `[MEMORY]`, `[RECOGNITION]`, `[ARBITRATION]`, etc.
- **YES** both human-readable (`.log`) and machine-parseable (`.jsonl`) formats
- **YES** `check_invariant()` calls logged when triggered

### Implementation References
- [utils/logging.py](../src/episodic_agent/utils/logging.py) - `StructuredLogger`, `SessionLogger`, `LogCategory`
- [runs/](../runs/) - Session logs directory

### Log Categories
| Category | Description |
|----------|-------------|
| `SENSOR` | Raw sensor data reception |
| `MEMORY` | Graph operations (add/update/remove nodes/edges) |
| `RECOGNITION` | Entity/location recognition decisions |
| `CONSOLIDATION` | Background graph maintenance |
| `LABEL` | Label assignments and changes |
| `SPATIAL` | Position and spatial relation calculations |
| `PROTOCOL` | Wire protocol messages |
| `ARBITRATION` | Motion/perception conflict resolution |
| `VISUAL` | Visual channel operations |
| `EVENT` | Event detection and learning |
| `RECALL` | Memory retrieval operations |
| `HYPOTHESIS` | Entity/location hypothesis tracking |
| `INVARIANT` | Invariant checks and violations |
| `SYSTEM` | System-level events |

### Verification Questions
- [ ] Can you trace why any label was assigned?
- [ ] Do arbitration decisions appear in logs?
- [ ] Are logs parseable for automated analysis?

---

## Enforcement

### Code Review Checklist

Before merging any PR, verify:

1. **No new hardcoded semantics** (Invariant 1)
2. **Labels require user input path** (Invariant 2)
3. **No Unity-specific protocol features** (Invariant 3)
4. **Position used for identity, not categories** (Invariant 4)
5. **Perceptual variation doesn't fragment locations** (Invariant 5)
6. **Motion/perception conflicts enter uncertainty** (Invariant 6)
7. **Merges preserve all labels** (Invariant 7)
8. **Relabeling adds aliases** (Invariant 8)
9. **Salience starts at 0.0** (Invariant 9)
10. **Significant decisions are logged** (Invariant 10)

### Automated Checks

The test suite includes invariant verification:
- `test_no_predefined_semantics` - Invariant 1
- `test_labels_require_user_input` - Invariant 2
- `test_protocol_is_sensor_agnostic` - Invariant 3
- `test_position_based_identity` - Invariant 4
- `test_acf_stability_against_lighting` - Invariant 5
- `test_arbitration_enters_uncertainty` - Invariant 6
- `test_merge_preserves_labels` - Invariant 7
- `test_relabel_creates_alias` - Invariant 8
- `test_salience_starts_at_zero` - Invariant 9
- `test_decisions_are_logged` - Invariant 10

---

## History

| Date | Change | Author |
|------|--------|--------|
| Phase 9 | Initial document | System |

---

## See Also

- [DATA_CONTRACTS.md](./architecture/DATA_CONTRACTS.md) - Schema definitions
- [INTERFACES.md](./architecture/INTERFACES.md) - Module interfaces
- [PROTOCOL.md](../UnitySensorSim/protocol/README.md) - Wire protocol spec
