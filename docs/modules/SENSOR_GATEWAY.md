# Sensor Gateway Module

The Sensor Gateway is a modular, extensible system for handling sensor data from multiple sources. It provides validation, error handling, and conversion to a universal message format.

## Overview

The gateway is designed to answer three fundamental questions:
1. **"Where am I?"** - Spatial/location context
2. **"What's around?"** - Entity/object awareness  
3. **"What's happening/can happen?"** - Events and predictions

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENSOR GATEWAY                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │  Unity   │   │  LiDAR   │   │  Audio   │   │  Custom  │     │
│  │  Sensor  │   │  Sensor  │   │  Sensor  │   │  Sensor  │     │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘     │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   VALIDATORS                             │   │
│  │  UnityValidator | LidarValidator | AudioValidator | ...  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   HANDLERS                               │   │
│  │  UnitySensorHandler | LidarHandler | AudioHandler | ... │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           │                                     │
│                           ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                UNIVERSAL SensorMessage                   │   │
│  │  • LocationContext (where am I?)                        │   │
│  │  • EntityObservations (what's around?)                  │   │
│  │  • EventObservations (what's happening?)                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from episodic_agent.modules.sensor_gateway import SensorGateway, SensorType

# Create gateway
gateway = SensorGateway()

# Register sensors
gateway.register_sensor("unity_main", SensorType.UNITY_WEBSOCKET)
gateway.register_sensor("lidar_front", SensorType.LIDAR)

# Process incoming data
message = gateway.process(raw_data, sensor_id="unity_main")

# Check validation status
if message.validation and not message.validation.is_valid:
    print(f"Validation issues: {message.validation.error_summary}")
    for error in message.validation.errors:
        print(f"  - {error.code}: {error.message}")

# Access the three fundamental answers
print(f"Location: {message.location}")
print(f"Entities: {len(message.entities)}")
print(f"Events: {len(message.events)}")
```

### With User Notifications

```python
def notify_user(severity: str, message: str):
    """Custom notification handler."""
    if severity == "CRITICAL":
        print(f"🚨 CRITICAL: {message}")
    elif severity == "ERROR":
        print(f"❌ ERROR: {message}")
    else:
        print(f"ℹ️ INFO: {message}")

gateway = SensorGateway(
    on_user_notification=notify_user,
    log_raw_data=True,  # Enable detailed logging
)
```

### Debugging Unity Communication

Enable the `--log-sensor-data` flag when running the CLI:

```bash
python -m episodic_agent run --profile unity_full \
    --unity-ws ws://localhost:8765 \
    --log-sensor-data \
    --verbose
```

This will show:
- Raw JSON data received from Unity
- Validation results and any auto-corrections
- Detailed error messages with suggestions

## Supported Sensor Types

| Type | Validator | Handler | Capabilities |
|------|-----------|---------|--------------|
| `UNITY_WEBSOCKET` | UnityValidator | UnitySensorHandler | Location, Pose, Room ID, Entities, State Changes |
| `LIDAR` | LidarValidator | LidarSensorHandler | Point Cloud, Depth |
| `MICROPHONE` | AudioValidator | AudioSensorHandler | Audio, Events (speech/sounds) |
| `GPS` | GenericValidator | GenericSensorHandler | Location |
| `IMU` | GenericValidator | GenericSensorHandler | Odometry |
| `UNKNOWN` | GenericValidator | GenericSensorHandler | Minimal |

## Extending the Gateway

### Adding a New Sensor Type

1. **Define the sensor type** in `types.py`:
```python
class SensorType(str, Enum):
    MY_SENSOR = "my_sensor"
```

2. **Create a validator** in `validators.py`:
```python
class MySensorValidator(SensorValidator):
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.MY_SENSOR
    
    def validate(self, data: dict[str, Any]) -> ValidationResult:
        # Implement validation logic
        ...
```

3. **Create a handler** in `handlers.py`:
```python
class MySensorHandler(SensorHandler):
    @property
    def sensor_type(self) -> SensorType:
        return SensorType.MY_SENSOR
    
    @property
    def capabilities(self) -> set[SensorCapability]:
        return {SensorCapability.PROVIDES_...}
    
    def process(self, data, validation, sensor_id) -> SensorMessage:
        # Extract location, entities, events
        ...
```

4. **Register in the registries**:
```python
VALIDATORS[SensorType.MY_SENSOR] = MySensorValidator
HANDLERS[SensorType.MY_SENSOR] = MySensorHandler
```

## Validation Errors

The gateway tracks validation errors with severity levels:

- **INFO**: Informational, auto-corrected without issue
- **WARNING**: Potential issue, but data can be used
- **ERROR**: Invalid data, may be partially corrected
- **CRITICAL**: Cannot process, requires user intervention

### Common Error Codes

| Code | Severity | Meaning |
|------|----------|---------|
| `MISSING_REQUIRED_FIELD` | ERROR | Required field not present |
| `INVALID_TYPE` | ERROR | Field has wrong type |
| `PROTOCOL_MISMATCH` | WARNING | Version mismatch with sensor |
| `AUTO_CORRECTED` | INFO | Field was automatically corrected |
| `JSON_PARSE_ERROR` | CRITICAL | Invalid JSON received |

## Statistics and Monitoring

```python
# Get gateway statistics
stats = gateway.get_stats()
print(f"Total messages: {stats['total_messages']}")
print(f"Error rate: {stats['error_rate']:.1%}")

# Get error history
errors = gateway.get_error_history(sensor_id="unity_main")
for error in errors[-10:]:
    print(f"{error.code}: {error.message}")

# Query sensors by capability
location_sensors = gateway.get_sensors_by_capability(
    SensorCapability.PROVIDES_LOCATION
)
```
