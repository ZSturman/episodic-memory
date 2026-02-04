# Documentation Index

This directory contains comprehensive documentation for the Episodic Memory Agent system.

## Quick Navigation

### Getting Started
- [Main README](../README.md) - Project overview and quick start

### Architecture
- [Architecture Overview](architecture/README.md) - System design and cognitive loop
- [Module Interfaces](architecture/INTERFACES.md) - Abstract base classes for all modules
- [Data Contracts](architecture/DATA_CONTRACTS.md) - Pydantic schema definitions

### Unity Integration
- [Unity Setup Guide](unity/SETUP.md) - Complete guide for setting up Unity simulation
- [Unity Protocol](unity/PROTOCOL.md) - WebSocket communication protocol

### Modules
- [Core Modules](modules/README.md) - Overview of all module implementations
- [Memory Stores](modules/MEMORY.md) - Episode and graph storage
- [Boundary Detection](modules/BOUNDARY.md) - Episode segmentation algorithms

### Analysis & Visualization
- [Visualization Guide](VISUALIZATION.md) - Plotting and analyzing run data

### Testing & Scenarios
- [Scenarios Guide](scenarios/README.md) - Automated test scenarios
- [Creating Custom Scenarios](scenarios/CUSTOM.md) - How to write your own scenarios

### Reference
- [Troubleshooting](TROUBLESHOOTING.md) - Common issues and solutions
- [CLI Reference](CLI.md) - Command-line interface documentation

## Documentation Structure

```
docs/
├── README.md                 # This file
├── TROUBLESHOOTING.md        # Common issues and solutions
├── CLI.md                    # CLI command reference
├── VISUALIZATION.md          # Analysis and plotting guide
├── architecture/
│   ├── README.md             # System architecture overview
│   ├── INTERFACES.md         # Module interface definitions
│   └── DATA_CONTRACTS.md     # Schema documentation
├── unity/
│   ├── SETUP.md              # Unity project setup
│   └── PROTOCOL.md           # Communication protocol
├── modules/
│   ├── README.md             # Module overview
│   ├── MEMORY.md             # Memory stores documentation
│   └── BOUNDARY.md           # Boundary detection
└── scenarios/
    ├── README.md             # Scenario framework
    └── CUSTOM.md             # Custom scenario creation
```
