"""Configuration for pytest."""

import pytest


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unity: marks tests requiring Unity connection"
    )


@pytest.fixture
def sample_sensor_frame():
    """Create a sample sensor frame for testing."""
    from datetime import datetime
    from episodic_agent.schemas import SensorFrame
    
    return SensorFrame(
        frame_id=1,
        timestamp=datetime.now(),
        sensor_type="test",
        raw_data={"test": True},
        extras={
            "current_room": "room-001",
            "current_room_label": "Test Room",
            "entities": [],
        },
    )


@pytest.fixture
def sample_percept():
    """Create a sample percept for testing."""
    from datetime import datetime
    from episodic_agent.schemas import ObjectCandidate, Percept
    
    return Percept(
        percept_id="test_percept",
        timestamp=datetime.now(),
        scene_embedding=[0.1, 0.2, 0.3],
        objects=[
            ObjectCandidate(
                candidate_id="obj_001",
                label="Test Object",
                confidence=0.9,
            ),
        ],
    )


@pytest.fixture
def sample_acf():
    """Create a sample ACF for testing."""
    from datetime import datetime
    from episodic_agent.schemas import ActiveContextFrame
    
    return ActiveContextFrame(
        acf_id="test_acf",
        location_label="Test Room",
        location_confidence=0.9,
        step_count=5,
    )


@pytest.fixture
def sample_episode():
    """Create a sample episode for testing."""
    from datetime import datetime
    from episodic_agent.schemas import Episode
    
    return Episode(
        episode_id="test_episode",
        source_acf_id="test_acf",
        location_label="Test Room",
        start_time=datetime.now(),
        end_time=datetime.now(),
        step_count=10,
    )
